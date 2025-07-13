#include "../include/host_code.cuh"
#include "../include/kernels.cuh"
#include <climits>
#include <cstddef>
#include <cub/cub.cuh>
#include <tuple>
#include <utility>
#include <vector>

using std::vector;

MinResult cpu_reduction(vector<MinResult> results) {
  MinResult min_result{INT_MAX, -1, -1};
  for (MinResult &result : results) {
    if (result.distance < min_result.distance)
      min_result = result;
  }
  return min_result;
}

void sort_nonzeros(int *&d_nonzero_idxs, int *&d_nonzero_values,
                   int h_num_nonzeros) {
  // This code reinterpret casts the ints to unsigned because they are positive
  // anyway, and radix sort is faster for unsigned ints. Besides, we do not care
  // about the order, we just want identical values to be adjacent.

  int *d_idxs_buffer, *d_values_buffer;
  CUDA_CHECK(cudaMalloc(&d_idxs_buffer, sizeof(int) * h_num_nonzeros));
  CUDA_CHECK(cudaMalloc(&d_values_buffer, sizeof(int) * h_num_nonzeros));

  // Create a DoubleBuffer to wrap the pair of device pointers
  cub::DoubleBuffer<unsigned int> d_keys(
      reinterpret_cast<unsigned int *>(d_nonzero_values),
      reinterpret_cast<unsigned int *>(d_values_buffer));

  cub::DoubleBuffer<unsigned int> d_values(
      reinterpret_cast<unsigned int *>(d_nonzero_idxs),
      reinterpret_cast<unsigned int *>(d_idxs_buffer));

  // Determine temporary device storage requirements
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                             d_keys, d_values, h_num_nonzeros,
                                             0, 8));

  // Allocate temporary storage
  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  // Run sorting operation
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                             d_keys, d_values, h_num_nonzeros,
                                             0, 8));
  CUDA_CHECK(cudaFree(d_temp_storage));
  bool output_in_buffer =
      (reinterpret_cast<int *>(d_keys.Current()) == d_values_buffer);
  if (output_in_buffer) {
    std::swap(d_idxs_buffer, d_nonzero_idxs);
    std::swap(d_values_buffer, d_nonzero_values);
  }
  CUDA_CHECK(cudaFree(d_idxs_buffer));
  CUDA_CHECK(cudaFree(d_values_buffer));
}

std::tuple<vector<int>, vector<int>, int>
encode_runs(int *d_sorted_nonzero_values, int num_nonzeros) {
  int *d_unique_values;
  CUDA_CHECK(cudaMalloc(&d_unique_values, sizeof(int) * 256));
  int *d_mask_sizes;
  CUDA_CHECK(cudaMalloc(&d_mask_sizes, sizeof(int) * 256));
  int *d_num_unique_values;
  CUDA_CHECK(cudaMalloc(&d_num_unique_values, sizeof(int)));

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
      d_temp_storage, temp_storage_bytes, d_sorted_nonzero_values,
      d_unique_values, d_mask_sizes, d_num_unique_values, num_nonzeros));

  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
      d_temp_storage, temp_storage_bytes, d_sorted_nonzero_values,
      d_unique_values, d_mask_sizes, d_num_unique_values, num_nonzeros));

  CUDA_CHECK(cudaFree(d_temp_storage));

  int h_num_unique_values;
  CUDA_CHECK(cudaMemcpy(&h_num_unique_values, d_num_unique_values, sizeof(int),
                        cudaMemcpyDeviceToHost));

  vector<int> h_unique_values(h_num_unique_values);
  vector<int> h_mask_sizes(h_num_unique_values);

  CUDA_CHECK(cudaMemcpy(h_unique_values.data(), d_unique_values,
                        sizeof(int) * h_num_unique_values,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_mask_sizes.data(), d_mask_sizes,
                        sizeof(int) * h_num_unique_values,
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_unique_values));
  CUDA_CHECK(cudaFree(d_num_unique_values));
  CUDA_CHECK(cudaFree(d_mask_sizes));

  return std::make_tuple(h_unique_values, h_mask_sizes, h_num_unique_values);
}

std::tuple<vector<int>, vector<int>, int>
index_masks(const int *const h_image, int total_pixels, int *&d_nonzero_idxs,
               int *&d_nonzero_values) {

  int *d_image;
  CUDA_CHECK(cudaMalloc(&d_image, sizeof(int) * total_pixels));
  CUDA_CHECK(cudaMemcpy(d_image, h_image, sizeof(int) * total_pixels,
                        cudaMemcpyHostToDevice));

  // Index shapes
  int *d_num_nonzeros;
  CUDA_CHECK(cudaMalloc(&d_num_nonzeros, sizeof(int)));

  int num_blocks =
      num_blocks_max_occupancy(find_nonzeros, THREADS_PER_BLOCK,
                               sizeof(int) * 2 * TILE_SIZE_INDEXING, 1.5f);
  find_nonzeros<<<num_blocks, THREADS_PER_BLOCK>>>(
      d_image, total_pixels, d_nonzero_idxs, d_nonzero_values, d_num_nonzeros);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaFree(d_image));

  int h_num_nonzeros;
  CUDA_CHECK(cudaMemcpy(&h_num_nonzeros, d_num_nonzeros, sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_num_nonzeros));

  // Sort
  sort_nonzeros(d_nonzero_idxs, d_nonzero_values, h_num_nonzeros);

  // Encode runs
  return encode_runs(d_nonzero_values, h_num_nonzeros);
}

vector<MinResult> launch_min_pair_thread_per_a(int num_as, int num_bs,
                                               const int img_width, int *d_as,
                                               int *d_bs) {

  // Kernel parallelizes over a's. Launch kernel with the larger mask as a.
  bool swapped = num_as < num_bs;
  if (swapped) {
    std::swap(num_as, num_bs);
    std::swap(d_as, d_bs);
  }

  int num_blocks = (num_as + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  MinResult *d_results;
  CUDA_CHECK(cudaMalloc(&d_results, sizeof(MinResult) * num_blocks));

  min_distances_thread_per_a<<<num_blocks, THREADS_PER_BLOCK>>>(
      d_as, d_bs, num_as, num_bs, img_width, d_results);
  CUDA_CHECK(cudaGetLastError());

  // Copy results to vector on host
  vector<MinResult> h_results(num_blocks);
  CUDA_CHECK(cudaMemcpy(h_results.data(), d_results,
                        sizeof(MinResult) * num_blocks,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_results));

  // Swap back
  if (swapped) {
    for (MinResult &result : h_results) {
      std::swap(result.a_idx, result.b_idx);
    }
  }
  return h_results;
}

vector<MinResult> launch_min_pair_thread_per_pair(const int num_as,
                                                  const int num_bs,
                                                  const int img_width,
                                                  int *d_as, int *d_bs) {

  uint block_dim = 16;
  dim3 block_size{block_dim, block_dim};
  int num_blocks = num_blocks_max_occupancy(
      min_distances_thread_per_pair, block_dim * block_dim,
      sizeof(MinResult) * WARP_SIZE, 1.42f);
  uint grid_dim = (uint)sqrt(num_blocks);
  dim3 grid_size{grid_dim, grid_dim};
  num_blocks = (int)grid_dim * (int)grid_dim;

  MinResult *d_results;
  CUDA_CHECK(cudaMalloc(&d_results, sizeof(MinResult) * num_blocks));

  int2 *d_points_a, *d_points_b;
  CUDA_CHECK(cudaMalloc(&d_points_a, sizeof(int2) * num_as));
  CUDA_CHECK(cudaMalloc(&d_points_b, sizeof(int2) * num_bs));
  make_points(d_as, d_bs, num_as, num_bs, img_width, d_points_a, d_points_b);

  min_distances_thread_per_pair<<<grid_size, block_size>>>(
      d_points_a, d_points_b, num_as, num_bs, img_width, d_results);
  CUDA_CHECK(cudaGetLastError());

  // Copy results to vector on host
  vector<MinResult> h_results(num_blocks);
  CUDA_CHECK(cudaMemcpy(h_results.data(), d_results,
                        sizeof(MinResult) * num_blocks,
                        cudaMemcpyDeviceToHost));

  // Unlike the other min_distance kernel whis one returns MinResults with
  // indices indicating at which indices in d_points the coordinates are
  // located. Before returning, get the actual linear indices.
  vector<int2> h_points_a(num_as);
  CUDA_CHECK(cudaMemcpy(h_points_a.data(), d_points_a, sizeof(int2) * num_as,
                        cudaMemcpyDeviceToHost));
  vector<int2> h_points_b(num_bs);
  CUDA_CHECK(cudaMemcpy(h_points_b.data(), d_points_b, sizeof(int2) * num_bs,
                        cudaMemcpyDeviceToHost));

  for (MinResult &result : h_results) {
    int2 a = h_points_a[result.a_idx];
    int2 b = h_points_b[result.b_idx];
    result.a_idx = a.x * img_width + a.y;
    result.b_idx = b.x * img_width + b.y;
  }

  CUDA_CHECK(cudaFree(d_points_a));
  CUDA_CHECK(cudaFree(d_points_b));
  CUDA_CHECK(cudaFree(d_results));

  return h_results;
}

vector<MinResult> get_min_pairs(int *d_as, int *d_bs, int num_as, int num_bs,
                                int img_width) {
  long long num_pairs = (long long)num_as * (long long)num_bs;
  int max_mask_size = std::max(num_as, num_bs);
  if (num_pairs > (long long)INT_MAX || max_mask_size > 5000) {
    return launch_min_pair_thread_per_a(num_as, num_bs, img_width, d_as, d_bs);
  } else {
    return launch_min_pair_thread_per_pair(num_as, num_bs, img_width, d_as,
                                           d_bs);
  }
}

vector<vector<vector<MinResult>>>
initial_pair_reductions(vector<int> unique_values, vector<int> mask_sizes,
                        int num_unique_values, int *d_sorted_idxs,
                        int img_width) {
  vector<vector<vector<MinResult>>> h_unreduced_pixel_pairs(
      num_unique_values, vector<vector<MinResult>>(num_unique_values));
  int a_offset = 0;
  for (int i = 0; i < num_unique_values; ++i) {
    int a = unique_values[i];
    int num_as = mask_sizes[i];
    int *a_ptr = d_sorted_idxs + a_offset;

    int b_offset = a_offset + num_as;
    for (int j = i + 1; j < num_unique_values; ++j) {
      int num_bs = mask_sizes[j];
      int b = unique_values[j];
      int *b_ptr = d_sorted_idxs + b_offset;

      vector<MinResult> h_result_vector =
          get_min_pairs(a_ptr, b_ptr, num_as, num_bs, img_width);

      h_unreduced_pixel_pairs[i][j] = h_result_vector;
      h_unreduced_pixel_pairs[j][i] = h_result_vector;
      b_offset += num_bs;
    }
    a_offset += num_as;
  }
  return h_unreduced_pixel_pairs;
}

vector<vector<Pair>> get_pairs(const int *const h_image, const int img_width,
                               const int img_height) {
  int total_pixels = img_width * img_height;

  int *d_sorted_idxs;
  int *d_sorted_values;
  CUDA_CHECK(cudaMalloc(&d_sorted_idxs, sizeof(int) * total_pixels));
  CUDA_CHECK(cudaMalloc(&d_sorted_values, sizeof(int) * total_pixels));

  // Index all masks
  auto [unique_values, mask_sizes, num_unique_values] =
      index_masks(h_image, total_pixels, d_sorted_idxs, d_sorted_values);
  CUDA_CHECK(cudaFree(d_sorted_values));

  // Calculate (not completely reduced) pixel pairings between all masks
  vector<vector<vector<MinResult>>> h_unreduced_pixel_pairs =
      initial_pair_reductions(unique_values, mask_sizes, num_unique_values,
                              d_sorted_idxs, img_width);
  CUDA_CHECK(cudaFree(d_sorted_idxs));

  // For each mask pair, do final reduction on cpu
  vector<vector<Pair>> pairs(num_unique_values,
                             vector<Pair>(num_unique_values));
  for (int i = 0; i < num_unique_values; ++i) {
    for (int j = i; j < num_unique_values; ++j) {
      MinResult result;
      if (i == j)
        result = MinResult{0, -1, -1};
      else
        result = cpu_reduction(h_unreduced_pixel_pairs[i][j]);
      Pair pair;
      pair.a = unique_values[i];
      pair.ax = result.a_idx % img_width;
      pair.ay = result.a_idx / img_width;
      pair.b = unique_values[j];
      pair.bx = result.b_idx % img_width;
      pair.by = result.b_idx / img_width;
      pair.distance = sqrt(result.distance);

      pairs[i][j] = pair;
      pairs[j][i] = pair.transpose();
    }
  }

  return pairs;
}
