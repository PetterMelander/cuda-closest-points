#include "../include/host_code.cuh"
#include "../include/kernels.cuh"
#include <algorithm>
#include <cmath>

std::tuple<int, int> launch_index_shapes(const int *const h_image,
                                         const int img_height,
                                         const int img_width, int *d_as,
                                         int *d_bs) {
  int total_pixels = img_height * img_width;
  int *d_image, *d_num_as, *d_num_bs, h_num_as, h_num_bs;
  CUDA_CHECK(cudaMalloc(&d_image, total_pixels * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_num_as, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_num_bs, sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_image, h_image, total_pixels * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_num_as, 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_num_bs, 0, sizeof(int)));

  int num_blocks = num_blocks_max_occupancy(index_shapes, THREADS_PER_BLOCK,
                                            sizeof(int) * 2048 * 2, 2.0f);
  index_shapes<<<num_blocks, THREADS_PER_BLOCK>>>(d_image, total_pixels, d_as,
                                                  d_num_as, d_bs, d_num_bs);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(
      cudaMemcpy(&h_num_as, d_num_as, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(&h_num_bs, d_num_bs, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_image));
  CUDA_CHECK(cudaFree(d_num_as));
  CUDA_CHECK(cudaFree(d_num_bs));

  return std::tuple<int, int>{h_num_as, h_num_bs};
}

Pair launch_min_pair_thread_per_a(int num_as, int num_bs, const int img_width,
                                  int *d_as, int *d_bs) {

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

  // Do final reduction on cpu
  MinResult h_min_results[num_blocks];
  CUDA_CHECK(cudaMemcpy(&h_min_results, d_results,
                        sizeof(MinResult) * num_blocks,
                        cudaMemcpyDeviceToHost));

  MinResult min_result{INT_MAX, -1, -1};
  for (int i = 0; i < num_blocks; ++i) {
    MinResult result = h_min_results[i];
    if (result.distance < min_result.distance)
      min_result = result;
  }

  CUDA_CHECK(cudaFree(d_results));

  if (swapped) {
    std::swap(min_result.a_idx, min_result.b_idx);
    std::swap(num_as, num_bs);
    std::swap(d_as, d_bs);
  }
  Pair result;
  result.distance = sqrt(min_result.distance);
  result.ax = min_result.a_idx % img_width;
  result.ay = min_result.a_idx / img_width;
  result.bx = min_result.b_idx % img_width;
  result.by = min_result.b_idx / img_width;
  return result;
}

Pair launch_min_pair_thread_per_pair(const int num_as, const int num_bs,
                                     const int img_width, int *d_as,
                                     int *d_bs) {

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

  // Do final reduction on cpu
  MinResult h_results[num_blocks];
  CUDA_CHECK(cudaMemcpy(&h_results, d_results, sizeof(MinResult) * num_blocks,
                        cudaMemcpyDeviceToHost));

  MinResult min_result{INT_MAX, -1, -1};
  for (int i = 0; i < num_blocks; ++i) {
    MinResult result = h_results[i];
    if (result.distance < min_result.distance)
      min_result = result;
  }

  int2 h_a, h_b;
  CUDA_CHECK(cudaMemcpy(&h_a, d_points_a + min_result.a_idx, sizeof(int2),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&h_b, d_points_b + min_result.b_idx, sizeof(int2),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_results));
  CUDA_CHECK(cudaFree(d_points_a));
  CUDA_CHECK(cudaFree(d_points_b));

  Pair result;
  result.distance = sqrt(min_result.distance);
  result.ax = h_a.x;
  result.ay = h_a.y;
  result.bx = h_b.x;
  result.by = h_b.y;
  return result;
}

Pair get_min_pair(const int *const h_image, const int img_height,
                  const int img_width) {
  int total_pixels = img_height * img_width;
  int *d_as, *d_bs;
  CUDA_CHECK(cudaMalloc(&d_as, total_pixels * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_bs, total_pixels * sizeof(int)));
  auto segment_sizes =
      launch_index_shapes(h_image, img_height, img_width, d_as, d_bs);
  auto [num_as, num_bs] = segment_sizes;

  if (num_as < 1 || num_bs < 1)
    throw std::runtime_error("Must have at least 1 a and 1 b");

  Pair h_min_pair;
  long long num_pairs = (long long)num_as * (long long)num_bs;
  int max_segment_size = std::max(num_as, num_bs);
  // if (num_pairs > (long long)INT_MAX || max_segment_size > 5000) {
  h_min_pair =
      launch_min_pair_thread_per_a(num_as, num_bs, img_width, d_as, d_bs);
  // } else {
  h_min_pair =
      launch_min_pair_thread_per_pair(num_as, num_bs, img_width, d_as, d_bs);
  // }
  CUDA_CHECK(cudaFree(d_as));
  CUDA_CHECK(cudaFree(d_bs));

  return h_min_pair;
}