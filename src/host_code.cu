#include "../include/kernels.cuh"
#include "../include/host_code.cuh"
#include <algorithm>
#include <tuple>

std::tuple<int, int> launch_index_shapes(int *h_image, int img_height,
                                         int img_width, int *d_as, int *d_bs) {
  int total_pixels = img_height * img_width;
  int *d_image, *d_num_ones, *d_num_twos, h_num_as, h_num_bs;
  cudaMalloc(&d_image, total_pixels * sizeof(int));
  cudaCheckErrors("cudaMalloc d_image error");
  cudaMalloc(&d_num_ones, sizeof(int));
  cudaCheckErrors("cudaMalloc d_num_ones error");
  cudaMalloc(&d_num_twos, sizeof(int));
  cudaCheckErrors("cudaMalloc d_num_twos error");

  cudaMemcpy(d_image, h_image, total_pixels * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy d_image error");
  cudaMemset(d_num_ones, 0, sizeof(int));
  cudaCheckErrors("cudaMemset d_num_ones error");
  cudaMemset(d_num_twos, 0, sizeof(int));
  cudaCheckErrors("cudaMemset d_num_twos error");

  // TODO: separate function for number of blocks
  int device_id;
  cudaGetDevice(&device_id);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device_id);

  const size_t shared_mem_per_block = sizeof(int) * 2048 * 2;

  int max_active_blocks_per_sm;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_sm,
                                                index_shapes, THREADS_PER_BLOCK,
                                                shared_mem_per_block);

  const int oversubscription_factor = 2;
  int num_blocks_indexing = props.multiProcessorCount *
                            max_active_blocks_per_sm * oversubscription_factor;
  index_shapes<<<num_blocks_indexing, THREADS_PER_BLOCK>>>(
      d_image, total_pixels, d_as, d_num_ones, d_bs, d_num_twos);
  cudaCheckErrors("index_shapes kernel launch error");
  cudaMemcpy(&h_num_as, d_num_ones, sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors("cudaMemcpy h_num_ones error");
  cudaMemcpy(&h_num_bs, d_num_twos, sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors("cudaMemcpy h_num_twos error");

  cudaFree(d_image);
  delete[] h_image;

  return std::tuple<int, int>{h_num_as, h_num_bs};
}

Pair launch_min_pair_thread_per_a(int num_as, int num_bs, int img_width,
                                  int *d_as, int *d_bs) {

  // Kernel parallelizes over a's. Launch kernel with the larger mask as a.
  bool swapped = num_as < num_bs;
  if (swapped) {
    std::swap(num_as, num_bs);
    std::swap(d_as, d_bs);
  }

  int num_blocks = (num_as + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  MinResult *d_results;
  cudaMalloc(&d_results, sizeof(MinResult) * num_blocks);
  cudaCheckErrors("cudaMalloc block_results error");

  min_distances_thread_per_one<<<num_blocks, THREADS_PER_BLOCK>>>(
      d_as, d_bs, num_as, num_bs, img_width, d_results);
  cudaCheckErrors("min_distances_thread_per_one kernel launch error");

  final_reduction<<<1, 1024>>>(d_results, num_blocks, d_results);
  cudaCheckErrors("final_reduction kernel launch error");

  MinResult h_min_result;
  cudaMemcpy(&h_min_result, d_results, sizeof(MinResult),
             cudaMemcpyDeviceToHost);
  cudaCheckErrors("h_min_result cudaMemcpy error");

  cudaFree(d_results);
  cudaCheckErrors("d_block_results cudaFree error");

  if (swapped) {
    std::swap(h_min_result.a_idx, h_min_result.b_idx);
  }
  Pair result;
  result.distance = h_min_result.distance;
  result.ax = h_min_result.a_idx % img_width;
  result.ay = h_min_result.a_idx / img_width;
  result.bx = h_min_result.b_idx % img_width;
  result.by = h_min_result.b_idx / img_width;
  return result;
}

Pair launch_min_pair_thread_per_pair(int num_as, int num_bs, int img_width,
                                     int *d_as, int *d_bs) {
  int num_blocks = 256; // TODO: fix

  MinResultSingleIndex *d_results;
  cudaMalloc(&d_results, sizeof(MinResultSingleIndex) * num_blocks);
  cudaCheckErrors("cudaMalloc d_block_results error");

  int2 *d_points_a, *d_points_b;
  cudaMalloc(&d_points_a, sizeof(int2) * num_as);
  cudaCheckErrors("cudaMalloc d_points_a error");
  cudaMalloc(&d_points_b, sizeof(int2) * num_bs);
  cudaCheckErrors("cudaMalloc d_points_b error");
  make_points(d_as, d_bs, num_as, num_bs, img_width, d_points_a, d_points_b);

  min_distances_thread_per_pair<<<num_blocks, THREADS_PER_BLOCK>>>(
      d_points_a, d_points_b, num_as, num_bs, img_width, d_results);
  cudaCheckErrors("min_distances_thread_per_pair kernel launch error");

  final_reduction<<<1, 1024>>>(d_results, num_blocks, d_results);
  cudaCheckErrors("final_reduction kernel launch error");

  MinResultSingleIndex h_min_result;
  cudaMemcpy(&h_min_result, d_results, sizeof(MinResultSingleIndex),
             cudaMemcpyDeviceToHost);
  cudaCheckErrors("h_min_result cudaMemcpy error");

  cudaFree(d_results);
  cudaCheckErrors("d_block_results cudaFree error");
  cudaFree(d_points_a);
  cudaCheckErrors("d_points_a cudaFree error");
  cudaFree(d_points_b);
  cudaCheckErrors("d_points_b cudaFree error");

  Pair result;
  result.distance = h_min_result.distance;
  int a_idx = h_min_result.idx % num_as;
  int b_idx = h_min_result.idx / num_as;
  result.ax = a_idx % img_width;
  result.ay = a_idx / img_width;
  result.bx = b_idx % img_width;
  result.by = b_idx / img_width;
  return result;
}

Pair get_min_pair(int *h_image, int img_height, int img_width) {
  int total_pixels = img_height * img_width;
  int *d_ones, *d_twos;
  cudaMalloc(&d_ones, total_pixels * sizeof(int));
  cudaCheckErrors("cudaMalloc d_ones error");
  cudaMalloc(&d_twos, total_pixels * sizeof(int));
  cudaCheckErrors("cudaMalloc d_twos error");
  auto segment_sizes =
      launch_index_shapes(h_image, img_height, img_width, d_ones, d_twos);
  auto [num_as, num_bs] = segment_sizes;

  Pair h_min_pair;
  long long num_pairs = num_as * num_bs;
  int max_segment_size = std::max(num_as, num_bs);
  if (num_pairs > (long long)INT_MAX || max_segment_size > 10000) {
    h_min_pair =
        launch_min_pair_thread_per_a(num_as, num_bs, img_width, d_ones, d_twos);
  } else {
    h_min_pair = launch_min_pair_thread_per_pair(num_as, num_bs, img_width,
                                                 d_ones, d_twos);
  }
  cudaFree(d_ones);
  cudaCheckErrors("cudaFree d_ones error");
  cudaFree(d_twos);
  cudaCheckErrors("cudaFree d_twos error");

  return h_min_pair;
}