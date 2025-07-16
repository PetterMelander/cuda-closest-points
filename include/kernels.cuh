#ifndef KERNELS_H
#define KERNELS_H

#include "types.h"
#include <climits>
#include <cooperative_groups.h>

template <typename KernelFunc>
int num_blocks_max_occupancy(KernelFunc kernel, int blockSize,
                             size_t smem_per_block,
                             float oversubscription_factor) {
  int deviceId;
  CUDA_CHECK(cudaGetDevice(&deviceId));

  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, deviceId));

  int max_active_blocks_per_sm;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks_per_sm, kernel, blockSize, smem_per_block));

  int total_blocks = props.multiProcessorCount * max_active_blocks_per_sm;
  return static_cast<int>(total_blocks * oversubscription_factor);
}

__global__ void min_distances_thread_per_a(
    const int *__restrict__ as, const int *__restrict__ bs, int num_as,
    int num_bs, int img_width, MinResult *block_results, bool swapped);

__global__ void min_distances_thread_per_a_int2(
    const int2 *__restrict__ as, const int2 *__restrict__ bs, int num_as,
    int num_bs, int img_width, MinResult *block_results, bool swapped);

__host__ void make_points(int *as, int *bs, int num_as, int num_bs,
                          int img_width, int2 *points_a, int2 *points_b,
                          cudaStream_t stream);

__global__ void min_distances_thread_per_pair(const int2 *__restrict__ points_a,
                                              const int2 *__restrict__ points_b,
                                              int num_as, int num_bs,
                                              int img_width,
                                              MinResult *block_results);

__global__ void final_reduction(MinResult *input, int num_elements,
                                MinResult *output);

__global__ void index_edges(const int *__restrict__ image, int img_height,
                            int img_width, int *g_nonzero_idxs,
                            int *g_nonzero_values, int *g_num_nonzeros);

__host__ dim3 get_block_dims_indexing();

__host__ dim3 get_grid_dims_indexing(int img_height, int img_width);

__host__ int get_block_size_distance();

__host__ int get_grid_size_distance(int num_as);

__host__ dim3 get_block_dims_distance_2d();

__host__ dim3 get_grid_dims_2d(int num_as, int num_bs);

#endif