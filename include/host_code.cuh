#ifndef HOST_CODE_H
#define HOST_CODE_H

#include "../include/types.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

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

std::tuple<int, int> launch_index_shapes(const int *const h_image,
                                         const int img_height,
                                         const int img_width, int *d_as,
                                         int *d_bs);

Pair launch_min_pair_thread_per_a(int num_as, int num_bs, const int img_width,
                                  int *d_as, int *d_bs);

Pair launch_min_pair_thread_per_pair(int num_as, int num_bs,
                                     const int img_width, int *d_as, int *d_bs);

Pair get_min_pair(const int *const h_image, const int img_height,
                  const int img_width);

#endif