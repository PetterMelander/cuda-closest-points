#ifndef HOST_CODE_H
#define HOST_CODE_H

#include "../include/types.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

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

std::vector<std::vector<Pair>>
get_pairs(const int *const h_image, const int img_height, const int img_width);

std::vector<int> detect_masks(int *d_image, int total_pixels);

#endif