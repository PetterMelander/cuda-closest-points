#ifndef KERNELS_H
#define KERNELS_H

#include <climits>
#include <cooperative_groups.h>
#include "../include/types.h"

#define THREADS_PER_BLOCK 512
#define TILE_SIZE 2048
#define WARP_SIZE 32

// Error checking macro
#define cudaCheckErrors(msg)                                                   \
  do {                                                                         \
    cudaError_t __err = cudaGetLastError();                                    \
    if (__err != cudaSuccess) {                                                \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg,                  \
              cudaGetErrorString(__err), __FILE__, __LINE__);                  \
      fprintf(stderr, "*** FAILED - ABORTING\n");                              \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__global__ void index_shapes(int *img_array, int dsize, int *ones,
                             int *ones_count, int *twos, int *twos_count);

__global__ void min_distances_thread_per_one(int *as, int *bs, int num_as,
                                             int num_bs, int img_width,
                                             MinResult *block_results);

__host__ void make_points(int *as, int *bs, int num_as, int num_bs,
                          int img_width, int2 *points_a, int2 *points_b);

__global__ void
min_distances_thread_per_pair(int2 *points_a, int2 *points_b, int num_as,
                              int num_bs, int img_width,
                              MinResultSingleIndex *block_results);

__global__ void final_reduction(MinResult *input, int num_elements,
                                MinResult *output);

__global__ void final_reduction(MinResultSingleIndex *input, int num_elements,
                                MinResultSingleIndex *output);

#endif