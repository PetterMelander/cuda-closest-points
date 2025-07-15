#ifndef KERNELS_H
#define KERNELS_H

#include "types.h"
#include <climits>
#include <cooperative_groups.h>

constexpr int THREADS_PER_BLOCK = 256;
constexpr int TILE_SIZE_INDEXING = 2048;
constexpr int WARP_SIZE = 32;
constexpr unsigned int FULL_MASK = 0xffffffff;
constexpr int BLOCK_SIZE_EDGE_FIND_X = 32;
constexpr int BLOCK_SIZE_EDGE_FIND_Y = 16;
constexpr int BLOCK_SIZE_2D_DISTANCE = 16;

__global__ void find_nonzeros(const int *__restrict__ img_array, int dsize,
                              int *g_nonzero_idxs, int *g_nonzero_values,
                              int *g_num_nonzeros);

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

__global__ void find_edges(const int *__restrict__ image, int img_height,
                           int img_width, int *output);

#endif