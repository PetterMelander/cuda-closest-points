#ifndef KERNELS_H
#define KERNELS_H

#include "../include/types.h"
#include <climits>
#include <cooperative_groups.h>

#define THREADS_PER_BLOCK 512
#define TILE_SIZE_INDEXING 2048
#define WARP_SIZE 32

__global__ void find_nonzeros(int *img_array, int dsize, int *g_nonzero_idxs,
                                int *g_nonzero_values, int *g_num_nonzeros);

__global__ void min_distances_thread_per_a(int *as, int *bs, int num_as,
                                           int num_bs, int img_width,
                                           MinResult *block_results);

__host__ void make_points(int *as, int *bs, int num_as, int num_bs,
                          int img_width, int2 *points_a, int2 *points_b);

__global__ void min_distances_thread_per_pair(int2 *points_a, int2 *points_b,
                                              int num_as, int num_bs,
                                              int img_width,
                                              MinResult *block_results);

__global__ void final_reduction(MinResult *input, int num_elements,
                                MinResult *output);

#endif