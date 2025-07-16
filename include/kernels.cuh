#ifndef KERNELS_H
#define KERNELS_H

#include "types.h"
#include <cooperative_groups.h>

/**
 * @brief Get the number of thread blocks needed for max occupancy, given a
 * kernel, its parameters, and an oversubscription factor.
 *
 * @tparam KernelFunc The kernel in question.
 * @param kernel The kernel in question.
 * @param blockSize Number of threads per block.
 * @param smem_per_block Bytes of shared memory.
 * @param oversubscription_factor The oversubscription factor.
 * @return int Number of blocks.
 */
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

/**
 * @brief Find the pixels that are closest to each other in two masks.
 *
 * The kernel has each thread attend to one pixel of mask a.
 *
 * This kernel does not do a complete reduction. It only reduces the input array
 * to one element per block.
 *
 * @param as Array of 1d pixel indices of mask a.
 * @param bs Array of 1d pixel indices of mask b.
 * @param num_as Number of pixels in mask a.
 * @param num_bs Number of pixels in mask b.
 * @param img_width Image width in pixels.
 * @param block_results Array of MinResults to write results to.
 * @param order_swapped Whether a and b have been swapped before kernel launch.
 */
__global__ void min_distances_thread_per_a(
    const int *__restrict__ as, const int *__restrict__ bs, int num_as,
    int num_bs, int img_width, MinResult *block_results, bool swapped);

/**
 * @brief Use thrust to transform arrays of 1d pixel indices to 2d.
 *
 * @param as 1d indices of pixels in mask a.
 * @param bs 1d indices of pixals in mask b.
 * @param num_as Number of pixels in mask a.
 * @param num_bs Number of pixels in mask b.
 * @param img_width Image width in pixels.
 * @param points_a Output array for mask a.
 * @param points_b Output array for mask b.
 * @param stream Cuda stream to do work in.
 */
__host__ void make_points(int *as, int *bs, int num_as, int num_bs,
                          int img_width, int2 *points_a, int2 *points_b,
                          cudaStream_t stream);

/**
 * @brief Find the pixels that are closest to each other in two masks.
 *
 * The kernel has each thread attend to one pair of pixels.
 *
 * This kernel does not do a complete reduction. It only reduces the input array
 * to one element per block.
 *
 * @param points_a 2d indices of pixels in mask a.
 * @param points_b 2d indices of pixels in mask b.
 * @param num_as Number of pixels in mask a.
 * @param num_bs Number of pixels in mask b.
 * @param img_width Image width in pixels.
 * @param block_results Array of MinResults to write results to.
 */
__global__ void min_distances_thread_per_pair(const int2 *__restrict__ points_a,
                                              const int2 *__restrict__ points_b,
                                              int num_as, int num_bs,
                                              int img_width,
                                              MinResult *block_results);

/**
 * @brief A kernel that uses a single thread block to reduce the input array to
 * a single value.
 *
 * @param input Array of MinResults to be reduced.
 * @param num_elements Number of elements in input array.
 * @param output Pointer to write result to.
 */
__global__ void final_reduction(MinResult *input, int num_elements,
                                MinResult *output);

/**
 * @brief Finds all mask edge pixels and write indices and values to gmem.
 *
 * @param image Image array.
 * @param img_height Image height in pixels.
 * @param img_width Image width in pixels.
 * @param g_mask_idxs Array to write mask indices to.
 * @param g_mask_values Array to write mask values to.
 * @param g_num_mask_pixels Variable to write number of mask pixels to.
 */
__global__ void index_edges(const int *__restrict__ image, int img_height,
                            int img_width, int *g_nonzero_idxs,
                            int *g_nonzero_values, int *g_num_nonzeros);

/**
 * @brief Get block dims for mask indexing kernel.
 *
 * @return Block dims.
 */
__host__ dim3 get_block_dims_indexing();

/**
 * @brief Get grid dims for mask indexing kernel.
 *
 * @param img_height Image height in pixels.
 * @param img_width Image width in pixels.
 * @return Grid dims.
 */
__host__ dim3 get_grid_dims_indexing(int img_height, int img_width);

/**
 * @brief Get block size for distance reduction kernel with one thread per a.
 *
 * @return Block size.
 */
__host__ int get_block_size_distance();

/**
 * @brief Get grid size for distance reduction kernel with one thread per a.
 *
 * @return Grid size.
 */
__host__ int get_grid_size_distance(int num_as);

/**
 * @brief Get block dims for distance reduction kernel with one thread per pixel
 * pair.
 *
 * @return Block dims.
 */
__host__ dim3 get_block_dims_distance_2d();

/**
 * @brief Get grid dims for distance reduction kernel with one thread per pixel
 * pair.
 *
 * @return Grid dims.
 */
__host__ dim3 get_grid_dims_2d(int num_as, int num_bs);

#endif