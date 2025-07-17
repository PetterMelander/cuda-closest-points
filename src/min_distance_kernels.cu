#include "../include/kernels.cuh"
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

constexpr int block_size = 512;
constexpr int warp_size = 32;
constexpr int block_size_2d = 16;
constexpr unsigned int full_mask = 0xffffffff;

__host__ int get_block_size_distance() { return block_size; }

__host__ int get_grid_size_distance(int num_as) {
  int num_blocks = num_blocks_max_occupancy(
      min_distances_thread_per_a, block_size, sizeof(int2) * block_size, 1.f);
  return std::min(num_blocks, (num_as + block_size - 1) / block_size);
}

__host__ dim3 get_block_dims_distance_2d() {
  uint block_dim = block_size_2d;
  return dim3{block_dim, block_dim};
}

__host__ dim3 get_grid_dims_2d(int num_as, int num_bs) {
  int num_blocks = (num_as * num_bs + block_size_2d * block_size_2d - 1) /
                   block_size_2d * block_size_2d;
  uint grid_dim = (uint)sqrt(num_blocks);
  return dim3{grid_dim, grid_dim};
}

/**
 * @brief Perform a warp shuffle reduction for the warp.
 *
 * @param min_result The thread's min result so far.
 * @return The warp's min result.
 */
__device__ __forceinline__ MinResult
single_warp_shuffle_reduction(MinResult min_result) {
  for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
    int new_distance = __shfl_down_sync(full_mask, min_result.distance, offset);
    int new_a_idx = __shfl_down_sync(full_mask, min_result.a_idx, offset);
    int new_b_idx = __shfl_down_sync(full_mask, min_result.b_idx, offset);
    if (new_distance < min_result.distance) {
      min_result.distance = new_distance;
      min_result.a_idx = new_a_idx;
      min_result.b_idx = new_b_idx;
    }
  }
  return min_result;
}

/**
 * @brief Perform a warp shuffle reduction on block level.
 *
 * @param min_result The thread's min result so far.
 * @param smem Shared memory that each warp writes its min value to.
 * @param output MinResult array in gmem that each block writes to.
 */
__device__ __forceinline__ void warp_shuffle_reduction(MinResult min_result,
                                                       MinResult *smem,
                                                       MinResult *output) {
  int lane = threadIdx.x % warp_size;
  int warpID = threadIdx.x / warp_size;

  // First warp reduction. All warps.
  min_result = single_warp_shuffle_reduction(min_result);
  if (lane == 0)
    smem[warpID] = min_result;
  __syncthreads();

  // Second warp reduction. First warp only.
  if (warpID == 0) {
    if (lane < block_size / warp_size)
      min_result = smem[lane];
    else
      min_result.distance = INT_MAX;
    min_result = single_warp_shuffle_reduction(min_result);

    // Write results to global memory. First thread of first warp only.
    if (lane == 0)
      output[blockIdx.x] = min_result;
  }
}

/**
 * @brief Perform a warp shuffle reduction on block level. Specialized for a
 * kernel using 2d blocks.
 *
 * @param min_result The thread's min result so far.
 * @param smem Shared memory that each warp writes its min value to.
 * @param output MinResult array in gmem that each block writes to.
 */
__device__ __forceinline__ void warp_shuffle_reduction_2d(MinResult min_result,
                                                          MinResult *smem,
                                                          MinResult *output) {
  int lane = (threadIdx.x + block_size_2d * threadIdx.y) % warp_size;
  int warpID = (threadIdx.x + block_size_2d * threadIdx.y) / warp_size;

  // First warp reduction. All warps.
  min_result = single_warp_shuffle_reduction(min_result);
  if (lane == 0)
    smem[warpID] = min_result;
  __syncthreads();

  // Second warp reduction. First warp only.
  if (warpID == 0) {
    if (lane < block_size_2d * block_size_2d / warp_size)
      min_result = smem[lane];
    else
      min_result.distance = INT_MAX;
    min_result = single_warp_shuffle_reduction(min_result);

    // Write results to global memory. First thread of first warp only.
    if (lane == 0)
      output[blockIdx.x + blockIdx.y * gridDim.x] = min_result;
  }
}

/**
 * @brief A kernel that uses a single thread block to reduce the input array to
 * a single value.
 *
 * @param input Array of MinResults to be reduced.
 * @param num_elements Number of elements in input array.
 * @param output Pointer to write result to.
 */
__global__ void final_reduction(MinResult *input, int num_elements,
                                MinResult *output) {
  MinResult min_result;
  min_result.a_idx = -1;
  min_result.b_idx = -1;
  min_result.distance = INT_MAX;
  __shared__ MinResult smem[warp_size];

  for (int i = threadIdx.x; i < num_elements; i += block_size) {
    MinResult new_result = input[i];
    if (new_result.distance < min_result.distance)
      min_result = new_result;
  }

  warp_shuffle_reduction(min_result, smem, output);
}

/**
 * @brief Find the pixels that are closest to each other in two masks.
 *
 * The kernel has each thread attend to one pixel of mask a. It then loops over
 * all pixels of mask b in tiles, loading them into shared memory.
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
    int num_bs, int img_width, MinResult *block_results, bool order_swapped) {

  __shared__ union {
    int bs[block_size];
    MinResult min_results[warp_size];
  } smem;
  __shared__ int s_b_xs[block_size];
  __shared__ int s_b_ys[block_size];

  int min_a_linear_coord, min_b_linear_coord = -1, min_distance = INT_MAX;

  int tid = threadIdx.x + block_size * blockIdx.x;
  for (int i = 0; i < num_as; i += block_size * gridDim.x) {
    tid += i;
    int my_a_linear_coord, a_x, a_y;

    __syncthreads();
    if (tid < num_as) {
      my_a_linear_coord = as[tid];
      a_x = my_a_linear_coord % img_width;
      a_y = my_a_linear_coord / img_width;
    }
    // Loop over all bs in blocks, loading into shared memory. Each thread finds
    // its closest b.
    for (int b_block = 0; b_block < (num_bs + block_size - 1) / block_size;
         ++b_block) {
      int b_idx = b_block * block_size + threadIdx.x;
      if (b_idx < num_bs) {
        int b_linear_coord = bs[b_idx];
        smem.bs[threadIdx.x] = b_linear_coord;
        s_b_xs[threadIdx.x] = b_linear_coord % img_width;
        s_b_ys[threadIdx.x] = b_linear_coord / img_width;
      }
      __syncthreads();

      if (tid < num_as) {
        for (int j = 0; j < block_size; ++j) {
          if (b_block * block_size + j == num_bs)
            break;

          int b_linear_coord = smem.bs[j];
          int dx = s_b_xs[j] - a_x;
          int dy = s_b_ys[j] - a_y;
          int distance = dx * dx + dy * dy;

          if (distance < min_distance) {
            min_distance = distance;
            min_b_linear_coord = b_linear_coord;
            min_a_linear_coord = my_a_linear_coord;
          }
        }
      }
      __syncthreads();
    }
  }

  MinResult min_result;
  min_result.distance = min_distance;
  if (order_swapped) {
    min_result.a_idx = min_b_linear_coord;
    min_result.b_idx = min_a_linear_coord;
  } else {
    min_result.a_idx = min_a_linear_coord;
    min_result.b_idx = min_b_linear_coord;
  }
  warp_shuffle_reduction(min_result, smem.min_results, block_results);
}

/**
 * @brief Helper function for converting a 1d index into 2d.
 *
 * @param idx 1d image index.
 * @param img_width Image width in pixels.
 * @return __host__ int2 representing 2d pixel index.
 */
__host__ __device__ int2 idx_to_point(int idx, int img_width) {
  return make_int2(idx % img_width, idx / img_width);
}

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
                          cudaStream_t stream) {
  auto op = [img_width] __device__(int idx) {
    return idx_to_point(idx, img_width);
  };
  auto policy = thrust::cuda::par_nosync.on(stream);

  thrust::transform(policy, as, as + num_as, points_a, op);
  thrust::transform(policy, bs, bs + num_bs, points_b, op);
}

/**
 * @brief Find the pixels that are closest to each other in two masks.
 *
 * The kernel has each thread attend to one pair of pixels. Because there is no
 * data reuse, it uses no shared memory. Also, since there is no opportunity to
 * reuse expensive conversions from 1d index to 2d for distance calculation,
 * this kernel expects 2d pixel indices as inputs.
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
                                              MinResult *block_results) {

  // Grid stride 2d
  int min_distance = INT_MAX;
  int min_a_idx = -1;
  int min_b_idx = -1;
  for (int x_idx = threadIdx.x + blockIdx.x * block_size_2d; x_idx < num_as;
       x_idx += gridDim.x * block_size_2d) {

    int2 my_a;
    if (x_idx < num_as)
      my_a = points_a[x_idx];

    for (int y_idx = threadIdx.y + blockIdx.y * block_size_2d; y_idx < num_bs;
         y_idx += gridDim.y * block_size_2d) {

      if (x_idx < num_as && y_idx < num_bs) {
        int2 my_b = points_b[y_idx];
        int dx = my_b.x - my_a.x;
        int dy = my_b.y - my_a.y;
        int distance = dx * dx + dy * dy;
        if (distance < min_distance) {
          min_distance = distance;
          min_a_idx = my_a.y * img_width + my_a.x;
          min_b_idx = my_b.y * img_width + my_b.x;
        }
      }
    }
  }

  __shared__ MinResult results[warp_size];
  MinResult min_result{min_distance, min_a_idx, min_b_idx};
  warp_shuffle_reduction_2d(min_result, results, block_results);
}