#include "../include/kernels.cuh"
#include <cooperative_groups.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace cg = cooperative_groups;

__device__ __forceinline__ MinResult
single_warp_shuffle_reduction(MinResult min_result) {
  unsigned mask = 0xFFFFFFFFU;
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    int new_distance = __shfl_down_sync(mask, min_result.distance, offset);
    int new_a_idx = __shfl_down_sync(mask, min_result.a_idx, offset);
    int new_b_idx = __shfl_down_sync(mask, min_result.b_idx, offset);
    if (new_distance < min_result.distance) {
      min_result.distance = new_distance;
      min_result.a_idx = new_a_idx;
      min_result.b_idx = new_b_idx;
    }
  }
  return min_result;
}

__device__ __forceinline__ void warp_shuffle_reduction(MinResult min_result,
                                                       MinResult *smem,
                                                       MinResult *output) {
  int lane = threadIdx.x % warpSize;
  int warpID = threadIdx.x / warpSize;

  // First warp reduction. All warps.
  min_result = single_warp_shuffle_reduction(min_result);
  if (lane == 0)
    smem[warpID] = min_result;
  __syncthreads();

  // Second warp reduction. First warp only.
  if (warpID == 0) {
    if (lane < blockDim.x / warpSize)
      min_result = smem[lane];
    else
      min_result.distance = INT_MAX;
    min_result = single_warp_shuffle_reduction(min_result);

    // Write results to global memory. First thread of first warp only.
    if (lane == 0)
      output[blockIdx.x] = min_result;
  }

  cg::grid_group grid = cg::this_grid();
  grid.sync();

  if (blockIdx.x == 0) {
    MinResult final_min = {INT_MAX, -1, -1};
    int num_blocks = gridDim.x;

    // Use a grid-stride loop for this block's threads to reduce further
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
      MinResult val = output[i];
      if (val.distance < final_min.distance) {
        final_min = val;
      }
    }

    // Reduce the results within the first block to get the final single value.
    final_min = single_warp_shuffle_reduction(final_min);
    if (lane == 0)
      smem[warpID] = final_min;
    __syncthreads();

    if (warpID == 0) {
      if (lane < blockDim.x / warpSize)
        final_min = smem[lane];
      else
        final_min.distance = INT_MAX;

      final_min = single_warp_shuffle_reduction(final_min);

      // The very first thread of the grid writes the final result.
      if (threadIdx.x == 0) {
        output[0] = final_min;
      }
    }
  }
}

__device__ __forceinline__ void warp_shuffle_reduction_2d(MinResult min_result,
                                                          MinResult *smem,
                                                          MinResult *output) {
  int lane = (threadIdx.x + blockDim.x * threadIdx.y) % warpSize;
  int warpID = (threadIdx.x + blockDim.x * threadIdx.y) / warpSize;
  int block_size = blockDim.x * blockDim.y;

  // First warp reduction. All warps.
  min_result = single_warp_shuffle_reduction(min_result);
  if (lane == 0)
    smem[warpID] = min_result;
  __syncthreads();

  // Second warp reduction. First warp only.
  if (warpID == 0) {
    if (lane < block_size / warpSize)
      min_result = smem[lane];
    else
      min_result.distance = INT_MAX;
    min_result = single_warp_shuffle_reduction(min_result);

    // Write results to global memory. First thread of first warp only.
    if (lane == 0)
      output[blockIdx.x + blockIdx.y * gridDim.x] = min_result;
  }

  cg::grid_group grid = cg::this_grid();
  grid.sync();

  if (blockIdx.x == 0 && blockIdx.y == 0) {
    MinResult final_min = {INT_MAX, -1, -1};
    int num_blocks = gridDim.x * gridDim.y;

    // Use a grid-stride loop for this block's threads to reduce further
    for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < num_blocks;
         i += block_size) {
      MinResult val = output[i];
      if (val.distance < final_min.distance) {
        final_min = val;
      }
    }

    // Reduce the results within the first block to get the final single value.
    final_min = single_warp_shuffle_reduction(final_min);
    if (lane == 0)
      smem[warpID] = final_min;
    __syncthreads();

    if (warpID == 0) {
      if (lane < block_size / warpSize)
        final_min = smem[lane];
      else
        final_min.distance = INT_MAX;
      final_min = single_warp_shuffle_reduction(final_min);

      // The very first thread of the grid writes the final result.
      if (lane == 0) {
        output[0] = final_min;
      }
    }
  }
}

__global__ void final_reduction(MinResult *input, int num_elements,
                                MinResult *output) {
  MinResult min_result;
  min_result.a_idx = -1;
  min_result.b_idx = -1;
  min_result.distance = INT_MAX;
  __shared__ MinResult smem[WARP_SIZE];

  for (int i = threadIdx.x; i < num_elements; i += blockDim.x) {
    MinResult new_result = input[i];
    if (new_result.distance < min_result.distance)
      min_result = new_result;
  }

  warp_shuffle_reduction(min_result, smem, output);
}

// __global__ void final_reduction_2d(MinResult *input, int num_elements,
//                                 MinResult *output) {
//   MinResult min_result;
//   min_result.a_idx = -1;
//   min_result.b_idx = -1;
//   min_result.distance = INT_MAX;
//   __shared__ MinResult smem[WARP_SIZE];

//   int tid = threadIdx.x + threadIdx.y * blockDim.x;
//   for (int i = tid; i < num_elements; i += blockDim.x * blockDim.y) {
//     MinResult new_result = input[i];
//     if (new_result.distance < min_result.distance)
//       min_result = new_result;
//   }

//   warp_shuffle_reduction_2d(min_result, smem, output);
// }

__global__ void min_distances_thread_per_a(int *as, int *bs, int num_as,
                                           int num_bs, int img_width,
                                           MinResult *block_results,
                                           bool order_swapped) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  __shared__ union {
    int bs[THREADS_PER_BLOCK];
    MinResult min_results[WARP_SIZE];
  } smem;
  __shared__ int s_b_xs[THREADS_PER_BLOCK];
  __shared__ int s_b_ys[THREADS_PER_BLOCK];
  int my_a_linear_coord, min_b_linear_coord = -1, a_x, a_y,
                         min_distance = INT_MAX;

  if (tid < num_as) {
    my_a_linear_coord = as[tid];
    a_x = my_a_linear_coord % img_width;
    a_y = my_a_linear_coord / img_width;
  }
  // Loop over all bs in blocks, loading into shared memory. Each thread finds
  // its closest b.
  for (int b_block = 0; b_block < (num_bs + blockDim.x - 1) / blockDim.x;
       ++b_block) {
    int b_idx = b_block * blockDim.x + threadIdx.x;
    if (b_idx < num_bs) {
      int b_linear_coord = bs[b_idx];
      smem.bs[threadIdx.x] = b_linear_coord;
      s_b_xs[threadIdx.x] = b_linear_coord % img_width;
      s_b_ys[threadIdx.x] = b_linear_coord / img_width;
    }
    __syncthreads();

    if (tid < num_as) {
      for (int j = 0; j < blockDim.x; ++j) {
        if (b_block * blockDim.x + j == num_bs)
          break;
        int b_linear_coord = smem.bs[j];
        int b_x = s_b_xs[j];
        int b_y = s_b_ys[j];
        int dx = b_x - a_x;
        int dy = b_y - a_y;
        int distance = dx * dx + dy * dy;
        if (distance < min_distance) {
          min_distance = distance;
          min_b_linear_coord = b_linear_coord;
        }
      }
    }
    __syncthreads();
  }

  MinResult min_result;
  min_result.distance = min_distance;
  if (order_swapped) {
    min_result.a_idx = min_b_linear_coord;
    min_result.b_idx = my_a_linear_coord;
  } else {
    min_result.a_idx = my_a_linear_coord;
    min_result.b_idx = min_b_linear_coord;
  }
  warp_shuffle_reduction(min_result, smem.min_results, block_results);
}

__host__ __device__ int2 idx_to_point(int idx, int img_width) {
  return make_int2(idx % img_width, idx / img_width);
}

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

__global__ void min_distances_thread_per_pair(int2 *points_a, int2 *points_b,
                                              int num_as, int num_bs,
                                              int img_width,
                                              MinResult *block_results) {

  // Grid stride 2d
  int min_distance = INT_MAX;
  int min_a_idx = -1;
  int min_b_idx = -1;
  for (int x_idx = threadIdx.x + blockIdx.x * blockDim.x; x_idx < num_as;
       x_idx += gridDim.x * blockDim.x) {

    int2 my_a;
    if (x_idx < num_as)
      my_a = points_a[x_idx];

    for (int y_idx = threadIdx.y + blockIdx.y * blockDim.y; y_idx < num_bs;
         y_idx += gridDim.y * blockDim.y) {

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

  __shared__ MinResult results[WARP_SIZE];
  MinResult min_result{min_distance, min_a_idx, min_b_idx};
  warp_shuffle_reduction_2d(min_result, results, block_results);
}