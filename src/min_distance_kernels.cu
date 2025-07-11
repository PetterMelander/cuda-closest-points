#include "../include/kernels.cuh"
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

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
}

__device__ __forceinline__ void warp_shuffle_reduction_2d(MinResult min_result,
                                                          MinResult *smem,
                                                          MinResult *output) {
  int lane = (threadIdx.x + blockDim.x * threadIdx.y) % warpSize;
  int warpID = (threadIdx.x + blockDim.x * threadIdx.y) / warpSize;

  // First warp reduction. All warps.
  min_result = single_warp_shuffle_reduction(min_result);
  if (lane == 0)
    smem[warpID] = min_result;
  __syncthreads();

  // Second warp reduction. First warp only.
  if (warpID == 0) {
    if (lane < blockDim.x * blockDim.y / warpSize)
      min_result = smem[lane];
    else
      min_result.distance = INT_MAX;
    min_result = single_warp_shuffle_reduction(min_result);

    // Write results to global memory. First thread of first warp only.
    if (lane == 0)
      output[blockIdx.x + blockIdx.y * gridDim.x] = min_result;
  }
}

__global__ void final_reduction(MinResult *input, int num_elements,
                                MinResult *output) {
  MinResult min_result;
  min_result.a_idx = -1;
  min_result.b_idx = -1;
  min_result.distance = INT_MAX;
  __shared__ MinResult smem[WARP_SIZE];

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < num_elements; i += blockDim.x) {
    MinResult new_result = input[i];
    if (new_result.distance < min_result.distance)
      min_result = new_result;
  }

  warp_shuffle_reduction(min_result, smem, output);
}

__global__ void min_distances_thread_per_a(int *as, int *bs, int num_as,
                                           int num_bs, int img_width,
                                           MinResult *block_results) {
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
  min_result.a_idx = my_a_linear_coord;
  min_result.b_idx = min_b_linear_coord;
  warp_shuffle_reduction(min_result, smem.min_results, block_results);
}

__host__ __device__ int2 idx_to_point(int idx, int img_width) {
  return make_int2(idx % img_width, idx / img_width);
}

__host__ void make_points(int *as, int *bs, int num_as, int num_bs,
                          int img_width, int2 *points_a, int2 *points_b) {
  auto op = [img_width] __device__(int idx) {
    return idx_to_point(idx, img_width);
  };
  // TODO: launch transforms in streams to parallelize
  thrust::transform(thrust::device, as, as + num_as, points_a, op);
  thrust::transform(thrust::device, bs, bs + num_bs, points_b, op);
}

__global__ void min_distances_thread_per_pair(int2 *points_a, int2 *points_b,
                                              int num_as, int num_bs,
                                              int img_width,
                                              MinResult *block_results) {

  // Grid stride 2d
  int min_distance = INT_MAX;
  int min_x_idx = -1;
  int min_y_idx = -1;
  for (int i = blockIdx.x; i < (num_as + blockDim.x - 1) / blockDim.x;
       i += gridDim.x) {
    int x_idx = i * blockDim.x + threadIdx.x;

    int2 my_a = points_a[x_idx];

    for (int j = blockIdx.y; j < (num_bs + blockDim.y - 1) / blockDim.y;
         j += gridDim.y) {
      int y_idx = j * blockDim.y + threadIdx.y;

      if (x_idx < num_as && y_idx < num_bs) {
        int2 my_b = points_b[y_idx];
        int dx = my_b.x - my_a.x;
        int dy = my_b.y - my_a.y;
        int distance = dx * dx + dy * dy;
        if (distance < min_distance) {
          min_distance = distance;
          min_x_idx = x_idx;
          min_y_idx = y_idx;
        }
      }
    }
  }

  __shared__ MinResult results[WARP_SIZE];
  MinResult min_result{min_distance, min_x_idx, min_y_idx};
  warp_shuffle_reduction_2d(min_result, results, block_results);
}