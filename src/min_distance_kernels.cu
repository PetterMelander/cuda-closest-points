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

__device__ __forceinline__ MinResultSingleIndex
single_warp_shuffle_reduction(MinResultSingleIndex min_result) {
  unsigned mask = 0xFFFFFFFFU;
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    int new_distance = __shfl_down_sync(mask, min_result.distance, offset);
    int new_idx = __shfl_down_sync(mask, min_result.idx, offset);
    if (new_distance < min_result.distance) {
      min_result.distance = new_distance;
      min_result.idx = new_idx;
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
    if (threadIdx.x < blockDim.x / warpSize)
      min_result = smem[lane];
    else
      min_result.distance = INT_MAX;
    min_result = single_warp_shuffle_reduction(min_result);

    // Write results to global memory. First thread of first warp only.
    if (threadIdx.x == 0)
      output[blockIdx.x] = min_result;
  }
}

__device__ __forceinline__ void
warp_shuffle_reduction(MinResultSingleIndex min_result,
                       MinResultSingleIndex *smem,
                       MinResultSingleIndex *output) {
  int lane = threadIdx.x % warpSize;
  int warpID = threadIdx.x / warpSize;

  // First warp reduction. All warps.
  min_result = single_warp_shuffle_reduction(min_result);
  if (lane == 0)
    smem[warpID] = min_result;
  __syncthreads();

  // Second warp reduction. First warp only.
  if (warpID == 0) {
    if (threadIdx.x < blockDim.x / warpSize)
      min_result = smem[lane];
    else
      min_result.distance = INT_MAX;
    min_result = single_warp_shuffle_reduction(min_result);

    // Write results to global memory. First thread of first warp only.
    if (threadIdx.x == 0)
      output[blockIdx.x] = min_result;
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

__global__ void final_reduction(MinResultSingleIndex *input, int num_elements,
                                MinResultSingleIndex *output) {
  MinResultSingleIndex min_result;
  min_result.idx = -1;
  min_result.distance = INT_MAX;
  __shared__ MinResultSingleIndex smem[WARP_SIZE];

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < num_elements; i += blockDim.x) {
    MinResultSingleIndex new_result = input[i];
    if (new_result.distance < min_result.distance)
      min_result = new_result;
  }

  warp_shuffle_reduction(min_result, smem, output);
}

__global__ void min_distances_thread_per_one(int *as, int *bs, int num_as,
                                             int num_bs, int img_width,
                                             MinResult *block_results) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  __shared__ union {
    int bs[THREADS_PER_BLOCK];
    MinResult min_results[WARP_SIZE];
  } smem;
  __shared__ int s_b_xs[THREADS_PER_BLOCK];
  __shared__ int s_b_ys[THREADS_PER_BLOCK];
  int min_a_idx, min_b_idx = -1, a_x, a_y, min_distance = INT_MAX;

  if (tid < num_as) {
    min_a_idx = as[tid];
    a_x = min_a_idx % img_width;
    a_y = min_a_idx / img_width;
  }
  // Loop over all twos in blocks, loading into shared memory. Each thread finds
  // its closest two.
  for (int block = 0; block < (num_bs + blockDim.x - 1) / blockDim.x; ++block) {
    int b_idx = block * blockDim.x + threadIdx.x;
    if (b_idx < num_bs) {
      int b = bs[b_idx];
      smem.bs[threadIdx.x] = b;
      s_b_xs[threadIdx.x] = b % img_width;
      s_b_ys[threadIdx.x] = b / img_width;
    }
    __syncthreads();

    if (tid < num_as) {
      for (int j = 0; j < THREADS_PER_BLOCK; ++j) {
        if (block * blockDim.x + j == num_bs)
          break;
        int two_idx = smem.bs[j];
        int b_x = s_b_xs[j];
        int b_y = s_b_ys[j];
        int dx = b_x - a_x;
        int dy = b_y - a_y;
        int distance = dx * dx + dy * dy;
        if (distance < min_distance) {
          min_distance = distance;
          min_b_idx = two_idx;
        }
      }
    }
    __syncthreads();
  }

  MinResult min_result;
  min_result.distance = min_distance;
  min_result.a_idx = min_a_idx;
  min_result.b_idx = min_b_idx;
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

__global__ void
min_distances_thread_per_pair(int2 *points_a, int2 *points_b, int num_as,
                              int num_bs, int img_width,
                              MinResultSingleIndex *block_results) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int grid_size = blockDim.x * gridDim.x;
  int num_pairs = num_as * num_bs;
  int min_distance = INT_MAX;
  int min_global_idx = -1;

  for (int i = tid; i < num_pairs; i += grid_size) {
    int2 point_a = points_a[i % num_as];
    int2 point_b = points_b[i / num_as];
    int dx = point_b.x - point_a.x;
    int dy = point_b.y - point_a.y;
    int distance = dx * dx + dy * dy;

    if (distance < min_distance) {
      min_distance = distance;
      min_global_idx = i;
    }
  }

  MinResultSingleIndex min_result;
  min_result.distance = min_distance;
  min_result.idx = min_global_idx;
  __shared__ MinResultSingleIndex smem[WARP_SIZE];
  warp_shuffle_reduction(min_result, smem, block_results);
}