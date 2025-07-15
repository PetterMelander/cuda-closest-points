#include "../include/kernels.cuh"
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

__device__ __forceinline__ MinResult
single_warp_shuffle_reduction(MinResult min_result) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    int new_distance = __shfl_down_sync(FULL_MASK, min_result.distance, offset);
    int new_a_idx = __shfl_down_sync(FULL_MASK, min_result.a_idx, offset);
    int new_b_idx = __shfl_down_sync(FULL_MASK, min_result.b_idx, offset);
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
  int lane = threadIdx.x % WARP_SIZE;
  int warpID = threadIdx.x / WARP_SIZE;

  // First warp reduction. All warps.
  min_result = single_warp_shuffle_reduction(min_result);
  if (lane == 0)
    smem[warpID] = min_result;
  __syncthreads();

  // Second warp reduction. First warp only.
  if (warpID == 0) {
    if (lane < THREADS_PER_BLOCK / WARP_SIZE)
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
  int lane =
      (threadIdx.x + BLOCK_SIZE_2D_DISTANCE * threadIdx.y) % THREADS_PER_BLOCK;
  int warpID =
      (threadIdx.x + BLOCK_SIZE_2D_DISTANCE * threadIdx.y) / THREADS_PER_BLOCK;

  // First warp reduction. All warps.
  min_result = single_warp_shuffle_reduction(min_result);
  if (lane == 0)
    smem[warpID] = min_result;
  __syncthreads();

  // Second warp reduction. First warp only.
  if (warpID == 0) {
    if (lane <
        BLOCK_SIZE_2D_DISTANCE * BLOCK_SIZE_2D_DISTANCE / THREADS_PER_BLOCK)
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

  for (int i = threadIdx.x; i < num_elements; i += THREADS_PER_BLOCK) {
    MinResult new_result = input[i];
    if (new_result.distance < min_result.distance)
      min_result = new_result;
  }

  warp_shuffle_reduction(min_result, smem, output);
}

__global__ void min_distances_thread_per_a_int2(
    const int2 *__restrict__ as, const int2 *__restrict__ bs, int num_as,
    int num_bs, int img_width, MinResult *block_results, bool order_swapped) {
  int tid = threadIdx.x + THREADS_PER_BLOCK * blockIdx.x;
  __shared__ union {
    struct {
      int x[THREADS_PER_BLOCK];
      int y[THREADS_PER_BLOCK];
    } bs_coords;
    MinResult min_results[WARP_SIZE];
  } smem;
  int2 my_a, min_b = {-1, -1};
  int min_distance = INT_MAX;

  if (tid < num_as) {
    my_a = as[tid];
  }
  // Loop over all bs in blocks, loading into shared memory. Each thread finds
  // its closest b.
  for (int b_block = 0;
       b_block < (num_bs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
       ++b_block) {
    int b_idx = b_block * THREADS_PER_BLOCK + threadIdx.x;
    if (b_idx < num_bs) {
      smem.bs_coords.x[threadIdx.x] = bs[b_idx].x;
      smem.bs_coords.y[threadIdx.x] = bs[b_idx].y;
    }
    __syncthreads();

    if (tid < num_as) {
      for (int j = 0; j < THREADS_PER_BLOCK; ++j) {
        if (b_block * THREADS_PER_BLOCK + j == num_bs)
          break;
        int b_x = smem.bs_coords.x[j];
        int b_y = smem.bs_coords.y[j];
        int dx = b_x - my_a.x;
        int dy = b_y - my_a.y;
        int distance = dx * dx + dy * dy;
        if (distance < min_distance) {
          min_distance = distance;
          min_b.x = b_x;
          min_b.y = b_y;
        }
      }
    }
    __syncthreads();
  }

  MinResult min_result;
  min_result.distance = min_distance;
  if (order_swapped) {
    min_result.a_idx = min_b.x + min_b.y * img_width;
    min_result.b_idx = my_a.x + my_a.y * img_width;
  } else {
    min_result.a_idx = my_a.x + my_a.y * img_width;
    min_result.b_idx = min_b.x + min_b.y * img_width;
  }
  warp_shuffle_reduction(min_result, smem.min_results, block_results);
}

__global__ void min_distances_thread_per_a(
    const int *__restrict__ as, const int *__restrict__ bs, int num_as,
    int num_bs, int img_width, MinResult *block_results, bool order_swapped) {
  int tid = threadIdx.x + THREADS_PER_BLOCK * blockIdx.x;
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
  for (int b_block = 0;
       b_block < (num_bs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
       ++b_block) {
    int b_idx = b_block * THREADS_PER_BLOCK + threadIdx.x;
    if (b_idx < num_bs) {
      int b_linear_coord = bs[b_idx];
      smem.bs[threadIdx.x] = b_linear_coord;
      s_b_xs[threadIdx.x] = b_linear_coord % img_width;
      s_b_ys[threadIdx.x] = b_linear_coord / img_width;
    }
    __syncthreads();

    if (tid < num_as) {
      for (int j = 0; j < THREADS_PER_BLOCK; ++j) {
        if (b_block * THREADS_PER_BLOCK + j == num_bs)
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

__global__ void min_distances_thread_per_pair(const int2 *__restrict__ points_a,
                                              const int2 *__restrict__ points_b,
                                              int num_as, int num_bs,
                                              int img_width,
                                              MinResult *block_results) {

  // Grid stride 2d
  int min_distance = INT_MAX;
  int min_a_idx = -1;
  int min_b_idx = -1;
  for (int x_idx = threadIdx.x + blockIdx.x * BLOCK_SIZE_2D_DISTANCE;
       x_idx < num_as; x_idx += gridDim.x * BLOCK_SIZE_2D_DISTANCE) {

    int2 my_a;
    if (x_idx < num_as)
      my_a = points_a[x_idx];

    for (int y_idx = threadIdx.y + blockIdx.y * BLOCK_SIZE_2D_DISTANCE;
         y_idx < num_bs; y_idx += gridDim.y * BLOCK_SIZE_2D_DISTANCE) {

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