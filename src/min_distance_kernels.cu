#include "../include/kernels.cuh"
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

constexpr int block_size = 768;
constexpr int warp_size = 32;
constexpr int block_size_2d = 16;
constexpr unsigned int full_mask = 0xffffffff;

__host__ int get_block_size_distance() { return block_size; }

__host__ int get_grid_size_distance(int num_as) {
  int num_blocks = num_blocks_max_occupancy(
      min_distances_thread_per_a, block_size, sizeof(int2) * block_size, 0.5f);
  return std::min(num_blocks, (num_as + block_size - 1) / block_size);
}

__host__ dim3 get_block_dims_distance_2d() {
  uint block_dim = block_size_2d;
  return dim3{block_dim, block_dim};
}

__host__ dim3 get_grid_dims_2d(int num_as, int num_bs) {
  int num_blocks = num_blocks_max_occupancy(min_distances_thread_per_pair,
                                            block_size_2d * block_size_2d,
                                            sizeof(MinResult) * warp_size, 1.f);
  num_blocks = std::min(num_blocks,
                        (num_as * num_bs + block_size_2d * block_size_2d - 1) /
                            block_size_2d * block_size_2d);
  uint grid_dim = (uint)sqrt(num_blocks);
  return dim3{grid_dim, grid_dim};
}

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

__global__ void min_distances_thread_per_a(
    const int *__restrict__ as, const int *__restrict__ bs, int num_as,
    int num_bs, int img_width, MinResult *block_results, bool order_swapped) {
  int tid = threadIdx.x + block_size * blockIdx.x;
  __shared__ union {
    int bs[block_size];
    MinResult min_results[warp_size];
  } smem;
  __shared__ int s_b_xs[block_size];
  __shared__ int s_b_ys[block_size];
  int my_a_linear_coord, min_b_linear_coord = -1, a_x, a_y,
                         min_distance = INT_MAX;

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