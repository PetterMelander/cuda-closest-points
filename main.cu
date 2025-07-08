#include <climits>
#include <cooperative_groups.h>
#include <cstring>
#include <random>

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

__device__ void flush_buffer(int *s_buffer, int &s_count, int *g_buffer,
                             int *g_count) {
  int items_to_flush = (s_count < TILE_SIZE) ? s_count : TILE_SIZE;

  __shared__ int global_base_idx;
  if (threadIdx.x == 0) {
    global_base_idx = atomicAdd(g_count, items_to_flush);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < items_to_flush; i += blockDim.x) {
    g_buffer[global_base_idx + i] = s_buffer[i];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    if (s_count > TILE_SIZE) {
      s_count = s_count - TILE_SIZE;
    } else {
      s_count = 0;
    }
  }
  __syncthreads();
}

__device__ void flush_buffer_final(int *s_buffer, int s_count, int *g_buffer,
                                   int *g_count) {
  if (s_count <= 0) {
    return;
  }

  __shared__ int global_base_idx;
  if (threadIdx.x == 0) {
    global_base_idx = atomicAdd(g_count, s_count);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < s_count; i += blockDim.x) {
    g_buffer[global_base_idx + i] = s_buffer[i];
  }
}

__global__ void index_shapes(int *img_array, int dsize, int *ones,
                             int *ones_count, int *twos, int *twos_count) {
  __shared__ int s_ones[TILE_SIZE];
  __shared__ int s_twos[TILE_SIZE];
  __shared__ int s_num_ones;
  __shared__ int s_num_twos;

  if (threadIdx.x == 0) {
    s_num_ones = 0;
    s_num_twos = 0;
  }
  __syncthreads();

  int block_base_idx = blockIdx.x * blockDim.x;

  while (block_base_idx < dsize) {
    int my_idx = block_base_idx + threadIdx.x;
    int my_value = -1;
    int write_idx_ones = -1;
    int write_idx_twos = -1;

    if (my_idx < dsize) {
      my_value = img_array[my_idx];
    }

    if (my_value == 1) {
      write_idx_ones = atomicAdd(&s_num_ones, 1);
      if (write_idx_ones < TILE_SIZE) {
        s_ones[write_idx_ones] = my_idx;
      }
    } else if (my_value == 2) {
      write_idx_twos = atomicAdd(&s_num_twos, 1);
      if (write_idx_twos < TILE_SIZE) {
        s_twos[write_idx_twos] = my_idx;
      }
    }

    __syncthreads();

    bool needs_flush_ones = s_num_ones >= TILE_SIZE;
    bool needs_flush_twos = s_num_twos >= TILE_SIZE;

    if (needs_flush_ones) {
      flush_buffer(s_ones, s_num_ones, ones, ones_count);
    }
    if (needs_flush_twos) {
      flush_buffer(s_twos, s_num_twos, twos, twos_count);
    }

    if (needs_flush_ones && write_idx_ones >= TILE_SIZE) {
      s_ones[write_idx_ones - TILE_SIZE] = my_idx;
    }
    if (needs_flush_twos && write_idx_twos >= TILE_SIZE) {
      s_twos[write_idx_twos - TILE_SIZE] = my_idx;
    }

    __syncthreads();

    block_base_idx += gridDim.x * blockDim.x;
  }

  flush_buffer_final(s_ones, s_num_ones, ones, ones_count);
  flush_buffer_final(s_twos, s_num_twos, twos, twos_count);
}

struct MinResult {
  int distance;
  int one_idx;
  int two_idx;
};

__device__ __forceinline__ MinResult
warp_shuffle_reduction(MinResult min_result) {
  unsigned mask = 0xFFFFFFFFU;
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    int new_distance = __shfl_down_sync(mask, min_result.distance, offset);
    int new_one_idx = __shfl_down_sync(mask, min_result.one_idx, offset);
    int new_two_idx = __shfl_down_sync(mask, min_result.two_idx, offset);
    if (new_distance < min_result.distance) {
      min_result.distance = new_distance;
      min_result.one_idx = new_one_idx;
      min_result.two_idx = new_two_idx;
    }
  }
  return min_result;
}

__global__ void min_distances(int *ones, int *twos, int num_ones, int num_twos,
                              int img_width, MinResult *block_results) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  __shared__ int s_two_idxs[THREADS_PER_BLOCK];
  __shared__ int s_x2s[THREADS_PER_BLOCK];
  __shared__ int s_y2s[THREADS_PER_BLOCK];
  int min_one_idx, x1, y1;
  int min_two_idx = -1;
  int min_distance = INT_MAX;

  if (idx < num_ones) {
    min_one_idx = ones[idx];
    x1 = min_one_idx % img_width;
    y1 = min_one_idx / img_width;
  }
  // Loop over all twos in blocks, loading into shared memory. Each thread finds its closest two.
  for (int block = 0; block < (num_twos + blockDim.x - 1) / blockDim.x;
       ++block) {
    int twos_index = block * blockDim.x + threadIdx.x;
    if (twos_index < num_twos) {
      int two = twos[twos_index];
      s_two_idxs[threadIdx.x] = two;
      s_x2s[threadIdx.x] = two % img_width;
      s_y2s[threadIdx.x] = two / img_width;
    } else
      s_two_idxs[threadIdx.x] = -1;
    __syncthreads();

    if (idx < num_ones) {
      for (int j = 0; j < THREADS_PER_BLOCK; ++j) {
        if (block * blockDim.x + j == num_twos)
          break;
        int two_idx = s_two_idxs[j];
        int x2 = s_x2s[j];
        int y2 = s_y2s[j];
        int d1 = x2 - x1;
        int d2 = y2 - y1;
        int distance = d1 * d1 + d2 * d2;
        if (distance < min_distance) {
          min_distance = distance;
          min_two_idx = two_idx;
        }
      }
    }
    __syncthreads();
  }

  // Warp shuffle reduction
  __shared__ MinResult s_min_results[WARP_SIZE];
  int lane = threadIdx.x % warpSize;
  int warpID = threadIdx.x / warpSize;
  MinResult min_result;
  min_result.distance = min_distance;
  min_result.one_idx = min_one_idx;
  min_result.two_idx = min_two_idx;

  // First warp reduction
  min_result = warp_shuffle_reduction(min_result);
  if (lane == 0)
    s_min_results[warpID] = min_result;
  __syncthreads();

  // Second warp reduction
  if (warpID == 0) {
    if (threadIdx.x < blockDim.x / warpSize)
      min_result = s_min_results[lane];
    else
      min_result.distance = INT_MAX;
    min_result = warp_shuffle_reduction(min_result);

    // Write results to global memory
    if (threadIdx.x == 0)
      block_results[blockIdx.x] = min_result;
  }
}

int main() {
  int img_size = 1024;
  int total_pixels = img_size * img_size;
  int *image = new int[total_pixels];

  // Use random device and mt19937 for reproducibility
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  // Assign 10% to 1, 10% to 2, rest to 0
  for (int i = 0; i < total_pixels; ++i) {
    double r = dis(gen);
    if (r < 0.1)
      image[i] = 1;
    else if (r < 0.2)
      image[i] = 2;
    else
      image[i] = 0;
  }

  int device_id;
  cudaGetDevice(&device_id);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device_id);

  const size_t shared_mem_per_block = sizeof(int) * 2048 * 2;

  int max_active_blocks_per_sm;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_sm,
                                                index_shapes, THREADS_PER_BLOCK,
                                                shared_mem_per_block);

  const int oversubscription_factor = 4;
  int num_blocks_indexing = props.multiProcessorCount *
                            max_active_blocks_per_sm * oversubscription_factor;

  int *d_image, *d_ones, *d_twos, *d_num_ones, *d_num_twos;
  int h_num_ones[1];
  int h_num_twos[1];

  cudaMalloc(&d_image, total_pixels * sizeof(int));
  cudaCheckErrors("cudaMalloc d_image error");
  cudaMalloc(&d_ones, total_pixels * sizeof(int));
  cudaCheckErrors("cudaMalloc d_ones error");
  cudaMalloc(&d_twos, total_pixels * sizeof(int));
  cudaCheckErrors("cudaMalloc d_twos error");
  cudaMalloc(&d_num_ones, sizeof(int));
  cudaCheckErrors("cudaMalloc d_num_ones error");
  cudaMalloc(&d_num_twos, sizeof(int));
  cudaCheckErrors("cudaMalloc d_num_twos error");

  cudaMemcpy(d_image, image, total_pixels * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy d_image error");
  cudaMemset(d_num_ones, 0, sizeof(int));
  cudaCheckErrors("cudaMemset d_num_ones error");
  cudaMemset(d_num_twos, 0, sizeof(int));
  cudaCheckErrors("cudaMemset d_num_twos error");

  index_shapes<<<num_blocks_indexing, THREADS_PER_BLOCK>>>(
      d_image, total_pixels, d_ones, d_num_ones, d_twos, d_num_twos);
  cudaCheckErrors("index_shapes kernel launch error");
  cudaMemcpy(h_num_ones, d_num_ones, sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors("cudaMemcpy h_num_ones error");
  cudaMemcpy(h_num_twos, d_num_twos, sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors("cudaMemcpy h_num_twos error");

  int num_blocks_reduction =
      (*h_num_ones + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  MinResult *block_results;
  cudaMalloc(&block_results, sizeof(MinResult) * num_blocks_reduction);
  cudaCheckErrors("cudaMalloc block_results error");
  min_distances<<<num_blocks_reduction, THREADS_PER_BLOCK>>>(
      d_ones, d_twos, *h_num_ones, *h_num_twos, img_size, block_results);
  cudaCheckErrors("min_distances kernel launch error");

  // DO THIS FINAL REDUCTION ON GPU
  int h_accum_ones[num_blocks_reduction];
  int h_accum_twos[num_blocks_reduction];
  int h_distances[num_blocks_reduction];
  // cudaMemcpy(h_accum_ones, d_accum_ones, num_blocks_reduction * sizeof(int),
  //  cudaMemcpyDeviceToHost);
  // cudaCheckErrors("cudaMemcpy h_accum_ones error");
  // cudaMemcpy(h_accum_twos, d_accum_twos, num_blocks_reduction * sizeof(int),
  //  cudaMemcpyDeviceToHost);
  // cudaCheckErrors("cudaMemcpy h_accum_twos error");
  // cudaMemcpy(h_distances, d_accum_dist, num_blocks_reduction * sizeof(int),
  //  cudaMemcpyDeviceToHost);
  // cudaCheckErrors("cudaMemcpy h_distances error");

  int min_one_idx = -1;
  int min_two_idx = -1;
  int min_distance = INT_MAX;
  for (int i = 0; i < num_blocks_reduction; ++i) {
    if (h_distances[i] < min_distance) {
      min_distance = h_distances[i];
      min_one_idx = h_accum_ones[i];
      min_two_idx = h_accum_twos[i];
    }
  }

  // REMEMBER TO CONVERT FROM 1D INDEX TO 2D
  printf("Distance: %d\nIndex 1: %d\nIndex 2: %d\n", min_distance, min_one_idx,
         min_two_idx);

  delete[] image;
}

// future idea: change the distance reduction kernel to use one thread per
// combination of 1-2 indices, and use cub block reduction with operator that
// both calculates distance and selects the minimum index combination