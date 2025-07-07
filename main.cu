#include <climits>
#include <cooperative_groups.h>
#include <cstring>
#include <random>

#define THREADS_PER_BLOCK 512
#define WARP_SIZE 32
#define MAX_VALUES_PER_THREAD 64

// error checking macro
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

__global__ void index_shapes(int *img_array, int dsize, int *ones,
                             int *ones_idx, int *twos, int *twos_idx) {
  int local_ones[MAX_VALUES_PER_THREAD];
  int num_ones = 0;
  int local_twos[MAX_VALUES_PER_THREAD];
  int num_twos = 0;

  // grid stride
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < dsize;
       idx += gridDim.x * blockDim.x) {
    int value = img_array[idx];
    if (value == 1) {
      local_ones[num_ones] = idx;
      num_ones += 1;
    } else if (value == 2) {
      local_twos[num_twos] = idx;
      num_twos += 1;
    }
  }

  // write indices to global memory
  int buffer_idx;
  if (num_ones > 0) {
    buffer_idx = atomicAdd(ones_idx, num_ones);
    memcpy(ones + buffer_idx, local_ones, num_ones * sizeof(int));
  }
  if (num_twos > 0) {
    buffer_idx = atomicAdd(twos_idx, num_twos);
    memcpy(twos + buffer_idx, local_twos, num_twos * sizeof(int));
  }
}

__global__ void min_distances(int *ones, int *twos, int num_ones, int num_twos,
                              int size_x, int size_y, int *accum_1_idxs,
                              int *accum_2_idxs, int *accum_distances) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  __shared__ int idx_2s[THREADS_PER_BLOCK];
  __shared__ int x2s[THREADS_PER_BLOCK];
  __shared__ int y2s[THREADS_PER_BLOCK];
  int min_1_idx, x1, y1;
  int min_2_idx = -1;
  int min_distance = INT_MAX;

  if (idx < num_ones) {
    min_1_idx = ones[idx];
    x1 = min_1_idx % size_y;
    y1 = min_1_idx / size_x;
  }
  // loop over all twos in blocks, loading into shared memory
  for (int block = 0; block < (num_twos + blockDim.x - 1) / blockDim.x;
       ++block) {
    int twos_index = block * blockDim.x + threadIdx.x;
    if (twos_index < num_twos) {
      int two = twos[twos_index];
      idx_2s[threadIdx.x] = two;
      x2s[threadIdx.x] = two % size_y;
      y2s[threadIdx.x] = two / size_x;
    } else
      idx_2s[threadIdx.x] = -1;
    __syncthreads();

    if (idx < num_ones) {
      for (int j = 0; j < THREADS_PER_BLOCK; ++j) {
        if (block * blockDim.x + j == num_twos)
          break;
        int idx_2 = idx_2s[j];
        int x2 = x2s[j];
        int y2 = y2s[j];
        int d1 = x2 - x1;
        int d2 = y2 - y1;
        int distance = d1 * d1 + d2 * d2;
        if (distance < min_distance) {
          min_distance = distance;
          min_2_idx = idx_2;
        }
      }
    }
    __syncthreads();
  }

  // warp shuffle reduction
  unsigned mask = 0xFFFFFFFFU;
  __shared__ int min_distances[WARP_SIZE];
  __shared__ int min_1_idxs[WARP_SIZE];
  __shared__ int min_2_idxs[WARP_SIZE];
  int lane = threadIdx.x % warpSize;
  int warpID = threadIdx.x / warpSize;

  // first warp reduction
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    int new_distance = __shfl_down_sync(mask, min_distance, offset);
    int new_1_idx = __shfl_down_sync(mask, min_1_idx, offset);
    int new_2_idx = __shfl_down_sync(mask, min_2_idx, offset);
    if (new_distance < min_distance) {
      min_distance = new_distance;
      min_1_idx = new_1_idx;
      min_2_idx = new_2_idx;
    }
  }
  if (lane == 0) {
    min_distances[warpID] = min_distance;
    min_1_idxs[warpID] = min_1_idx;
    min_2_idxs[warpID] = min_2_idx;
  }
  __syncthreads();

  // second warp reduction
  if (warpID == 0) {
    if (threadIdx.x < blockDim.x / warpSize) {
      min_distance = min_distances[lane];
      min_1_idx = min_1_idxs[lane];
      min_2_idx = min_2_idxs[lane];
    } else {
      min_distance = INT_MAX;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      int new_distance = __shfl_down_sync(mask, min_distance, offset);
      int new_1_idx = __shfl_down_sync(mask, min_1_idx, offset);
      int new_2_idx = __shfl_down_sync(mask, min_2_idx, offset);
      if (new_distance < min_distance) {
        min_distance = new_distance;
        min_1_idx = new_1_idx;
        min_2_idx = new_2_idx;
      }
    }

    // write results to gmem
    if (threadIdx.x == 0) {
      accum_1_idxs[blockIdx.x] = min_1_idx;
      accum_2_idxs[blockIdx.x] = min_2_idx;
      accum_distances[blockIdx.x] = min_distance;
    }
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

  int num_blocks_indexing =
      (total_pixels + MAX_VALUES_PER_THREAD - 1) / MAX_VALUES_PER_THREAD;

  int *d_image, *d_ones, *d_twos, *d_num_ones, *d_num_twos, *d_accum_ones,
      *d_accum_twos;
  int h_num_ones[1];
  int h_num_twos[1];

  int *d_accum_dist;
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
  cudaCheckErrors("cudaMemset d_ones_idx error");
  cudaMemset(d_num_twos, 0, sizeof(int));
  cudaCheckErrors("cudaMemset d_twos_idx error");

  index_shapes<<<num_blocks_indexing, THREADS_PER_BLOCK>>>(
      d_image, total_pixels, d_ones, d_num_ones, d_twos, d_num_twos);
  cudaCheckErrors("index_shapes kernel launch error");
  cudaMemcpy(h_num_ones, d_num_ones, sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors("cudaMemcpy h_ones_idx error");
  cudaMemcpy(h_num_twos, d_num_twos, sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors("cudaMemcpy h_twos_idx error");

  int num_blocks_reduction =
      (*h_num_ones + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  cudaMalloc(&d_accum_ones, num_blocks_reduction * sizeof(int));
  cudaCheckErrors("cudaMalloc d_accum_ones error");
  cudaMalloc(&d_accum_twos, num_blocks_reduction * sizeof(int));
  cudaCheckErrors("cudaMalloc d_accum_twos error");
  cudaMalloc(&d_accum_dist, num_blocks_reduction * sizeof(int));
  cudaCheckErrors("cudaMalloc d_accum_dist error");
  min_distances<<<num_blocks_reduction, THREADS_PER_BLOCK>>>(
      d_ones, d_twos, *h_num_ones, *h_num_twos, img_size, img_size,
      d_accum_ones, d_accum_twos, d_accum_dist);
  cudaCheckErrors("calculate_distances kernel launch error");

  // DO THIS FINAL REDUCTION ON GPU
  int h_accum_ones[num_blocks_reduction];
  int h_accum_twos[num_blocks_reduction];
  int distance[num_blocks_reduction];
  cudaMemcpy(h_accum_ones, d_accum_ones, num_blocks_reduction * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaCheckErrors("cudaMemcpy accum_ones error");
  cudaMemcpy(h_accum_twos, d_accum_twos, num_blocks_reduction * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaCheckErrors("cudaMemcpy accum_twos error");
  cudaMemcpy(distance, d_accum_dist, num_blocks_reduction * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaCheckErrors("cudaMemcpy distance error");

  int min_one_idx = -1;
  int min_two_idx = -1;
  int min_distance = INT_MAX;
  for (int i = 0; i < num_blocks_reduction; ++i) {
    if (distance[i] < min_distance) {
      min_distance = distance[i];
      min_one_idx = h_accum_ones[i];
      min_two_idx = h_accum_twos[i];
    }
  }

  // REMEMBER TO CONVERT FROM 1D INDEX TO 2D
  printf("Distance: %d\nIndex 1: %d\nIndex 2: %d", min_distance, min_one_idx,
         min_two_idx);
}

// future idea: change the distance reduction kernel to use one thread per
// combination of 1-2 indices, and use cub block reduction with operator that
// both calculates distance and selects the minimum index combination. cannot
// use cub device level reduction because that requires all values to be in
// memory, which is problematic because we could have 500000 * 500000 values
// in worst case scenario for a 1000x1000 image. though maybe some loading
// primitives from cub combined with stream partitioning could help?