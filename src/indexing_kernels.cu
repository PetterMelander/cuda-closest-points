#include "../include/kernels.cuh"

__global__ void find_edges(const int *__restrict__ image, int img_height,
                           int img_width, int *output) {
  // TODO: load multiple elements at once right at the beginning, to hide
  // latency.
  int gidx = threadIdx.x + blockIdx.x * (BLOCK_SIZE_EDGE_FIND_X - 2) - 1;
  int gidy = threadIdx.y + blockIdx.y * (BLOCK_SIZE_EDGE_FIND_Y - 2) - 1;
  int g_linear_idx = gidx + img_width * gidy;
  bool g_valid =
      gidx >= 0 && gidx < img_width && gidy >= 0 && gidy < img_height;
  bool l_valid = threadIdx.x > 0 && threadIdx.x < BLOCK_SIZE_EDGE_FIND_X - 1 &&
                 threadIdx.y > 0 && threadIdx.y < BLOCK_SIZE_EDGE_FIND_Y - 1;
  __shared__ int s_tile[BLOCK_SIZE_EDGE_FIND_Y][BLOCK_SIZE_EDGE_FIND_X];

  int center, top, bottom, right, left;
  if (g_valid) {
    center = image[g_linear_idx];
    s_tile[threadIdx.y][threadIdx.x] = center;
  } else
    center = -1;
  __syncthreads();

  left = __shfl_up_sync(FULL_MASK, center, 1);
  right = __shfl_down_sync(FULL_MASK, center, 1);
  if (g_valid && l_valid) {

    if (gidx <= 0)
      left = center;
    if (gidx >= img_width)
      right = center;
    if (gidy > 0)
      top = s_tile[threadIdx.y - 1][threadIdx.x];
    else
      top = center;
    if (gidy < img_height - 1)
      bottom = s_tile[threadIdx.y + 1][threadIdx.x];
    else
      bottom = center;

    if (center != top || center != bottom || center != left || center != right)
      output[g_linear_idx] = center;
  }
}

__device__ void flush_buffer(int *s_buffer_idxs, int *s_buffer_values,
                             int &s_count, int *g_buffer_idxs,
                             int *g_buffer_values, int *g_count) {
  int items_to_flush =
      (s_count < TILE_SIZE_INDEXING) ? s_count : TILE_SIZE_INDEXING;

  __shared__ int global_base_idx;
  if (threadIdx.x == 0) {
    global_base_idx = atomicAdd(g_count, items_to_flush);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < items_to_flush; i += THREADS_PER_BLOCK) {
    g_buffer_idxs[global_base_idx + i] = s_buffer_idxs[i];
    g_buffer_values[global_base_idx + i] = s_buffer_values[i];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    if (s_count > TILE_SIZE_INDEXING) {
      s_count = s_count - TILE_SIZE_INDEXING;
    } else {
      s_count = 0;
    }
  }
  __syncthreads();
}

__device__ void flush_buffer_final(int *s_buffer_idxs, int *s_buffer_values,
                                   int s_count, int *g_buffer_idxs,
                                   int *g_buffer_values, int *g_count) {
  if (s_count <= 0) {
    return;
  }

  __shared__ int global_base_idx;
  if (threadIdx.x == 0) {
    global_base_idx = atomicAdd(g_count, s_count);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < s_count; i += THREADS_PER_BLOCK) {
    g_buffer_idxs[global_base_idx + i] = s_buffer_idxs[i];
    g_buffer_values[global_base_idx + i] = s_buffer_values[i];
  }
}

__global__ void find_nonzeros(const int *__restrict__ img_array, int dsize,
                              int *g_nonzero_idxs, int *g_nonzero_values,
                              int *g_num_nonzeros) {
  __shared__ int s_nonzero_idxs[TILE_SIZE_INDEXING];
  __shared__ int s_nonzero_values[TILE_SIZE_INDEXING];
  __shared__ int s_num_nonzeros;

  if (threadIdx.x == 0) {
    s_num_nonzeros = 0;
  }
  __syncthreads();

  int block_base_idx = blockIdx.x * THREADS_PER_BLOCK;
  while (block_base_idx < dsize) {
    int my_idx = block_base_idx + threadIdx.x;
    int write_idx_as = -1;

    int my_value;
    if (my_idx < dsize) {
      my_value = img_array[my_idx];
    }

    if (my_value != 0) {
      write_idx_as = atomicAdd(&s_num_nonzeros, 1);
      if (write_idx_as < TILE_SIZE_INDEXING) {
        s_nonzero_idxs[write_idx_as] = my_idx;
        s_nonzero_values[write_idx_as] = my_value;
      }
    }

    __syncthreads();

    bool needs_flush = s_num_nonzeros >= TILE_SIZE_INDEXING;

    if (needs_flush) {
      flush_buffer(s_nonzero_idxs, s_nonzero_values, s_num_nonzeros,
                   g_nonzero_idxs, g_nonzero_values, g_num_nonzeros);
    }

    if (needs_flush && write_idx_as >= TILE_SIZE_INDEXING) {
      s_nonzero_idxs[write_idx_as - TILE_SIZE_INDEXING] = my_idx;
      s_nonzero_values[write_idx_as - TILE_SIZE_INDEXING] = my_value;
    }

    __syncthreads();

    block_base_idx += gridDim.x * THREADS_PER_BLOCK;
  }

  flush_buffer_final(s_nonzero_idxs, s_nonzero_values, s_num_nonzeros,
                     g_nonzero_idxs, g_nonzero_values, g_num_nonzeros);
}
