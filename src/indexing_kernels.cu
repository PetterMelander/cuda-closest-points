#include "../include/kernels.cuh"

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

  for (int i = threadIdx.x; i < items_to_flush; i += blockDim.x) {
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

  for (int i = threadIdx.x; i < s_count; i += blockDim.x) {
    g_buffer_idxs[global_base_idx + i] = s_buffer_idxs[i];
    g_buffer_values[global_base_idx + i] = s_buffer_values[i];
  }
}

__global__ void find_nonzeros(int *img_array, int dsize, int *g_nonzero_idxs,
                                int *g_nonzero_values, int *g_num_nonzeros) {
  __shared__ int s_nonzero_idxs[TILE_SIZE_INDEXING];
  __shared__ int s_nonzero_values[TILE_SIZE_INDEXING];
  __shared__ int s_num_nonzeros;

  if (threadIdx.x == 0) {
    s_num_nonzeros = 0;
  }
  __syncthreads();

  int block_base_idx = blockIdx.x * blockDim.x;
  // int buffer_size = 0;
  // int buffer[8];
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

    block_base_idx += gridDim.x * blockDim.x;
  }

  flush_buffer_final(s_nonzero_idxs, s_nonzero_values, s_num_nonzeros,
                        g_nonzero_idxs, g_nonzero_values, g_num_nonzeros);
}
