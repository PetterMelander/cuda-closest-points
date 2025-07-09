#include "../include/kernels.cuh"

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
