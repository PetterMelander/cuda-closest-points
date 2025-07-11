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

__global__ void index_shapes(int *img_array, int dsize, int *as,
                             int *num_as, int *bs, int *num_bs) {
  __shared__ int s_as[TILE_SIZE];
  __shared__ int s_bs[TILE_SIZE];
  __shared__ int s_num_as;
  __shared__ int s_num_bs;

  if (threadIdx.x == 0) {
    s_num_as = 0;
    s_num_bs = 0;
  }
  __syncthreads();

  int block_base_idx = blockIdx.x * blockDim.x;

  while (block_base_idx < dsize) {
    int my_idx = block_base_idx + threadIdx.x;
    int my_value = -1;
    int write_idx_as = -1;
    int write_idx_bs = -1;

    if (my_idx < dsize) {
      my_value = img_array[my_idx];
    }

    if (my_value == 1) {
      write_idx_as = atomicAdd(&s_num_as, 1);
      if (write_idx_as < TILE_SIZE) {
        s_as[write_idx_as] = my_idx;
      }
    } else if (my_value == 2) {
      write_idx_bs = atomicAdd(&s_num_bs, 1);
      if (write_idx_bs < TILE_SIZE) {
        s_bs[write_idx_bs] = my_idx;
      }
    }

    __syncthreads();

    bool needs_flush_as = s_num_as >= TILE_SIZE;
    bool needs_flush_bs = s_num_bs >= TILE_SIZE;

    if (needs_flush_as) {
      flush_buffer(s_as, s_num_as, as, num_as);
    }
    if (needs_flush_bs) {
      flush_buffer(s_bs, s_num_bs, bs, num_bs);
    }

    if (needs_flush_as && write_idx_as >= TILE_SIZE) {
      s_as[write_idx_as - TILE_SIZE] = my_idx;
    }
    if (needs_flush_bs && write_idx_bs >= TILE_SIZE) {
      s_bs[write_idx_bs - TILE_SIZE] = my_idx;
    }

    __syncthreads();

    block_base_idx += gridDim.x * blockDim.x;
  }

  flush_buffer_final(s_as, s_num_as, as, num_as);
  flush_buffer_final(s_bs, s_num_bs, bs, num_bs);
}
