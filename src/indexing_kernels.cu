#include "../include/kernels.cuh"

constexpr unsigned int full_mask = 0xffffffff;
constexpr int block_size_x = 32;
constexpr int block_size_y = 16;
constexpr int items_per_thread = 4;
constexpr int buffer_size = 2048;

__host__ dim3 get_block_dims_indexing() {
  return dim3(block_size_x, block_size_y);
}

__host__ dim3 get_grid_dims_indexing(int img_height, int img_width) {
  uint num_blocks_x = (img_width + (block_size_x - 2) * items_per_thread - 1) /
                      ((block_size_x - 2) * items_per_thread);
  uint num_blocks_y = (img_height + block_size_y - 1) / block_size_y;
  dim3 grid_size(num_blocks_x, num_blocks_y);
  return grid_size;
}

/**
 * @brief Writes the shared buffer to global memory.
 *
 * This function is called when the shared buffer is full.
 *
 * @param s_buffer_idxs Shared buffer of indices.
 * @param s_buffer_values Shared buffer of values.
 * @param s_count Shared element counter.
 * @param g_buffer_idxs Global buffer of indices.
 * @param g_buffer_values Global buffer of values.
 * @param g_count Global element counter.
 */
__device__ void flush_buffer(int *s_buffer_idxs, int *s_buffer_values,
                             int &s_count, int *g_buffer_idxs,
                             int *g_buffer_values, int *g_count) {
  int items_to_flush = (s_count < buffer_size) ? s_count : buffer_size;

  __shared__ int global_base_idx;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    global_base_idx = atomicAdd(g_count, items_to_flush);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < items_to_flush;
       i += block_size_x * block_size_y) {
    g_buffer_idxs[global_base_idx + i] = s_buffer_idxs[i];
    g_buffer_values[global_base_idx + i] = s_buffer_values[i];
  }
  __syncthreads();

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    if (s_count > buffer_size) {
      s_count = s_count - buffer_size;
    } else {
      s_count = 0;
    }
  }
  __syncthreads();
}

/**
 * @brief Writes the shared buffer to global memory.
 *
 * This function is called when the kernel ends, even if the buffer is not full.
 *
 * @param s_buffer_idxs Shared buffer of indices.
 * @param s_buffer_values Shared buffer of values.
 * @param s_count Shared element counter.
 * @param g_buffer_idxs Global buffer of indices.
 * @param g_buffer_values Global buffer of values.
 * @param g_count Global element counter.
 */
__device__ void flush_buffer_final(int *s_buffer_idxs, int *s_buffer_values,
                                   int s_count, int *g_buffer_idxs,
                                   int *g_buffer_values, int *g_count) {
  if (s_count <= 0) {
    return;
  }

  __shared__ int global_base_idx;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    global_base_idx = atomicAdd(g_count, s_count);
  }
  __syncthreads();

  for (int i = threadIdx.x + threadIdx.y * block_size_x; i < s_count;
       i += block_size_x * block_size_y) {
    g_buffer_idxs[global_base_idx + i] = s_buffer_idxs[i];
    g_buffer_values[global_base_idx + i] = s_buffer_values[i];
  }
}

/**
 * @brief Determines if a pixel is a mask edge.
 *
 * @param center Mask value of the thread's pixel.
 * @param g_valid If the pixel is part of the image.
 * @param l_valid If the pixel is part of the block's tile.
 * @param gidx x index of pixel.
 * @param gidy y index of pixel.
 * @param img_height Image height in pixels.
 * @param img_width Image width in pixels.
 * @param s_tile Tile of shared memory, used for sharing top and bottom values.
 * @return Whether the thread's pixel is a mask edge.
 */
__device__ __forceinline__ bool
is_edge(int center, bool g_valid, bool l_valid, int gidx, int gidy,
        int img_height, int img_width, int s_tile[block_size_y][block_size_x]) {
  int left = __shfl_up_sync(full_mask, center, 1);
  int right = __shfl_down_sync(full_mask, center, 1);
  if ((center != 0 && g_valid && l_valid)) {
    int top, bottom;
    if (gidx == 0)
      left = center;
    if (gidx == img_width - 1)
      right = center;
    if (gidy == 0)
      top = center;
    else
      top = s_tile[threadIdx.y - 1][threadIdx.x];
    if (gidy == img_height - 1)
      bottom = center;
    else
      bottom = s_tile[threadIdx.y + 1][threadIdx.x];

    return center != top || center != bottom || center != left ||
           center != right;
  }
  return false;
}

/**
 * @brief Finds all mask edge pixels and write indices and values to gmem.
 *
 * In this kernel, each thread block has a halo of one. The threads responsible
 * for the halo only participate in data loading. The value of each thread is
 * loaded into shared memory so it can be accessed by neighbouring threads.
 *
 * To write indices of edge pixels to gmem, in a 1d array of unknown length,
 * atomics are used. To reduce the atomics pressure, each thread block first
 * accumulates its values in a shared buffer before writing to gmem.
 *
 * The kernel has each thread read several pixels from gmem up front to hide
 * latency.
 *
 * @param image Image array.
 * @param img_height Image height in pixels.
 * @param img_width Image width in pixels.
 * @param g_mask_idxs Array to write mask indices to.
 * @param g_mask_values Array to write mask values to.
 * @param g_num_mask_pixels Variable to write number of mask pixels to.
 */
__global__ void index_edges(const int *__restrict__ image, int img_height,
                            int img_width, int *g_mask_idxs, int *g_mask_values,
                            int *g_num_mask_pixels) {

  bool l_valid = threadIdx.x > 0 && threadIdx.x < block_size_x - 1 &&
                 threadIdx.y > 0 && threadIdx.y < block_size_y - 1;
  int base_gidx =
      threadIdx.x + blockIdx.x * (block_size_x - 2) * items_per_thread - 1;
  int gidy = threadIdx.y + blockIdx.y * (block_size_y - 2) - 1;
  __shared__ int s_tile[block_size_y][block_size_x];
  __shared__ int s_nonzero_idxs[buffer_size];
  __shared__ int s_nonzero_values[buffer_size];
  __shared__ int s_num_nonzeros;
  int centers[items_per_thread];

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    s_num_nonzeros = 0;
  }
  __syncthreads();

  if (gidy >= 0 && gidy < img_height) {
    for (int i = 0; i < items_per_thread; ++i) {
      int gidx = base_gidx + i * (block_size_x - 2);
      bool g_valid = gidx >= 0 && gidx < img_width;
      if (g_valid) {
        int g_linear_idx = gidx + img_width * gidy;
        centers[i] = image[g_linear_idx];
      }
    }
  }

  for (int i = 0; i < items_per_thread; ++i) {
    int write_idx_as = -1;
    int gidx = base_gidx + i * (block_size_x - 2);
    bool g_valid =
        gidx >= 0 && gidx < img_width && gidy >= 0 && gidy < img_height;

    int center;
    if (g_valid) {
      center = centers[i];
      s_tile[threadIdx.y][threadIdx.x] = center;
    } else
      center = 0;
    __syncthreads();

    int g_linear_idx;
    if (is_edge(center, g_valid, l_valid, base_gidx, gidx, img_height,
                img_width, s_tile)) {
      write_idx_as = atomicAdd(&s_num_nonzeros, 1);
      if (write_idx_as < buffer_size) {
        g_linear_idx = gidx + img_width * gidy;
        s_nonzero_idxs[write_idx_as] = g_linear_idx;
        s_nonzero_values[write_idx_as] = center;
      }
    }

    __syncthreads();
    bool needs_flush = s_num_nonzeros >= buffer_size;
    if (needs_flush) {
      flush_buffer(s_nonzero_idxs, s_nonzero_values, s_num_nonzeros,
                   g_mask_idxs, g_mask_values, g_num_mask_pixels);
    }

    if (needs_flush && write_idx_as >= buffer_size) {
      s_nonzero_idxs[write_idx_as - buffer_size] = g_linear_idx;
      s_nonzero_values[write_idx_as - buffer_size] = center;
    }
    __syncthreads();
  }
  flush_buffer_final(s_nonzero_idxs, s_nonzero_values, s_num_nonzeros,
                     g_mask_idxs, g_mask_values, g_num_mask_pixels);
}