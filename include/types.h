#ifndef TYPES_H
#define TYPES_H

#include <cuda_runtime.h>
#include <sstream>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/**
 * @brief A struct representing two pixels, with 1d indices, and their distance.
 * Used for internal functions.
 *
 */
struct MinResult {
  int distance;
  int a_idx;
  int b_idx;
};

/**
 * @brief A struct representing a pair of pixels in two masks, with mask values,
 * indices, and distance. Used as final output of the algorithm.
 *
 */
struct Pair {
  int a;
  int ax;
  int ay;
  int b;
  int bx;
  int by;
  float distance;

  Pair transpose() { return Pair{b, bx, by, a, ax, ay, distance}; }

  std::string to_string() const {
    std::ostringstream oss;
    oss << a << ": (" << ax << ", " << ay << ")\n"
        << b << ": (" << bx << ", " << by << ")\n"
        << "distance: " << distance << "\n";
    return oss.str();
  }
};

/**
 * @brief An allocator for a std::vector that allocates elements in pinned
 * memory for use in async memory transfers.
 *
 * @tparam T Type of elemnt to put in the vector.
 */
template <typename T> struct PinnedAllocator {
  using value_type = T;

  T *allocate(std::size_t n) {
    T *ptr;
    // Use cudaMallocHost to allocate page-locked host memory
    CUDA_CHECK(cudaMallocHost(&ptr, n * sizeof(T)));
    return ptr;
  }

  void deallocate(T *p, std::size_t n) {
    // Use cudaFreeHost to free it
    CUDA_CHECK(cudaFreeHost(p));
  }
};

template <typename T> using PinnedVector = std::vector<T, PinnedAllocator<T>>;

#endif