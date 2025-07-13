#include "../include/host_code.cuh"
#include <cstddef>
#include <cub/cub.cuh>
#include <random>
#include <vector>

void select_nonzeros(int *d_image, int total_pixels, int *d_out,
                     int *d_num_selected_out) {
  auto nonzero_op = [] __device__(const int &val) { return val != 0; };

  // Determine temporary device storage requirements
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_image,
                                   d_out, d_num_selected_out, total_pixels,
                                   nonzero_op));

  // Allocate temporary storage
  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  // Run selection
  CUDA_CHECK(cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_image,
                                   d_out, d_num_selected_out, total_pixels,
                                   nonzero_op));
  CUDA_CHECK(cudaFree(d_temp_storage));
}

int *sort_nonzeros(int *d_nonzeros, int d_num_nonzeros,
                   int *d_sorted_nonzeros) {
  // Create a DoubleBuffer to wrap the pair of device pointers
  cub::DoubleBuffer<unsigned int> d_keys(
      reinterpret_cast<unsigned int *>(d_nonzeros),
      reinterpret_cast<unsigned int *>(d_sorted_nonzeros));

  // Determine temporary device storage requirements
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                            d_keys, d_num_nonzeros, 0, 8));

  // Allocate temporary storage
  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  // Run sorting operation
  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                            d_keys, d_num_nonzeros, 0, 8));
  CUDA_CHECK(cudaFree(d_temp_storage));
  return reinterpret_cast<int *>(d_keys.Current());
}

void find_unique(int *d_sorted_nonzeros, int num_nonzeros, int *d_unique_values,
                 int *d_num_unique) {
  // Determine temporary device storage requirements
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes,
                                       d_sorted_nonzeros, d_unique_values,
                                       d_num_unique, num_nonzeros));

  // Allocate temporary storage
  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  // Run selection
  CUDA_CHECK(cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes,
                                       d_sorted_nonzeros, d_unique_values,
                                       d_num_unique, num_nonzeros));
  CUDA_CHECK(cudaFree(d_temp_storage));
}

std::vector<int> detect_masks(int *d_image, int total_pixels) {
  int *d_nonzeros;
  int *d_num_nonzeros;
  CUDA_CHECK(cudaMalloc(&d_nonzeros, sizeof(int) * total_pixels));
  CUDA_CHECK(cudaMalloc(&d_num_nonzeros, sizeof(int)));

  select_nonzeros(d_image, total_pixels, d_nonzeros, d_num_nonzeros);
  int h_num_nonzeros;
  cudaMemcpy(&h_num_nonzeros, d_num_nonzeros, sizeof(int),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK(cudaFree(d_num_nonzeros));

  int *d_sort_buffer;
  CUDA_CHECK(cudaMalloc(&d_sort_buffer, sizeof(int) * h_num_nonzeros));
  int *d_sorted_nonzeros =
      sort_nonzeros(d_nonzeros, h_num_nonzeros, d_sort_buffer);

  int *d_unique_values;
  int *d_num_unique;
  CUDA_CHECK(cudaMalloc(&d_unique_values, sizeof(int) * 256));
  CUDA_CHECK(cudaMalloc(&d_num_unique, sizeof(int)));
  find_unique(d_sorted_nonzeros, h_num_nonzeros, d_unique_values, d_num_unique);

  int num_unique;
  cudaMemcpy(&num_unique, d_num_unique, sizeof(int), cudaMemcpyDeviceToHost);
  std::vector<int> unique_values(num_unique);
  cudaMemcpy(unique_values.data(), d_unique_values, sizeof(int) * num_unique,
             cudaMemcpyDeviceToHost);

  CUDA_CHECK(cudaFree(d_nonzeros));
  CUDA_CHECK(cudaFree(d_sort_buffer));
  CUDA_CHECK(cudaFree(d_unique_values));
  CUDA_CHECK(cudaFree(d_num_unique));

  return unique_values;
}

void temp(int *d_image, std::vector<int> unique_values) {}

int main() {
  int img_size = 1024;
  int total_pixels = img_size * img_size;
  int *image = new int[total_pixels];

  // Use random device and mt19937 for reproducibility
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  std::uniform_int_distribution<> dis2(1, 25);

  // Assign 10% to 1, 10% to 2, rest to 0
  for (int i = 0; i < total_pixels; ++i) {
    double r = dis(gen);
    if (r < 0.2)
      image[i] = dis2(gen);
    else
      image[i] = 0;
  }
  int *d_image;
  cudaMalloc(&d_image, sizeof(int) * total_pixels);
  cudaMemcpy(d_image, image, sizeof(int) * total_pixels,
             cudaMemcpyHostToDevice);

  auto unique_values = detect_masks(d_image, total_pixels);

  for (int i = 0; i < (int)unique_values.size(); ++i) {
    std::cout << unique_values[i];
  }

  cudaFree(d_image);
  delete[] image;
  return 0;
}