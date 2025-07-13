cmake_minimum_required(VERSION 3.18)
# Add this policy to use the modern FindPython required by recent pybind11 versions
# cmake_policy(SET CMP0148 NEW)

project(closest_points_cuda LANGUAGES CXX CUDA)

# Find necessary packages
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Set the C++ and CUDA standard to 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")

# Specify CUDA architecture (e.g., Hopper)
# This will generate code for sm_89
set(CMAKE_CUDA_ARCHITECTURES 89)

# Automatically find all .cpp and .cu source files in the src/ directory
set(SOURCES
    src/host_code.cu
    src/indexing_kernels.cu
    src/min_distance_kernels.cu
    src/bindings.cpp
)

include(FetchContent)
FetchContent_Declare(
  cpptrace
  GIT_REPOSITORY https://github.com/jeremy-rifkin/cpptrace.git
  GIT_TAG        v1.0.2 # <HASH or TAG>
)
FetchContent_MakeAvailable(cpptrace)

# Create the Python module from the collected sources
pybind11_add_module(closest_points_cuda ${SOURCES})
target_link_libraries(closest_points_cuda PRIVATE cpptrace::cpptrace)

# This makes headers like <cuda_runtime.h> available to both nvcc and clang++
target_include_directories(closest_points_cuda PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Set target-specific compile options
target_compile_options(closest_points_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -lineinfo --compiler-options -Wall>
)

install(TARGETS closest_points_cuda
        LIBRARY DESTINATION .
)


# Test

set(TEST_SOURCES
    src/host_code.cu
    src/indexing_kernels.cu
    src/min_distance_kernels.cu
)

add_executable(run_test src/test.cpp ${TEST_SOURCES})

target_include_directories(run_test PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(run_test PRIVATE
    ${CUDAToolkit_LIBRARIES}
    cpptrace::cpptrace
)

target_compile_options(run_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -lineinfo --compiler-options -Wall>
)
