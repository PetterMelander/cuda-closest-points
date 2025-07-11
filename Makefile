# --- Compiler ---
NVCC = nvcc

# --- Executable/Library Name ---
TARGET = closest_points

# --- Get Python's dynamic library suffix (e.g., .cpython-312-x86_64-linux-gnu.so) ---
# This is the corrected, robust way to do this.
PY_EXT_SUFFIX = $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PYTHON_MODULE = $(TARGET)$(PY_EXT_SUFFIX)

# --- Directories ---
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
INCLUDE_DIR = include

# --- Source Files ---
# Find all .cu and .cpp files in the source directory
CU_SRCS = $(wildcard $(SRC_DIR)/*.cu)
CPP_SRCS = $(wildcard $(SRC_DIR)/*.cpp)

# --- Object Files ---
# Create .o paths for all sources
CU_OBJS = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SRCS))
CPP_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS))
OBJS = $(CU_OBJS) $(CPP_OBJS)

# --- Python & Pybind11 Flags ---
PYBIND_INCLUDES = $(shell python3 -m pybind11 --includes)
PYTHON_FLAGS = $(shell python3-config --cflags --ldflags --embed)

# --- Compiler & Linker Flags ---
# Unified flags for NVCC. NVCC passes C++ specific flags to the host compiler.
# -Xcompiler -fPIC is crucial for creating a shared library.
# We add the Pybind/Python includes here for compiling the bindings file.
NVCC_FLAGS = -I$(INCLUDE_DIR) $(PYBIND_INCLUDES) \
             -O3 -std=c++17 --compiler-options -Wall,-fPIC \
             -rdc=true -gencode arch=compute_89,code=lto_89 --extended-lambda

# We need to tell NVCC to create a shared library for the final linking step
LINK_FLAGS = -shared

# --- Final Target Path ---
TARGET_LIB = $(BIN_DIR)/$(PYTHON_MODULE)

# --- Rules ---

# Default rule: Build the Python module
all: $(TARGET_LIB)

# Rule to link the final Python module (shared library) using NVCC
$(TARGET_LIB): $(OBJS)
	@mkdir -p $(@D)
	@echo "Linking with NVCC to create Python module..."
	$(NVCC) $(LINK_FLAGS) -o $@ $^ $(PYTHON_FLAGS)

# Pattern rule to compile .cu files into .o files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(@D)
	@echo "Compiling CUDA $< -> $@"
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Pattern rule to compile .cpp files into .o files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	@echo "Compiling C++ $< -> $@"
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean rule to remove generated files
clean:
	@echo "Cleaning up..."
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Phony targets are not files
.PHONY: all clean