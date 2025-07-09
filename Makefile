# Compiler
NVCC = nvcc

# Executable name
TARGET = program

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
INCLUDE_DIR = include

# Find all .cu files in the source directory
SRCS = $(wildcard $(SRC_DIR)/*.cu)

# Create object file names (e.g., obj/main.o) from source file names (e.g., src/main.cu)
OBJS = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRCS))

# Compiler flags
CPPFLAGS = -I$(INCLUDE_DIR) -O3 -rdc=true --gpu-architecture=sm_89 -lineinfo --compiler-options -Wall --extended-lambda

# The final executable path
TARGET_EXEC = $(BIN_DIR)/$(TARGET)

# --- Rules ---

# Default rule: Build everything
all: $(TARGET_EXEC)

# Rule to link the final executable
# The input ($^) is all the object files (OBJS)
$(TARGET_EXEC): $(OBJS)
	@mkdir -p $(@D)
	@echo "Linking..."
	$(NVCC) $(CPPFLAGS) -o $@ $^

# Pattern rule to compile .cu files into .o files
# This rule creates object files in the OBJ_DIR
# The input ($<) is the first prerequisite (the .cu file)
# The output ($@) is the target (the .o file)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(@D)
	@echo "Compiling $< -> $@"
	$(NVCC) $(CPPFLAGS) -c $< -o $@

# Clean rule to remove generated files
clean:
	@echo "Cleaning up..."
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Phony targets are not files
.PHONY: all clean