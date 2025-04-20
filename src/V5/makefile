# # Compiler and flags
# CC = gcc
# CFLAGS = -O3 -fopenacc -march=native -ffast-math -Wall -Wextra
# LDFLAGS = -lm -fopenacc

# # Source files and output folders
# SRCS = nn.c
# OBJS = $(patsubst %.c, build/%.o, $(SRCS))
# EXEC = build/mnist_nn

# # Default target
# all: build $(EXEC)

# # Create build directory if not exists
# build:
# 	mkdir -p build

# # Link object files into the executable
# $(EXEC): $(OBJS)
# 	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# # Compile source files into object files in the build folder
# build/%.o: %.c
# 	$(CC) $(CFLAGS) -c $< -o $@

# # Run the program
# run: all
# 	./$(EXEC)

# # Clean up build files
# clean:
# 	rm -rf build

# .PHONY: all clean run build


# Compiler and flags
CC = gcc
CFLAGS = -O3 -fopenacc -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -fopenacc

# Source files and output folders
SRCS = nn.c
OBJS = $(patsubst %.c, build/%.o, $(SRCS))
EXEC = build/mnist_nn

# Nsight Systems
NSYS = nsys
PROFILING_DIR = profiling

# Default target
all: build $(EXEC)

# Create build directory if not exists
build:
	mkdir -p build

# Create profiling directory if not exists
$(PROFILING_DIR):
	mkdir -p $(PROFILING_DIR)

# Link object files into the executable
$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile source files into object files in the build folder
build/%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Run the program
run: all
	./$(EXEC)

# Nsight profiling
nsys: all | $(PROFILING_DIR)
	$(NSYS) profile --stats=true --output $(PROFILING_DIR)/nsys_report_`date +%Y%m%d_%H%M%S` ./$(EXEC)

# Clean up build files
clean:
	rm -rf build $(PROFILING_DIR)

.PHONY: all clean run build nsys

