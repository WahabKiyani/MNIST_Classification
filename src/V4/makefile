# # # Compilers
NVCC = nvcc
NSYS = nsys

# # # Flags
FLAGS = -O2 -Xcompiler -Wall,-Wextra,-ffast-math,-funroll-loops
NVFLAGS = -gencode arch=compute_61,code=sm_61 \
          -gencode arch=compute_70,code=sm_70 \
          -gencode arch=compute_75,code=sm_75 \
          -gencode arch=compute_80,code=sm_80 \
          -gencode arch=compute_86,code=sm_86

LDFLAGS = -lcublas

# # # Files
SRC = nn.cu
EXE = bin/nn.exe
BIN_DIR = bin
PROFILING_DIR = profiling

# # # Rules

all: $(EXE) run

$(BIN_DIR):
	mkdir -p $(BIN_DIR)


$(PROFILING_DIR):
	mkdir -p $(PROFILING_DIR)

$(EXE): $(SRC) | $(BIN_DIR)
	$(NVCC) $(NVFLAGS) $(FLAGS) -o $(EXE) $(SRC) -lm $(LDFLAGS)

run: $(EXE)
	./$(EXE)

nsys: $(EXE) | $(PROFILING_DIR)
	$(NSYS) profile --stats=true --output $(PROFILING_DIR)/nsys_report_`date +%Y%m%d_%H%M%S` ./$(EXE)

clean:
	rm -rf $(BIN_DIR) $(PROFILING_DIR)
