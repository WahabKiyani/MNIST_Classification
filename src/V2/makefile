# # # Compilers

NVCC = nvcc

# Flags

FLAGS = -O2 -Xcompiler -Wall,-Wextra,-ffast-math,-funroll-loops -pg
NVFLAGS = -gencode arch=compute_61,code=sm_61 \
          -gencode arch=compute_70,code=sm_70 \
          -gencode arch=compute_75,code=sm_75 \
          -gencode arch=compute_80,code=sm_80 \
          -gencode arch=compute_86,code=sm_86

# LDFLAGS = -lcublas -lcurand
LDFLAGS = -lcublas

# Files
SRC = nn.cu
EXE = bin/nn.exe
BIN_DIR = bin
PROF_DIR = prof_analysis

# Rules
all: $(EXE) run

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(PROF_DIR):
	mkdir -p $(PROF_DIR)

$(EXE): $(SRC) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $(EXE) $(SRC) -lm

run: $(EXE)
	./$(EXE)

gprof: $(EXE) | $(PROF_DIR)
	./$(EXE)  # Generate gmon.out
	mv gmon.out $(BIN_DIR)/gmon.out  # Move gmon.out to bin/
	gprof $(EXE) $(BIN_DIR)/gmon.out > $(PROF_DIR)/gprof_analysis.txt
	@if command -v gprof2dot >/dev/null 2>&1 && command -v dot >/dev/null 2>&1; then \
		gprof $(EXE) $(BIN_DIR)/gmon.out -q | gprof2dot -n 0 -e 0 | dot -Tpng -o $(PROF_DIR)/gprof_graph.png 2>/dev/null; \
		echo "Generated $(PROF_DIR)/gprof_graph.png"; \
	else \
		echo "WARNING: gprof2dot or dot not found. Skipping graph generation."; \
	fi

clean:
	rm -rf $(BIN_DIR) $(PROF_DIR)

