# Neural Network Profiler

This project contains a simple neural network (`nn.c`) and a Makefile for compiling, running, and profiling the program using `gprof`.

## Requirements
**Core:**
- GCC
- CUDA Toolkit (V2-V4)
- OpenACC (V5)

**Optional Profiling:**
- gprof
- gprof2dot
- Graphviz
- NVProf

## Installation
```bash
# Ubuntu/Debian
sudo apt install gcc nvidia-cuda-toolkit graphviz python3-gprof2dot

```


## Implementations Overview

| Version | Description | Key Features |
|---------|-------------|-------------|
| **V1**  | Baseline CPU | Sequential execution, reference implementation |
| **V2**  | Naive GPU   | Basic CUDA port, minimal optimizations |
| **V3**  | Optimized GPU | Advanced CUDA optimizations (memory hierarchy, occupancy tuning) |
| **V4**  | Tensor Core  | V3 + NVIDIA Tensor Core acceleration |
| **V5**  | OpenACC     | Directive-based GPU acceleration |

## Makefile Targets

### 1. Build and Run

To compile and execute the program:

```sh
make run
```

This:
- Creates the `bin/` directory if it doesn't exist.
- Compiles `nn.c` into `bin/nn.exe` with profiling enabled (`-pg` flag).
- Runs `bin/nn.exe`.

### 2. Profiling with `gprof`

To analyze performance using `gprof`, run:

```sh
make gprof
```

This:
- Runs `bin/nn.exe` to generate a profiling report (`gmon.out`).
- Moves `gmon.out` to `bin/`.
- Uses `gprof` to generate a human-readable profiling report (`prof_analysis/gprof_analysis.txt`).
- If `gprof2dot` and `dot` are available, generates a call graph (`prof_analysis/gprof_graph.png`).


### 3. Profiling with Nvidia nsight System Profiler
To analyze performance using `nsight`, run:
```sh
make nsys
```

This:
- Runs `bin/nn.exe` to generate a profiling file.
-The profiling file could be seen if you have installed nvidia nsight.


### 4. Cleaning the Project

To remove compiled files and profiling data:

```sh
make clean
```

This deletes the `bin/` and `prof_analysis/` directories.

## Profiling Output

After running `make gprof`, you will find:

- **`prof_analysis/gprof_analysis.txt`**: Text-based profiling data from `gprof`.
- **`prof_analysis/gprof_graph.png`** *(if dependencies installed)*: A graphical call graph of function execution times.

## Notes

- **Hardware Requirements**:
  - All GPU versions require NVIDIA CUDA-capable GPU
  - V4 specifically needs Volta/Turing/Ampere architecture for Tensor Cores
  - Minimum compute capability: 3.5 (V2-V3), 7.0+ (V4)

- **Performance Considerations**:
  - Results may vary based on GPU model and driver version
  - CPU version (V1) serves as baseline for speedup measurements
  - For accurate benchmarks, run on dedicated hardware


- **Troubleshooting**:
  - Clean builds with `make clean` if encountering issues
  - Verify CUDA toolkit version matches GPU architecture
  - OpenACC requires compatible compiler (PGI/NVIDIA HPC SDK)
---

