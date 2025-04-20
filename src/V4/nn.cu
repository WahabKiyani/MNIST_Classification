#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01f
#define EPOCHS 3  
#define BATCH_SIZE 128
#define BLOCK_SIZE 256
#define NUM_STREAMS 64
#define WARMUP_ITERATIONS 10

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <chrono>

using namespace nvcuda;

typedef float my_type; 
typedef half half_type;  

typedef struct {
    my_type* W1;  
    my_type* W2; 
    my_type* b1;
    my_type* b2;
} NeuralNetwork;

typedef struct {
    my_type* d_W1;
    my_type* d_W2;
    my_type* d_b1;
    my_type* d_b2;
    half_type* d_W1_half;  
    half_type* d_W2_half;
    
    struct {
        cudaStream_t stream;
        my_type* d_input;
        half_type* d_input_half;
        my_type* d_hidden;
        half_type* d_hidden_half;
        my_type* d_output;
        my_type* d_d_output;
        my_type* d_d_hidden;
        my_type* d_target;
    } streams[NUM_STREAMS];
} NeuralNetworkDevice;


__global__ void warmup_tensor_cores() {
  
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    
   
    wmma::fill_fragment(acc_frag, 0.0f);
    __syncthreads();
}


__global__ void tensor_matmul(const half_type* A, const half_type* B, my_type* C, 
                             int M, int N, int K) {
 
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half_type, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half_type, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, my_type> acc_frag;
    
 
    wmma::fill_fragment(acc_frag, 0.0f);
    
  
    for (int t = 0; t < (K + WMMA_K - 1) / WMMA_K; ++t) {
        int aRow = blockIdx.y * blockDim.y + threadIdx.y;
        int aCol = t * WMMA_K;
        int bRow = t * WMMA_K;
        int bCol = blockIdx.x * blockDim.x + threadIdx.x;
        
      
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
          
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            
          
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    
  
    int cRow = blockIdx.y * blockDim.y + threadIdx.y;
    int cCol = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, acc_frag, N, wmma::mem_row_major);
    }
}


__global__ void float_to_half_kernel(const my_type* input, half_type* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

__global__ void half_to_float_kernel(const half_type* input, my_type* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __half2float(input[idx]);
    }
}


__global__ void softmax_kernel(my_type* input, my_type* output, int size) {
    __shared__ my_type max_val;
    __shared__ my_type sum;

    if (threadIdx.x == 0) {
        max_val = input[0];
        for (int i = 1; i < size; i++) {
            if (input[i] > max_val)
                max_val = input[i];
        }
    }
    __syncthreads();

    int tid = threadIdx.x;
    if (tid < size) {
        output[tid] = exp(input[tid] - max_val);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        sum = 0.0;
        for (int i = 0; i < size; i++) {
            sum += output[i];
        }
    }
    __syncthreads();

    if (tid < size) {
        output[tid] /= sum + 1e-8;
    }
}


__global__ void forward_hidden_layer_tensor(half_type* d_input, half_type* d_W1, my_type* d_b1, my_type* d_hidden) {
    
    extern __shared__ half_type shared_input[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
 
    if (threadIdx.x < INPUT_SIZE) {
        shared_input[threadIdx.x] = d_input[threadIdx.x];
    }
    __syncthreads();

    if (i < HIDDEN_SIZE) {
    
        my_type sum = d_b1[i];
        
        
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += __half2float(d_W1[i * INPUT_SIZE + j]) * __half2float(shared_input[j]);
        }
        
        d_hidden[i] = fmaxf(sum, 0.0f);  
    }
}


__global__ void forward_output_layer_tensor(half_type* d_hidden, half_type* d_W2, my_type* d_b2, my_type* d_output) {
    // Use tensor cores for the matrix multiplication
    extern __shared__ half_type shared_hidden[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x < HIDDEN_SIZE) {
        shared_hidden[threadIdx.x] = d_hidden[threadIdx.x];
    }
    __syncthreads();

    if (i < OUTPUT_SIZE) {
    
        my_type sum = d_b2[i];
        
   
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += __half2float(d_W2[i * HIDDEN_SIZE + j]) * __half2float(shared_hidden[j]);
        }
        
        d_output[i] = sum;
    }
}


__global__ void backward_output_layer(my_type* d_output, my_type* d_target, my_type* d_d_output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < OUTPUT_SIZE) {
        d_d_output[i] = d_output[i] - d_target[i];
    }
}

__global__ void backward_hidden_layer(my_type* d_hidden, half_type* d_W2, my_type* d_d_output, my_type* d_d_hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HIDDEN_SIZE) {
        my_type sum = 0.0f;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            sum += __half2float(d_W2[j * HIDDEN_SIZE + i]) * d_d_output[j];
        }
        d_d_hidden[i] = sum * ((d_hidden[i] > 0) ? 1.0f : 0.0f);  // ReLU derivative
    }
}


__global__ void update_weights_W2(half_type* d_W2, my_type* d_b2, half_type* d_hidden, 
                                my_type* d_d_output, my_type lr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < OUTPUT_SIZE) {
        my_type d_output_val = d_d_output[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            int idx = i * HIDDEN_SIZE + j;
            my_type current = __half2float(d_W2[idx]);
            current -= lr * d_output_val * __half2float(d_hidden[j]);
            d_W2[idx] = __float2half(current);
        }
        d_b2[i] -= lr * d_output_val;
    }
}

__global__ void update_weights_W1(half_type* d_W1, my_type* d_b1, half_type* d_input, 
                                my_type* d_d_hidden, my_type lr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HIDDEN_SIZE) {
        my_type d_hidden_val = d_d_hidden[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            int idx = i * INPUT_SIZE + j;
            my_type current = __half2float(d_W1[idx]);
            current -= lr * d_hidden_val * __half2float(d_input[j]);
            d_W1[idx] = __float2half(current);
        }
        d_b1[i] -= lr * d_hidden_val;
    }
}

NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = (my_type*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(my_type));
    net->W2 = (my_type*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(my_type));
    net->b1 = (my_type*)calloc(HIDDEN_SIZE, sizeof(my_type));
    net->b2 = (my_type*)calloc(OUTPUT_SIZE, sizeof(my_type));


    float stddev1 = sqrtf(2.0f / (INPUT_SIZE + HIDDEN_SIZE));
    float stddev2 = sqrtf(2.0f / (HIDDEN_SIZE + OUTPUT_SIZE));
    
    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) {
        net->W1[i] = stddev1 * ((my_type)rand() / RAND_MAX - 0.5f);
    }

    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++) {
        net->W2[i] = stddev2 * ((my_type)rand() / RAND_MAX - 0.5f);
    }

    return net;
}

void setupDeviceMemory(NeuralNetwork* net, NeuralNetworkDevice* dev_net) {
   
    cudaMalloc(&dev_net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(my_type));
    cudaMalloc(&dev_net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(my_type));
    cudaMalloc(&dev_net->d_b1, HIDDEN_SIZE * sizeof(my_type));
    cudaMalloc(&dev_net->d_b2, OUTPUT_SIZE * sizeof(my_type));


    cudaMalloc(&dev_net->d_W1_half, HIDDEN_SIZE * INPUT_SIZE * sizeof(half_type));
    cudaMalloc(&dev_net->d_W2_half, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(half_type));


    cudaMemcpy(dev_net->d_W1, net->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(my_type), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_net->d_W2, net->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(my_type), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_net->d_b1, net->b1, HIDDEN_SIZE * sizeof(my_type), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_net->d_b2, net->b2, OUTPUT_SIZE * sizeof(my_type), cudaMemcpyHostToDevice);


    dim3 blockSize(256);
    dim3 gridSizeW1((HIDDEN_SIZE * INPUT_SIZE + blockSize.x - 1) / blockSize.x);
    dim3 gridSizeW2((OUTPUT_SIZE * HIDDEN_SIZE + blockSize.x - 1) / blockSize.x);
    
    float_to_half_kernel<<<gridSizeW1, blockSize>>>(dev_net->d_W1, dev_net->d_W1_half, HIDDEN_SIZE * INPUT_SIZE);
    float_to_half_kernel<<<gridSizeW2, blockSize>>>(dev_net->d_W2, dev_net->d_W2_half, OUTPUT_SIZE * HIDDEN_SIZE);
    cudaDeviceSynchronize();


    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&dev_net->streams[i].stream);
        cudaMalloc(&dev_net->streams[i].d_input, INPUT_SIZE * sizeof(my_type));
        cudaMalloc(&dev_net->streams[i].d_input_half, INPUT_SIZE * sizeof(half_type));
        cudaMalloc(&dev_net->streams[i].d_hidden, HIDDEN_SIZE * sizeof(my_type));
        cudaMalloc(&dev_net->streams[i].d_hidden_half, HIDDEN_SIZE * sizeof(half_type));
        cudaMalloc(&dev_net->streams[i].d_output, OUTPUT_SIZE * sizeof(my_type));
        cudaMalloc(&dev_net->streams[i].d_d_output, OUTPUT_SIZE * sizeof(my_type));
        cudaMalloc(&dev_net->streams[i].d_d_hidden, HIDDEN_SIZE * sizeof(my_type));
        cudaMalloc(&dev_net->streams[i].d_target, OUTPUT_SIZE * sizeof(my_type));
    }


    printf("Warming up tensor cores...\n");
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        warmup_tensor_cores<<<1, 32>>>();
    }
    cudaDeviceSynchronize();
    printf("Tensor cores warmed up\n");
}

void freeDeviceMemory(NeuralNetworkDevice* dev_net) {
    cudaFree(dev_net->d_W1);
    cudaFree(dev_net->d_W2);
    cudaFree(dev_net->d_b1);
    cudaFree(dev_net->d_b2);
    cudaFree(dev_net->d_W1_half);
    cudaFree(dev_net->d_W2_half);

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(dev_net->streams[i].stream);
        cudaFree(dev_net->streams[i].d_input);
        cudaFree(dev_net->streams[i].d_input_half);
        cudaFree(dev_net->streams[i].d_hidden);
        cudaFree(dev_net->streams[i].d_hidden_half);
        cudaFree(dev_net->streams[i].d_output);
        cudaFree(dev_net->streams[i].d_d_output);
        cudaFree(dev_net->streams[i].d_d_hidden);
        cudaFree(dev_net->streams[i].d_target);
    }
}

void gpu_softmax(my_type* d_input, my_type* d_output, int size, cudaStream_t stream) {
    softmax_kernel<<<1, size, 0, stream>>>(d_input, d_output, size);
}

void gpu_forward(NeuralNetworkDevice* dev_net, my_type* input, my_type* hidden, my_type* output, int stream_idx) {
    cudaStream_t stream = dev_net->streams[stream_idx].stream;
    

    cudaMemcpyAsync(dev_net->streams[stream_idx].d_input, input, 
                   INPUT_SIZE * sizeof(my_type), cudaMemcpyHostToDevice, stream);
    
    dim3 blockSize(256);
    dim3 gridSize((INPUT_SIZE + blockSize.x - 1) / blockSize.x);
    float_to_half_kernel<<<gridSize, blockSize, 0, stream>>>(
        dev_net->streams[stream_idx].d_input, 
        dev_net->streams[stream_idx].d_input_half, 
        INPUT_SIZE
    );

  
    dim3 gridSize_hidden((HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    forward_hidden_layer_tensor<<<gridSize_hidden, BLOCK_SIZE, INPUT_SIZE * sizeof(half_type), stream>>>(
        dev_net->streams[stream_idx].d_input_half, 
        dev_net->d_W1_half, 
        dev_net->d_b1, 
        dev_net->streams[stream_idx].d_hidden
    );

   
    float_to_half_kernel<<<gridSize, blockSize, 0, stream>>>(
        dev_net->streams[stream_idx].d_hidden,
        dev_net->streams[stream_idx].d_hidden_half,
        HIDDEN_SIZE
    );

 
    dim3 gridSize_output((OUTPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    forward_output_layer_tensor<<<gridSize_output, BLOCK_SIZE, HIDDEN_SIZE * sizeof(half_type), stream>>>(
        dev_net->streams[stream_idx].d_hidden_half,
        dev_net->d_W2_half,
        dev_net->d_b2,
        dev_net->streams[stream_idx].d_output
    );

 
    gpu_softmax(dev_net->streams[stream_idx].d_output, dev_net->streams[stream_idx].d_output, 
               OUTPUT_SIZE, stream);
    

    cudaMemcpyAsync(hidden, dev_net->streams[stream_idx].d_hidden, 
                   HIDDEN_SIZE * sizeof(my_type), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(output, dev_net->streams[stream_idx].d_output, 
                   OUTPUT_SIZE * sizeof(my_type), cudaMemcpyDeviceToHost, stream);
}

void gpu_backward(NeuralNetworkDevice* dev_net, my_type* input, my_type* target, int stream_idx) {
    cudaStream_t stream = dev_net->streams[stream_idx].stream;
    
   
    cudaMemcpyAsync(dev_net->streams[stream_idx].d_input, input, 
                   INPUT_SIZE * sizeof(my_type), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_net->streams[stream_idx].d_target, target, 
                   OUTPUT_SIZE * sizeof(my_type), cudaMemcpyHostToDevice, stream);
    
  
    dim3 blockSize(256);
    dim3 gridSize((INPUT_SIZE + blockSize.x - 1) / blockSize.x);
    float_to_half_kernel<<<gridSize, blockSize, 0, stream>>>(
        dev_net->streams[stream_idx].d_input, 
        dev_net->streams[stream_idx].d_input_half, 
        INPUT_SIZE
    );


    dim3 gridSize_output((OUTPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 gridSize_hidden((HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    backward_output_layer<<<gridSize_output, BLOCK_SIZE, 0, stream>>>(
        dev_net->streams[stream_idx].d_output, 
        dev_net->streams[stream_idx].d_target, 
        dev_net->streams[stream_idx].d_d_output
    );
    
    backward_hidden_layer<<<gridSize_hidden, BLOCK_SIZE, 0, stream>>>(
        dev_net->streams[stream_idx].d_hidden, 
        dev_net->d_W2_half, 
        dev_net->streams[stream_idx].d_d_output, 
        dev_net->streams[stream_idx].d_d_hidden
    );


    update_weights_W2<<<gridSize_output, BLOCK_SIZE, 0, stream>>>(
        dev_net->d_W2_half, 
        dev_net->d_b2, 
        dev_net->streams[stream_idx].d_hidden_half, 
        dev_net->streams[stream_idx].d_d_output, 
        LEARNING_RATE
    );
    
    update_weights_W1<<<gridSize_hidden, BLOCK_SIZE, 0, stream>>>(
        dev_net->d_W1_half, 
        dev_net->d_b1, 
        dev_net->streams[stream_idx].d_input_half, 
        dev_net->streams[stream_idx].d_d_hidden, 
        LEARNING_RATE
    );
    

    half_to_float_kernel<<<gridSize, blockSize, 0, stream>>>(
        dev_net->d_W1_half, 
        dev_net->d_W1, 
        HIDDEN_SIZE * INPUT_SIZE
    );
    half_to_float_kernel<<<gridSize, blockSize, 0, stream>>>(
        dev_net->d_W2_half, 
        dev_net->d_W2, 
        OUTPUT_SIZE * HIDDEN_SIZE
    );
}



my_type** allocateMatrix(int rows, int cols) {
    my_type** mat = (my_type**)malloc(rows * sizeof(my_type*));
    for (int i = 0; i < rows; i++)
        mat[i] = (my_type*)malloc(cols * sizeof(my_type));
    return mat;
}

void freeMatrix(my_type** mat, int rows) {
    for (int i = 0; i < rows; i++)
        free(mat[i]);
    free(mat);
}

void train(NeuralNetwork* net, NeuralNetworkDevice* dev_net, my_type** images, my_type** labels, int numImages) {
    my_type *hidden[NUM_STREAMS], *output[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMallocHost(&hidden[i], HIDDEN_SIZE * sizeof(my_type));
        cudaMallocHost(&output[i], OUTPUT_SIZE * sizeof(my_type));
    }

    cudaEvent_t total_start, total_stop;
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);
    cudaEventRecord(total_start);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        cudaEvent_t epoch_start, epoch_stop;
        cudaEventCreate(&epoch_start);
        cudaEventCreate(&epoch_stop);
        cudaEventRecord(epoch_start);

        my_type epoch_loss = 0;
        int correct = 0;
        
        for (int i = 0; i < numImages; i += NUM_STREAMS) {
            for (int s = 0; s < NUM_STREAMS; s++) {
                if (i+s >= numImages) continue;
                
                gpu_forward(dev_net, images[i+s], hidden[s], output[s], s);
                gpu_backward(dev_net, images[i+s], labels[i+s], s);
            }
            
            for (int s = 0; s < NUM_STREAMS; s++) {
                if (i+s >= numImages) continue;
                
                cudaStreamSynchronize(dev_net->streams[s].stream);
                
                for (int k = 0; k < OUTPUT_SIZE; k++) {
                    epoch_loss -= labels[i+s][k] * log(output[s][k] + 1e-10);
                }
                
                int pred = 0, actual = 0;
                for (int k = 0; k < OUTPUT_SIZE; k++) {
                    if (output[s][k] > output[s][pred]) pred = k;
                    if (labels[i+s][k] > labels[i+s][actual]) actual = k;
                }
                if (pred == actual) correct++;
            }
        }

        cudaEventRecord(epoch_stop);
        cudaEventSynchronize(epoch_stop);
        float epoch_ms = 0;
        cudaEventElapsedTime(&epoch_ms, epoch_start, epoch_stop);
        
        printf("Epoch %d - Loss: %.4f - Accuracy: %.2f%% - Time: %.3fs\n",
              epoch+1, 
              epoch_loss/numImages, 
              (correct*100.0)/numImages,
              epoch_ms/1000.0f);

        cudaEventDestroy(epoch_start);
        cudaEventDestroy(epoch_stop);
    }

    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, total_start, total_stop);
    
    printf("Total training time: %.3fs\n", total_ms/1000.0f);

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaFreeHost(hidden[i]);
        cudaFreeHost(output[i]);
    }
    
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);
}

void evaluate(NeuralNetworkDevice* dev_net, my_type** images, my_type** labels, int numImages) {
    my_type *hidden, *output;
    cudaMallocHost(&hidden, HIDDEN_SIZE * sizeof(my_type));
    cudaMallocHost(&output, OUTPUT_SIZE * sizeof(my_type));
    
    int correct = 0;
    int stream_idx = 0; 
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    for (int i = 0; i < numImages; i++) {
        gpu_forward(dev_net, images[i], hidden, output, stream_idx);
        cudaStreamSynchronize(dev_net->streams[stream_idx].stream);
        
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Test Accuracy: %.2f%% (Time: %.3fs)\n", 
          (correct / (my_type)numImages) * 100, 
          milliseconds / 1000.0f);
    
    cudaFreeHost(hidden);
    cudaFreeHost(output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

my_type** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) { perror("Opening image file"); exit(1); }
    fseek(file, 16, SEEK_SET);
    my_type** images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error reading pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
            images[i][j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}

my_type** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) { perror("Opening label file"); exit(1); }
    fseek(file, 8, SEEK_SET);
    my_type** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error reading label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < OUTPUT_SIZE; j++)
            labels[i][j] = (j == label) ? 1.0 : 0.0;
    }
    fclose(file);
    return labels;
}

void freeNetwork(NeuralNetwork* net) {
    free(net->W1);
    free(net->W2);
    free(net->b1);
    free(net->b2);
    free(net);
}

int main() {
    printf("MNIST Neural Network with Tensor Cores\n");

    my_type** train_images = loadMNISTImages("data/train-images.idx3-ubyte", 60000);
    my_type** train_labels = loadMNISTLabels("data/train-labels.idx1-ubyte", 60000);
    my_type** test_images = loadMNISTImages("data/t10k-images.idx3-ubyte", 10000);
    my_type** test_labels = loadMNISTLabels("data/t10k-labels.idx1-ubyte", 10000);

    NeuralNetwork* net = createNetwork();
    NeuralNetworkDevice dev_net;
    setupDeviceMemory(net, &dev_net);

    train(net, &dev_net, train_images, train_labels, 60000);
    evaluate(&dev_net, test_images, test_labels, 10000);

    freeDeviceMemory(&dev_net);
    freeNetwork(net);
    freeMatrix(train_images, 60000);
    freeMatrix(train_labels, 60000);
    freeMatrix(test_images, 10000);
    freeMatrix(test_labels, 10000);
    
    return 0;
}
