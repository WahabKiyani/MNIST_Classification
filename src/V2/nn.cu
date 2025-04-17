#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define BLOCK_SIZE 256
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <chrono>


typedef struct {
    double* W1;  // Flattened [HIDDEN_SIZE][INPUT_SIZE]
    double* W2;  // Flattened [OUTPUT_SIZE][HIDDEN_SIZE]
    double* b1;  // [HIDDEN_SIZE]
    double* b2;  // [OUTPUT_SIZE]
} NeuralNetwork;

typedef struct {
    double* d_W1;
    double* d_W2;
    double* d_b1;
    double* d_b2;
    double* d_input;
    double* d_hidden;
    double* d_output;
    double* d_d_output;
    double* d_d_hidden;
    double* d_target;
} NeuralNetworkDevice;

__global__ void softmax_kernel(double* input, double* output, int size) {
    __shared__ double max_val;
    __shared__ double sum;

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


void gpu_softmax(double* d_input, double* d_output, int size) {
    softmax_kernel<<<1, size>>>(d_input, d_output, size);
    cudaDeviceSynchronize(); 
}


__global__ void forward_hidden_layer(double* d_input, double* d_W1, double* d_b1, double* d_hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HIDDEN_SIZE) {
        double sum = d_b1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
            sum += d_W1[i * INPUT_SIZE + j] * d_input[j];
        d_hidden[i] = (sum > 0) ? sum : 0.0;  
    }
}

__global__ void forward_output_layer(double* d_hidden, double* d_W2, double* d_b2, double* d_output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < OUTPUT_SIZE) {
        double sum = d_b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            sum += d_W2[i * HIDDEN_SIZE + j] * d_hidden[j];
        d_output[i] = sum;
    }
}

__global__ void backward_output_layer(double* d_output, double* d_target, double* d_d_output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < OUTPUT_SIZE) {
        d_d_output[i] = d_output[i] - d_target[i];
    }
}

__global__ void backward_hidden_layer(double* d_hidden, double* d_W2, double* d_d_output, double* d_d_hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HIDDEN_SIZE) {
        double sum = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++)
            sum += d_W2[j * HIDDEN_SIZE + i] * d_d_output[j];
        d_d_hidden[i] = sum * ((d_hidden[i] > 0) ? 1.0 : 0.0);  
    }
}

__global__ void update_weights_W2(double* d_W2, double* d_b2, double* d_hidden, 
                                 double* d_d_output, double lr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < OUTPUT_SIZE) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            int idx = i * HIDDEN_SIZE + j;
            d_W2[idx] -= lr * d_d_output[i] * d_hidden[j];
        }
        d_b2[i] -= lr * d_d_output[i];
    }
}

__global__ void update_weights_W1(double* d_W1, double* d_b1, double* d_input, 
                                 double* d_d_hidden, double lr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HIDDEN_SIZE) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            int idx = i * INPUT_SIZE + j;
            d_W1[idx] -= lr * d_d_hidden[i] * d_input[j];
        }
        d_b1[i] -= lr * d_d_hidden[i];
    }
}



NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    net->W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++)
        net->W1[i] = ((double)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++)
        net->W2[i] = ((double)rand() / RAND_MAX) * 0.01;

    return net;
}

void setupDeviceMemory(NeuralNetwork* net, NeuralNetworkDevice* dev_net) {
    cudaMalloc(&dev_net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    cudaMalloc(&dev_net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&dev_net->d_b1, HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&dev_net->d_b2, OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&dev_net->d_input, INPUT_SIZE * sizeof(double));
    cudaMalloc(&dev_net->d_hidden, HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&dev_net->d_output, OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&dev_net->d_d_output, OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&dev_net->d_d_hidden, HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&dev_net->d_target, OUTPUT_SIZE * sizeof(double));

    cudaMemcpy(dev_net->d_W1, net->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_net->d_W2, net->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_net->d_b1, net->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_net->d_b2, net->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
}

void freeDeviceMemory(NeuralNetworkDevice* dev_net) {
    cudaFree(dev_net->d_W1);
    cudaFree(dev_net->d_W2);
    cudaFree(dev_net->d_b1);
    cudaFree(dev_net->d_b2);
    cudaFree(dev_net->d_input);
    cudaFree(dev_net->d_hidden);
    cudaFree(dev_net->d_output);
    cudaFree(dev_net->d_d_output);
    cudaFree(dev_net->d_d_hidden);
    cudaFree(dev_net->d_target);
}

void gpu_forward(NeuralNetworkDevice* dev_net, double* input, double* hidden, double* output) {
    cudaMemcpy(dev_net->d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    int gridSize_hidden = (HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int gridSize_output = (OUTPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    forward_hidden_layer<<<gridSize_hidden, BLOCK_SIZE>>>(dev_net->d_input, dev_net->d_W1, dev_net->d_b1, dev_net->d_hidden);
    forward_output_layer<<<gridSize_output, BLOCK_SIZE>>>(dev_net->d_hidden, dev_net->d_W2, dev_net->d_b2, dev_net->d_output);

    cudaMemcpy(hidden, dev_net->d_hidden, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    
    gpu_softmax(dev_net->d_output, dev_net->d_output, OUTPUT_SIZE);
    cudaMemcpy(output, dev_net->d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);



}

void gpu_backward(NeuralNetworkDevice* dev_net, double* input, double* target) {
    cudaMemcpy(dev_net->d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_net->d_target, target, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    int gridSize_output = (OUTPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int gridSize_hidden = (HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    backward_output_layer<<<gridSize_output, BLOCK_SIZE>>>(dev_net->d_output, dev_net->d_target, dev_net->d_d_output);
    backward_hidden_layer<<<gridSize_hidden, BLOCK_SIZE>>>(dev_net->d_hidden, dev_net->d_W2, dev_net->d_d_output, dev_net->d_d_hidden);

    update_weights_W2<<<gridSize_output, BLOCK_SIZE>>>(dev_net->d_W2, dev_net->d_b2, dev_net->d_hidden, dev_net->d_d_output, LEARNING_RATE);
    update_weights_W1<<<gridSize_hidden, BLOCK_SIZE>>>(dev_net->d_W1, dev_net->d_b1, dev_net->d_input, dev_net->d_d_hidden, LEARNING_RATE);
}

double** allocateMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++)
        mat[i] = (double*)malloc(cols * sizeof(double));
    return mat;
}

void freeMatrix(double** mat, int rows) {
    for (int i = 0; i < rows; i++)
        free(mat[i]);
    free(mat);
}





void train(NeuralNetwork* net, NeuralNetworkDevice* dev_net, double** images, double** labels, int numImages) {
    double* hidden = (double*)malloc(HIDDEN_SIZE * sizeof(double));
    double* output = (double*)malloc(OUTPUT_SIZE * sizeof(double));

   
    cudaEvent_t epoch_start, epoch_stop;
    cudaEventCreate(&epoch_start);
    cudaEventCreate(&epoch_stop);

   
    cudaEvent_t total_start, total_stop;
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);

    cudaEventRecord(total_start, 0); 

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        cudaEventRecord(epoch_start, 0);  

        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i++) {
            gpu_forward(dev_net, images[i], hidden, output);
            gpu_backward(dev_net, images[i], labels[i]);

            for (int k = 0; k < OUTPUT_SIZE; k++)
                loss -= labels[i][k] * log(output[k] + 1e-10);

            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
        }

        cudaEventRecord(epoch_stop, 0);      
        cudaEventSynchronize(epoch_stop);    

        float epoch_ms = 0;
        cudaEventElapsedTime(&epoch_ms, epoch_start, epoch_stop);
        float epoch_sec = epoch_ms / 1000.0f;

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.4f sec\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, epoch_sec);
    }

    cudaEventRecord(total_stop, 0);    
    cudaEventSynchronize(total_stop);

    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, total_start, total_stop);
    float total_sec = total_ms / 1000.0f;

    printf("Total Training Time: %.4f sec\n", total_sec);


    cudaEventDestroy(epoch_start);
    cudaEventDestroy(epoch_stop);
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);

    free(hidden);
    free(output);
}


void evaluate(NeuralNetworkDevice* dev_net, double** images, double** labels, int numImages) {
    double* hidden = (double*)malloc(HIDDEN_SIZE * sizeof(double));
    double* output = (double*)malloc(OUTPUT_SIZE * sizeof(double));
    int correct = 0;

    for (int i = 0; i < numImages; i++) {
        gpu_forward(dev_net, images[i], hidden, output);
        
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }

    free(hidden);
    free(output);
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
}

double** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) { perror("Opening image file"); exit(1); }
    fseek(file, 16, SEEK_SET);
    double** images = allocateMatrix(numImages, INPUT_SIZE);
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

double** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) { perror("Opening label file"); exit(1); }
    fseek(file, 8, SEEK_SET);
    double** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
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
    printf("MNIST Neural Network (Optimized CUDA)\n");

    double** train_images = loadMNISTImages("data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("data/t10k-labels.idx1-ubyte", 10000);

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

