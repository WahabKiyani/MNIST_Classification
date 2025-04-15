#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64

typedef struct {
    double** W1;
    double** W2;
    double* b1;
    double* b2;
} NeuralNetwork;



__global__ void forward_hidden_layer(double* d_input, double* d_W1, double* d_b1, double* d_hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HIDDEN_SIZE) {
        double sum = d_b1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
            sum += d_W1[i * INPUT_SIZE + j] * d_input[j];
        d_hidden[i] = (sum > 0) ? sum : 0.0;  // ReLU
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



double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
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

void softmax(double* x, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i]);
        sum += x[i];
    }
    for (int i = 0; i < size; i++)
        x[i] /= sum;
}



NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    return net;
}

void gpu_forward(NeuralNetwork* net, double* input, double* hidden, double* output) {
    double *d_input, *d_W1, *d_b1, *d_hidden, *d_W2, *d_b2, *d_output;
    size_t input_bytes = INPUT_SIZE * sizeof(double);
    size_t hidden_bytes = HIDDEN_SIZE * sizeof(double);
    size_t output_bytes = OUTPUT_SIZE * sizeof(double);
    size_t W1_bytes = HIDDEN_SIZE * INPUT_SIZE * sizeof(double);
    size_t W2_bytes = OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double);

    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_W1, W1_bytes);
    cudaMalloc(&d_b1, hidden_bytes);
    cudaMalloc(&d_hidden, hidden_bytes);
    cudaMalloc(&d_W2, W2_bytes);
    cudaMalloc(&d_b2, output_bytes);
    cudaMalloc(&d_output, output_bytes);

    double* flat_W1 = (double*)malloc(W1_bytes);
    double* flat_W2 = (double*)malloc(W2_bytes);

    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            flat_W1[i * INPUT_SIZE + j] = net->W1[i][j];

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            flat_W2[i * HIDDEN_SIZE + j] = net->W2[i][j];

    cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, flat_W1, W1_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, net->b1, hidden_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, flat_W2, W2_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, net->b2, output_bytes, cudaMemcpyHostToDevice);

    int blockSize = 128;
    int gridSize_hidden = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    int gridSize_output = (OUTPUT_SIZE + blockSize - 1) / blockSize;

    forward_hidden_layer<<<gridSize_hidden, blockSize>>>(d_input, d_W1, d_b1, d_hidden);
    cudaDeviceSynchronize();

    forward_output_layer<<<gridSize_output, blockSize>>>(d_hidden, d_W2, d_b2, d_output);
    cudaDeviceSynchronize();

    cudaMemcpy(hidden, d_hidden, hidden_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost);
    softmax(output, OUTPUT_SIZE);

    free(flat_W1); free(flat_W2);
    cudaFree(d_input); cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_hidden);
    cudaFree(d_W2); cudaFree(d_b2); cudaFree(d_output);
}

void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target) {
    double d_output[OUTPUT_SIZE], d_hidden[HIDDEN_SIZE];

    for (int i = 0; i < OUTPUT_SIZE; i++)
        d_output[i] = output[i] - target[i];

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        d_hidden[i] = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++)
            d_hidden[i] += net->W2[j][i] * d_output[j];
        d_hidden[i] *= (hidden[i] > 0);
    }

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] -= LEARNING_RATE * d_output[i] * hidden[j];

    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] -= LEARNING_RATE * d_hidden[i] * input[j];

    for (int i = 0; i < OUTPUT_SIZE; i++)
        net->b2[i] -= LEARNING_RATE * d_output[i];

    for (int i = 0; i < HIDDEN_SIZE; i++)
        net->b1[i] -= LEARNING_RATE * d_hidden[i];
}

void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i++) {
            double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
            gpu_forward(net, images[i], hidden, output);
            backward(net, images[i], hidden, output, labels[i]);

            for (int k = 0; k < OUTPUT_SIZE; k++) loss -= labels[i][k] * log(output[k]);
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));
}

void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages) {
    int correct = 0;
    for (int i = 0; i < numImages; i++) {
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
        gpu_forward(net, images[i], hidden, output);
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
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
    freeMatrix(net->W1, HIDDEN_SIZE);
    freeMatrix(net->W2, OUTPUT_SIZE);
    free(net->b1); free(net->b2); free(net);
}

int main() {
    printf("MNIST Neural Network (CUDA)\n");

    double** train_images = loadMNISTImages("data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("data/t10k-labels.idx1-ubyte", 10000);

    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, 60000);
    evaluate(net, test_images, test_labels, 10000);

    freeNetwork(net);
    return 0;
}
