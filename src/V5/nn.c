#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01f
#define EPOCHS 3  
#define BATCH_SIZE 128
#define NUM_STREAMS 8

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <openacc.h>
#include <omp.h>

typedef float my_type;

typedef struct {
    my_type* W1;  
    my_type* W2; 
    my_type* b1;
    my_type* b2;
} NeuralNetwork;

void softmax(my_type* input, my_type* output, int size) {
    my_type max_val = input[0];
    
    // Find maximum value for numerical stability
    #pragma acc parallel loop reduction(max:max_val) present(input[0:size])
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val)
            max_val = input[i];
    }

    my_type sum = 0.0f;
    
    // Calculate exponentials and sum
    #pragma acc parallel loop reduction(+:sum) present(input[0:size], output[0:size])
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    // Normalize
    #pragma acc parallel loop present(output[0:size])
    for (int i = 0; i < size; i++) {
        output[i] /= (sum + 1e-8f);
    }
}

NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    
    // Allocate host memory
    net->W1 = (my_type*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(my_type));
    net->W2 = (my_type*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(my_type));
    net->b1 = (my_type*)calloc(HIDDEN_SIZE, sizeof(my_type));
    net->b2 = (my_type*)calloc(OUTPUT_SIZE, sizeof(my_type));

    // Xavier/Glorot initialization
    float stddev1 = sqrtf(2.0f / (INPUT_SIZE + HIDDEN_SIZE));
    float stddev2 = sqrtf(2.0f / (HIDDEN_SIZE + OUTPUT_SIZE));
    
    srand(42); // Fixed seed for reproducibility
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) {
        net->W1[i] = stddev1 * ((my_type)rand() / RAND_MAX - 0.5f);
    }

    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++) {
        net->W2[i] = stddev2 * ((my_type)rand() / RAND_MAX - 0.5f);
    }

    // Transfer data to device
    #pragma acc enter data copyin(net->W1[0:HIDDEN_SIZE*INPUT_SIZE], \
                                 net->W2[0:OUTPUT_SIZE*HIDDEN_SIZE], \
                                 net->b1[0:HIDDEN_SIZE], \
                                 net->b2[0:OUTPUT_SIZE])

    return net;
}

void forward_pass(NeuralNetwork* net, my_type* input, my_type* hidden, my_type* output) {
    // Forward pass
    #pragma acc data present(net->W1[0:HIDDEN_SIZE*INPUT_SIZE], \
                            net->W2[0:OUTPUT_SIZE*HIDDEN_SIZE], \
                            net->b1[0:HIDDEN_SIZE], net->b2[0:OUTPUT_SIZE], \
                            input[0:INPUT_SIZE], hidden[0:HIDDEN_SIZE], output[0:OUTPUT_SIZE])
    {
        // Hidden layer with ReLU activation
        #pragma acc parallel loop gang vector
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            my_type sum = net->b1[i];
            #pragma acc loop reduction(+:sum)
            for (int j = 0; j < INPUT_SIZE; j++) {
                sum += net->W1[i * INPUT_SIZE + j] * input[j];
            }
            hidden[i] = fmaxf(sum, 0.0f);  // ReLU
        }

        // Output layer before softmax
        #pragma acc parallel loop gang vector
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            my_type sum = net->b2[i];
            #pragma acc loop reduction(+:sum)
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                sum += net->W2[i * HIDDEN_SIZE + j] * hidden[j];
            }
            output[i] = sum;
        }

        // Apply softmax activation
        softmax(output, output, OUTPUT_SIZE);
    }
}

void backward_pass(NeuralNetwork* net, my_type* input, my_type* hidden, my_type* output, 
                  my_type* target, my_type* d_output, my_type* d_hidden) {
    #pragma acc data present(net->W1[0:HIDDEN_SIZE*INPUT_SIZE], \
                            net->W2[0:OUTPUT_SIZE*HIDDEN_SIZE], \
                            net->b1[0:HIDDEN_SIZE], net->b2[0:OUTPUT_SIZE], \
                            input[0:INPUT_SIZE], hidden[0:HIDDEN_SIZE], \
                            output[0:OUTPUT_SIZE], target[0:OUTPUT_SIZE], \
                            d_output[0:OUTPUT_SIZE], d_hidden[0:HIDDEN_SIZE])
    {
        // Calculate output layer gradients (softmax derivative)
        #pragma acc parallel loop gang vector
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            d_output[i] = output[i] - target[i];
        }

        // Calculate hidden layer gradients (backprop through ReLU)
        #pragma acc parallel loop gang vector
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            my_type sum = 0.0f;
            #pragma acc loop reduction(+:sum)
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                sum += net->W2[j * HIDDEN_SIZE + i] * d_output[j];
            }
            // ReLU derivative = (x > 0) ? 1 : 0
            d_hidden[i] = (hidden[i] > 0.0f) ? sum : 0.0f;
        }

        // Update output layer weights and biases
        #pragma acc parallel loop gang vector
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            #pragma acc loop
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                net->W2[i * HIDDEN_SIZE + j] -= LEARNING_RATE * d_output[i] * hidden[j];
            }
            net->b2[i] -= LEARNING_RATE * d_output[i];
        }

        // Update hidden layer weights and biases
        #pragma acc parallel loop gang vector
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            #pragma acc loop
            for (int j = 0; j < INPUT_SIZE; j++) {
                net->W1[i * INPUT_SIZE + j] -= LEARNING_RATE * d_hidden[i] * input[j];
            }
            net->b1[i] -= LEARNING_RATE * d_hidden[i];
        }
    }
}

// Fixed multi-dimensional array handling in OpenACC
void train(NeuralNetwork* net, my_type** images, my_type** labels, int numImages) {
   
    
    my_type* hidden = (my_type*)malloc(HIDDEN_SIZE * sizeof(my_type));
    my_type* output = (my_type*)malloc(OUTPUT_SIZE * sizeof(my_type));
    my_type* d_output = (my_type*)malloc(OUTPUT_SIZE * sizeof(my_type));
    my_type* d_hidden = (my_type*)malloc(HIDDEN_SIZE * sizeof(my_type));
    
    // Create buffers for current sample to avoid non-contiguous memory access
    my_type* current_image = (my_type*)malloc(INPUT_SIZE * sizeof(my_type));
    my_type* current_label = (my_type*)malloc(OUTPUT_SIZE * sizeof(my_type));
    
    #pragma acc enter data create(hidden[0:HIDDEN_SIZE], \
                                output[0:OUTPUT_SIZE], \
                                d_output[0:OUTPUT_SIZE], \
                                d_hidden[0:HIDDEN_SIZE], \
                                current_image[0:INPUT_SIZE], \
                                current_label[0:OUTPUT_SIZE])
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double epoch_loss = 0.0;
        int correct = 0;
        double start_time = omp_get_wtime();
        
        // Process full dataset
        for (int i = 0; i < numImages; i++) {
            // Copy current sample to contiguous memory
            for (int j = 0; j < INPUT_SIZE; j++) {
                current_image[j] = images[i][j];
            }
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                current_label[j] = labels[i][j];
            }
            
            // Update device memory with current sample
            #pragma acc update device(current_image[0:INPUT_SIZE])
            #pragma acc update device(current_label[0:OUTPUT_SIZE])
            
            forward_pass(net, current_image, hidden, output);
            backward_pass(net, current_image, hidden, output, current_label, d_output, d_hidden);
            
            // Copy output back for loss calculation
            #pragma acc update host(output[0:OUTPUT_SIZE])
            
            // Calculate cross-entropy loss
            my_type sample_loss = 0.0f;
            for (int k = 0; k < OUTPUT_SIZE; k++) {
                sample_loss -= current_label[k] * logf(output[k] + 1e-10f);
            }
            epoch_loss += sample_loss;
            
            // Calculate accuracy
            int pred = 0, actual = 0;
            for (int k = 0; k < OUTPUT_SIZE; k++) {
                if (output[k] > output[pred]) pred = k;
                if (current_label[k] > current_label[actual]) actual = k;
            }
            if (pred == actual) correct++;
            
          
        }

        double epoch_time = omp_get_wtime() - start_time;
        printf("Epoch %d - Loss: %.4f - Accuracy: %.2f%% - Time: %.3fs\n",
              epoch+1, 
              epoch_loss/numImages, 
              (correct*100.0)/numImages,
              epoch_time);
    }
    
    #pragma acc exit data delete(hidden[0:HIDDEN_SIZE], \
                               output[0:OUTPUT_SIZE], \
                               d_output[0:OUTPUT_SIZE], \
                               d_hidden[0:HIDDEN_SIZE], \
                               current_image[0:INPUT_SIZE], \
                               current_label[0:OUTPUT_SIZE])
    
    free(hidden);
    free(output);
    free(d_output);
    free(d_hidden);
    free(current_image);
    free(current_label);
}

void evaluate(NeuralNetwork* net, my_type** images, my_type** labels, int numImages) {
    //printf("Evaluating on %d images...\n", numImages);
    
    my_type* hidden = (my_type*)malloc(HIDDEN_SIZE * sizeof(my_type));
    my_type* output = (my_type*)malloc(OUTPUT_SIZE * sizeof(my_type));
    
    // Create buffer for current sample to avoid non-contiguous memory access
    my_type* current_image = (my_type*)malloc(INPUT_SIZE * sizeof(my_type));
    
    #pragma acc enter data create(hidden[0:HIDDEN_SIZE], output[0:OUTPUT_SIZE], current_image[0:INPUT_SIZE])

    int correct = 0;
    double start_time = omp_get_wtime();
    double total_loss = 0.0;

    for (int i = 0; i < numImages; i++) {
        // Copy current sample to contiguous memory
        for (int j = 0; j < INPUT_SIZE; j++) {
            current_image[j] = images[i][j];
        }
        
        // Update device with current sample
        #pragma acc update device(current_image[0:INPUT_SIZE])
        
        forward_pass(net, current_image, hidden, output);
        
        #pragma acc update host(output[0:OUTPUT_SIZE])
        
        // Calculate loss
        float sample_loss = 0.0f;
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            sample_loss -= labels[i][k] * logf(output[k] + 1e-10f);
        }
        total_loss += sample_loss;
        
        // Calculate accuracy
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }

    double eval_time = omp_get_wtime() - start_time;
    printf("Test Loss: %.4f - Accuracy: %.2f%% (Time: %.3fs)\n", 
          total_loss / numImages,
          (correct * 100.0) / numImages, 
          eval_time);
    
    #pragma acc exit data delete(hidden[0:HIDDEN_SIZE], output[0:OUTPUT_SIZE], current_image[0:INPUT_SIZE])
    free(hidden);
    free(output);
    free(current_image);
}

// Allocate matrix with correct parameters
my_type** allocateMatrix(int rows, int cols) {
    my_type** mat = (my_type**)malloc(rows * sizeof(my_type*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (my_type*)malloc(cols * sizeof(my_type));
    }
    return mat;
}

// Fixed: actually use the cols parameter
void freeMatrix(my_type** mat, int rows, int cols) {
    // cols parameter unused but kept for API consistency
    (void)cols; // Silence unused parameter warning
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

void freeNetwork(NeuralNetwork* net) {
    #pragma acc exit data delete(net->W1[0:HIDDEN_SIZE*INPUT_SIZE])
    #pragma acc exit data delete(net->W2[0:OUTPUT_SIZE*HIDDEN_SIZE])
    #pragma acc exit data delete(net->b1[0:HIDDEN_SIZE])
    #pragma acc exit data delete(net->b2[0:OUTPUT_SIZE])
    
    free(net->W1);
    free(net->W2);
    free(net->b1);
    free(net->b2);
    free(net);
}

my_type** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) { 
        perror("Opening image file"); 
        exit(1); 
    }
    
    // Skip file header
    fseek(file, 16, SEEK_SET);
    
    my_type** images = allocateMatrix(numImages, INPUT_SIZE);
    
    for (int i = 0; i < numImages; i++) {
        unsigned char* buffer = (unsigned char*)malloc(INPUT_SIZE);
        
        // Read an entire image at once for better I/O performance
        if (fread(buffer, sizeof(unsigned char), INPUT_SIZE, file) != INPUT_SIZE) {
            fprintf(stderr, "Error reading image %d\n", i);
            fclose(file);
            exit(EXIT_FAILURE);
        }
        
        // Normalize pixel values to [0,1]
        for (int j = 0; j < INPUT_SIZE; j++) {
            images[i][j] = buffer[j] / 255.0f;
        }
        
        free(buffer);
    }
    
    fclose(file);
    return images;
}

my_type** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) { 
        perror("Opening label file"); 
        exit(1); 
    }
    
    // Skip file header
    fseek(file, 8, SEEK_SET);
    
    // Read all labels at once for better I/O performance
    unsigned char* buffer = (unsigned char*)malloc(numLabels);
    size_t read_items = fread(buffer, sizeof(unsigned char), numLabels, file);
    if (read_items != (size_t)numLabels) {
        fprintf(stderr, "Error reading labels: expected %d, got %zu\n", 
                numLabels, read_items);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    
    my_type** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    
    for (int i = 0; i < numLabels; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == buffer[i]) ? 1.0f : 0.0f;
        }
    }
    
    free(buffer);
    fclose(file);
    return labels;
}

int main() {


    // Record total execution time
    double start_time = omp_get_wtime();
    
    // Load MNIST dataset
    printf("Loading MNIST dataset...\n");
    my_type** train_images = loadMNISTImages("../../data/train-images.idx3-ubyte", 60000);
    my_type** train_labels = loadMNISTLabels("../../data/train-labels.idx1-ubyte", 60000);
    my_type** test_images = loadMNISTImages("../../data/t10k-images.idx3-ubyte", 10000);
    my_type** test_labels = loadMNISTLabels("../../data/t10k-labels.idx1-ubyte", 10000);

    
    // Initialize OpenACC device
    acc_init(acc_device_default);
    
    // Create and train neural network
    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, 60000);
    evaluate(net, test_images, test_labels, 10000);
    
    // Free resources - pass correct column sizes
    freeNetwork(net);
    freeMatrix(train_images, 60000, INPUT_SIZE);
    freeMatrix(train_labels, 60000, OUTPUT_SIZE);
    freeMatrix(test_images, 10000, INPUT_SIZE);
    freeMatrix(test_labels, 10000, OUTPUT_SIZE);
    
    double total_time = omp_get_wtime() - start_time;
    printf("Total execution time: %.3fs\n", total_time);

    
    return 0;
}