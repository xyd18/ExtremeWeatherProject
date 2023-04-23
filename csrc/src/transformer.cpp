#include <random>
#include "../include/transformer.h"
#include "../include/common.h"

int main() {
    // Parameters for demonstration purposes
    int input_dim = 512;   // Dimension of input representation
    int hidden_dim = 2048; // Dimension of hidden representation
    int batch_size = 32;  // Number of input samples in the batch
    std::cout << "==================Transformer Encoder Layer==================" << std::endl;
    // Instantiate FeedForwardLayer with specified input, hidden, and output dimensions
    TransformerEncoderLayer transformer(input_dim, hidden_dim, 8);
    
    // Input matrix (batch size = batch_size, input dimension = input_dim)
    Matrix input(batch_size, input_dim);
#ifdef DEBUG
    printf("TransformerEncoderLayer input size: (batch_size=%d, d_model=%d)\n", input.rows, input.cols);
#endif

    // Forward pass through the feedforward layer
    Matrix output = transformer.forward(input);
#ifdef DEBUG
    printf("TransformerEncoderLayer Output size: (batch_size=%d, d_model=%d)\n", output.rows, output.cols);
#endif
    // random labels :)
    std::vector<int> labels(batch_size);
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, input_dim - 1); 
    for(int i = 0; i < batch_size; i++) {
        labels[i] = distribution(generator);
    }
    Matrix dO = common::softMaxCrossEntropyBackward(output, labels);

    // Backward pass
    Matrix dI = transformer.backward(dO);
    std::cout << "dI shape: (" << dI.rows << ", " << dI.cols << ")" << std::endl;

    return 0;
}