#include "../include/transformer_cube.h"

int main() {
    // Parameters for demonstration purposes
    int input_dim = 512;   // Dimension of input representation
    int hidden_dim = 2048; // Dimension of hidden representation
    int output_dim = 32;  // Dimension of output representation
    int batch_size = 32;  // Number of input samples in the batch
    int seq_length = 100;
    std::cout << "==================Transformer Encoder Layer==================" << std::endl;
    // Instantiate FeedForwardLayer with specified input, hidden, and output dimensions
    TransformerEncoderLayer_cube transformer(input_dim, hidden_dim, 8);
    std::cout << "Transformer Encoder Layer initialized" << std::endl;
    
    // Input matrix (batch size = batch_size, input dimension = input_dim)
    Cube input(batch_size, seq_length, input_dim);
    std::cout << "Input shape: (" << input.batch_size << ", " << input.rows << ", " << input.cols << ")" << std::endl;

    // Forward pass through the feedforward layer
    Cube output = transformer.forward(input);

    // Output shape should be (batch_size, output_dim)
    std::cout << "Output shape: (" << output.batch_size << ", " << output.rows << ", " << output.cols << ")" << std::endl;

    return 0;
}