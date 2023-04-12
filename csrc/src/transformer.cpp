#include "../include/transformer.h"

int main() {
    // Parameters for demonstration purposes
    int input_dim = 512;   // Dimension of input representation
    int hidden_dim = 2048; // Dimension of hidden representation
    int output_dim = 32;  // Dimension of output representation
    int batch_size = 10;  // Number of input samples in the batch

    // Instantiate FeedForwardLayer with specified input, hidden, and output dimensions
    TransformerEncoderLayer transformer(input_dim, hidden_dim, 8);

    // Input matrix (batch size = batch_size, input dimension = input_dim)
    Matrix input(batch_size, input_dim);

    // Forward pass through the feedforward layer
    Matrix output = transformer.forward(input);

    // Output shape should be (batch_size, output_dim)
    std::cout << "Output shape: (" << output.rows << ", " << output.cols << ")" << std::endl;

    return 0;
}