#include "../include/transformer_cube.h"

int main() {
    // Parameters for demonstration purposes
    int input_dim = 512;   // Dimension of input representation
    int hidden_dim = 2048; // Dimension of hidden representation
    int output_dim = 32;   // Dimension of output representation
    int batch_size = 32;   // Number of input samples in the batch
    int seq_length = 100;

    std::cout << "==================Transformer Encoder Layer==================" << std::endl;
    TransformerEncoderLayer_cube transformer(input_dim, hidden_dim, 8);

    Cube input(batch_size, seq_length, input_dim); 

#ifdef DEBUG
    printf("TransformerEncoderLayer input size: (batch_size=%d, seq_len=%d, d_model=%d)\n", input.batch_size, input.rows, input.cols);
#endif

    // forward pass
    Cube output = transformer.forward(input);

#ifdef DEBUG
    printf("TransformerEncoderLayer Output size: (batch_size=%d, seq_len=%d, d_model=%d)\n", output.batch_size, output.rows, output.cols);
#endif
    
    return 0;
}