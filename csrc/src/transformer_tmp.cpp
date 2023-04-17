#include <mpi.h>
#include <assert.h>
#include "../include/transformer_tmp.h"

int main(int argc, char *argv[]) {
    int pid;
    int nproc;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Get process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    // Get total number of processes specificed at start of run
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    // Parameters for demonstration purposes
    int input_dim = 512;   // Dimension of input representation
    int hidden_dim = 2048; // Dimension of hidden representation
    int output_dim = 32;  // Dimension of output representation
    int batch_size = 10;  // Number of input samples in the batch
    int num_heads = 8;    // Number of heads in the multi-head attention sublayer
    assert (hidden_dim % nproc == 0);
    assert (nproc > num_heads || num_heads % nproc == 0);
    std::cout << "==================Transformer Encoder Layer==================" << std::endl;
    // Instantiate FeedForwardLayer with specified input, hidden, and output dimensions
    TransformerEncoderLayerTMP transformer(input_dim, hidden_dim, num_heads, pid, nproc);
    std::cout << "Transformer Encoder Layer initialized" << std::endl;

    // Input matrix (batch size = batch_size, input dimension = input_dim)
    Matrix input(batch_size, input_dim);
    std::cout << "Input shape: (" << input.rows << ", " << input.cols << ")" << std::endl;

    // Forward pass through the feedforward layer
    Matrix output = transformer.forward(input);

    // Output shape should be (batch_size, output_dim)
    std::cout << "Output shape: (" << output.rows << ", " << output.cols << ")" << std::endl;

    // Finalize MPI
    MPI_Finalize();
}