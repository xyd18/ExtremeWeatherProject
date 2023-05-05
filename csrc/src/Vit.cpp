#include <mpi.h>

#include "../include/Vit.h"
#include "../include/utils.h"


int main(int argc, char** argv) {
    StartupOptions options = parseOptions(argc, argv);
    printf("Using TPM=%s PIP=%s\n", options.usingTMP ? "true" : "false", options.usingPip ? "true" : "false");
    printf("numMicroBatch=%d numBatchSize=%d\n", options.numMicroBatch, options.numBatchSize);

    if (options.usingTMP || options.usingPip) {
        // Initialize MPI
        MPI_Init(&argc, &argv);
    }

    // Instantiate the ViT model
    int num_hidden_layers = 24;  // Number of Transformer blocks
    int num_classes = 1000; // Number of classes for classification
    int patch_size = 16;  // Patch size for tokenization
    int num_channels = 3;  // Number of num_channels in the input image (3 for RGB)
    int hidden_dim = 768;  // Dimensionality of the hidden states in the Transformer


    // Parameters for demonstration purposes
    int input_dim = 512;   // Dimension of input representation
    int output_dim = 32;   // Dimension of output representation
    int batch_size = options.numBatchSize;   // Number of input samples in the batch
    int seq_length = 100;

    std::cout << "==================VisionTransformer==================" << std::endl;
    VisionTransformer vit(options, num_hidden_layers, 12, num_classes, patch_size, num_channels, hidden_dim);

    Cube input(batch_size, seq_length, hidden_dim);
    if (options.inputFile == "") {
        input.reset();
    } else {
        input.load(options.inputFile);
    }

#ifdef DEBUG
    printf("VisionTransformer input size: (batch_size=%d, seq_len=%d, d_model=%d)\n", input.batch_size, input.rows, input.cols);
    fflush(stdout);;
#endif
    auto forward_start = std::chrono::system_clock::now();
    Cube output = vit.forward(input);
    auto forward_end = std::chrono::system_clock::now();
    printf("VIT forward cost:\t%.6fs\n", std::chrono::duration<float>(forward_end - forward_start).count());
    output.save(options.outputFile);

#ifdef DEBUG
    printf("VisionTransformer Output size: (batch_size=%d, seq_len=%d, d_model=%d)\n", output.batch_size, output.rows, output.cols);
    fflush(stdout);;
#endif

    if (options.usingTMP || options.usingPip) {
        // Finalize MPI
        MPI_Finalize();
    }

    return 0;
}