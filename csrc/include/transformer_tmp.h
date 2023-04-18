#ifndef TRANSFORMER_TMP_H_
#define TRANSFORMER_TMP_H_

#include <cmath>
#include <chrono>
#include "mpi.h"
#include "feedforward_tmp.h"
#include "dropout.h"
#include "matrix.h"
#include "layernorm.h"
#include "multiheadattention.h"

class TransformerEncoderLayerTMP {
public:
    int pid;
    int nproc;
    MultiHeadAttention multi_head_attention; // Multi-Head Attention sublayer
    LayerNorm attention_norm;                // Layer normalization for attention sublayer
    FeedForwardLayerTMP feedforward_layer;      // Feed-Forward sublayer
    LayerNorm feedforward_norm;              // Layer normalization for feedforward sublayer

    // Constructor with specified input dimension, hidden dimension, number of heads, and output dimension
    TransformerEncoderLayerTMP(int input_dim, int hidden_dim, int num_heads, int pid, int nproc)
        : multi_head_attention(input_dim, input_dim / num_heads, num_heads, nproc, pid),
          pid(pid), nproc(nproc),
          feedforward_layer(input_dim, hidden_dim, pid, nproc),
          feedforward_norm(hidden_dim, 1e-6f),
          attention_norm(hidden_dim, 1e-6f) {}

    // Forward pass of the transformer encoder layer
    Matrix forward(const Matrix& input) {
        // Pass input through the multi-head attention sublayer
        Matrix attention_output = multi_head_attention.forward(input);
        std::cout << "attention_output shape: " << attention_output.rows << " " << attention_output.cols << std::endl;
        Matrix attention_output_global = Matrix(attention_output.rows, attention_output.cols);
        MPI_Allreduce(attention_output.data, attention_output_global.data, attention_output.rows * attention_output.cols, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        // Add and normalize (residual connection + layer normalization)
        // DESIGN: accroding megatron LM, LN, Residuals computation are duplicated and optimzied inidividually in each process, instead of one process + broadcast
        Matrix ff_input = attention_norm.forward(input + attention_output_global);
        int ff_input_size[2] = { ff_input.rows, ff_input.cols };
        
        auto ffStart = std::chrono::system_clock::now(); // get the current time

        // broadcast input to all processes
        // MPI_Bcast(&ff_input_size[0], 2, MPI_INT, 0, MPI_COMM_WORLD);
        // if(pid != 0) ff_input = Matrix(ff_input_size[0], ff_input_size[1]);
        // MPI_Bcast(ff_input.data, ff_input_size[0] * ff_input_size[1], MPI_FLOAT, 0, MPI_COMM_WORLD);

        // Pass the result through the feedforward sublayer
        Matrix ff_output = feedforward_layer.forward(ff_input);
        // gather output from all processes
        Matrix ff_output_reduce(ff_output.rows, ff_output.cols);
        MPI_Allreduce(ff_output.data, ff_output_reduce.data, ff_output.rows * ff_output.cols, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        
        auto ffEnd = std::chrono::system_clock::now();
        std::chrono::duration<float> elapsed_seconds = ffEnd - ffStart;
        printf("[Worker %d] ff cost: %.6fs\n", pid, elapsed_seconds.count());
        // std::cout << "[Worker " << pid << "] ff cost: " << elapsed_seconds.count << std::endl;

        // Add and normalize (residual connection + layer normalization)
        Matrix output = feedforward_norm.forward(ff_input + ff_output_reduce);

        return output;
    }

    // Backward pass of the transformer encoder layer
    Matrix backward(const Matrix& dO) {
        Matrix dFeedforward = feedforward_norm.backward(dO);

        Matrix dFeedforwardAddNorm = feedforward_layer.backward(dFeedforward);

        Matrix dAttentionAddNorm = attention_norm.backward(dFeedforwardAddNorm);
        
        Matrix dAttention = multi_head_attention.backward(dAttentionAddNorm);

        return dAttention;
    }
};

#endif