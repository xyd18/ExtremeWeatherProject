#ifndef TRANSFORMER_TMP_H_
#define TRANSFORMER_TMP_H_

#include <cmath>
#include <chrono>
#include "model.h"
#include "mpi.h"
#include "feedforward_tmp_cube.h"
#include "dropout.h"
#include "cube.h"
#include "layernorm_cube.h"
#include "multiheadattention_cube.h"

class TransformerEncoderLayerTMP_CUBE : public Model {
public:
    int pid;
    int nproc;
    MultiHeadAttention_cube multi_head_attention; // Multi-Head Attention sublayer
    LayerNorm_cube attention_norm;                // Layer normalization for attention sublayer
    FeedForwardLayerTMP_cube feedforward_layer;      // Feed-Forward sublayer
    LayerNorm_cube feedforward_norm;              // Layer normalization for feedforward sublayer

    // Constructor with specified input dimension, hidden dimension, number of heads, and output dimension
    TransformerEncoderLayerTMP_CUBE(int input_dim, int hidden_dim, int num_heads, int pid, int nproc)
        : multi_head_attention(input_dim, input_dim / num_heads, num_heads, nproc, pid),
          pid(pid), nproc(nproc),
          feedforward_layer(input_dim, hidden_dim, pid, nproc),
          feedforward_norm(input_dim, 1e-6f),
          attention_norm(input_dim, 1e-6f) {}

    // Forward pass of the transformer encoder layer
    Cube forward(const Cube& input) override {
        auto mhaStart = std::chrono::system_clock::now();
        // Pass input through the multi-head attention sublayer
        Cube attention_output = multi_head_attention.forward(input);
#ifdef DEBUG
        printf("TransformerEncoderLayerTMP_CUBE::forward attention_output shape: (batch_size=%d, seq_len=%d, d_model=%d)\n",attention_output.batch_size, attention_output.rows, attention_output.cols);
#endif
        Cube attention_output_global = Cube(attention_output.batch_size, attention_output.rows, attention_output.cols);
        MPI_Allreduce(attention_output.data, attention_output_global.data, attention_output.batch_size * attention_output.rows * attention_output.cols, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        auto mhaEnd = std::chrono::system_clock::now();
        std::chrono::duration<float> mha_forward_seconds = mhaEnd - mhaStart;
        printf("[Worker %d] multihead attention forward cost: %.6fs\n", pid, mha_forward_seconds.count());

        // Add and normalize (residual connection + layer normalization)
        // DESIGN: accroding megatron LM, LN, Residuals computation are duplicated and optimzied inidividually in each process, instead of one process + broadcast
        Cube ff_input = attention_norm.forward(input + attention_output_global);
        
        auto ffStart = std::chrono::system_clock::now(); // get the current time
        // Pass the result through the feedforward sublayer
        Cube ff_output = feedforward_layer.forward(ff_input);
        // gather output from all processes
        Cube ff_output_reduce(ff_output.batch_size, ff_output.rows, ff_output.cols);
        MPI_Allreduce(ff_output.data, ff_output_reduce.data, ff_output.batch_size * ff_output.rows * ff_output.cols, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        auto ffEnd = std::chrono::system_clock::now();
        std::chrono::duration<float> elapsed_seconds = ffEnd - ffStart;
        printf("[Worker %d] ff cost: %.6fs\n", pid, elapsed_seconds.count());

        // Add and normalize (residual connection + layer normalization)
        Cube output = feedforward_norm.forward(ff_input + ff_output_reduce);

        auto ffnEnd = std::chrono::system_clock::now();

        std::chrono::duration<float> total_seconds = ffnEnd - mhaStart;
        printf("[Worker %d] total forward cost: %.6fs\n", pid, total_seconds.count());

        return output;
    }

    // Backward pass of the transformer encoder layer
    Cube backward(const Cube& dO) {
        // FIXME: parallel, boy
        auto ff_start = std::chrono::system_clock::now();
        Cube d_ffn = feedforward_norm.backward(dO);
        
        Cube d_ff = feedforward_layer.backward(d_ffn);
        Cube d_ff_global(d_ff.batch_size, d_ff.rows, d_ff.cols);
        MPI_Allreduce(d_ff.data, d_ff_global.data, d_ff.batch_size * d_ff.rows * d_ff.cols, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        auto ff_end = std::chrono::system_clock::now();
        std::chrono::duration<float> ff_backward_cost = ff_end - ff_start;
        printf("[Worker %d] FF backward cost: %.6fs\n", pid, ff_backward_cost.count());

        Cube d_mha_n = attention_norm.backward(d_ff_global);
        
        auto mhab_start = std::chrono::system_clock::now();
        Cube d_mha = multi_head_attention.backward(d_mha_n);
        Cube d_mha_global(d_mha.batch_size, d_mha.rows, d_mha.cols);
        MPI_Allreduce(d_mha.data, d_mha_global.data, d_mha.batch_size * d_mha.rows * d_mha.cols, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        auto mhab_end = std::chrono::system_clock::now();
        std::chrono::duration<float> mha_backward_cost = mhab_end - mhab_start;
        printf("[Worker %d] MHAL backward cost: %.6fs\n", pid, mha_backward_cost.count());

        std::chrono::duration<float> b_total_cost = mhab_end - ff_start;
        printf("[Worker %d] Total backward cost: %.6fs\n", pid, b_total_cost.count());

        return d_mha_global;
    }
};

#endif