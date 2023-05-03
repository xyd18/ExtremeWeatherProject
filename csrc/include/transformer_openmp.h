#ifndef TRANSFORMER_TMP_H_
#define TRANSFORMER_TMP_H_

#include <cmath>
#include <chrono>
#include "feedforward_openmp.h"
#include "multiheadattention_openmp.h"
#include "dropout.h"
#include "cube.h"
#include "layernorm_cube.h"


class TransformerEncoderLayer_openmp {
public:
    int num_workers;
    MultiHeadAttention_openmp multi_head_attention; // Multi-Head Attention sublayer
    LayerNorm_cube attention_norm;                // Layer normalization for attention sublayer
    FeedForwardLayer_openmp feedforward_layer;      // Feed-Forward sublayer
    LayerNorm_cube feedforward_norm;              // Layer normalization for feedforward sublayer

    // Constructor with specified input dimension, hidden dimension, number of heads, and output dimension
    TransformerEncoderLayer_openmp(int input_dim, int hidden_dim, int num_heads, int num_workers)
        : multi_head_attention(input_dim, input_dim / num_heads, num_heads, num_workers),
          feedforward_layer(input_dim, hidden_dim, num_workers),
          feedforward_norm(input_dim, 1e-6f),
          attention_norm(input_dim, 1e-6f) {}

    // Forward pass of the transformer encoder layer
    Cube forward(const Cube& input) {
        auto mhaStart = std::chrono::system_clock::now();
        // Pass input through the multi-head attention sublayer
        Cube attention_output = multi_head_attention.forward(input);
#ifdef DEBUG
        printf("TransformerEncoderLayerTMP_CUBE::forward attention_output shape: (batch_size=%d, seq_len=%d, d_model=%d)\n",attention_output.batch_size, attention_output.rows, attention_output.cols);
#endif
        auto mhaEnd = std::chrono::system_clock::now();
        std::chrono::duration<float> mha_forward_seconds = mhaEnd - mhaStart;
        printf("[TIME] multihead attention forward cost: %.6fs\n", mha_forward_seconds.count());

        // Add and normalize (residual connection + layer normalization)
        Cube ff_input = attention_norm.forward(input + attention_output);
        
        auto ffStart = std::chrono::system_clock::now(); // get the current time
        // Pass the result through the feedforward sublayer
        Cube ff_output = feedforward_layer.forward(ff_input);
        auto ffEnd = std::chrono::system_clock::now();
        std::chrono::duration<float> elapsed_seconds = ffEnd - ffStart;
        printf("[TIME] ff cost: %.6fs\n", elapsed_seconds.count());

        // Add and normalize (residual connection + layer normalization)
        Cube output = feedforward_norm.forward(ff_input + ff_output);

        auto ffnEnd = std::chrono::system_clock::now();
        std::chrono::duration<float> total_seconds = ffnEnd - mhaStart;
        printf("[TIME] total cost: %.6fs\n", total_seconds.count());
        return output;
    }

    // Backward pass of the transformer encoder layer
    Cube backward(const Cube& dO) {
        auto b_start = std::chrono::system_clock::now();
        Cube dFeedforward = feedforward_norm.backward(dO);

        auto ff_start = std::chrono::system_clock::now();
        Cube dFeedforwardAddNorm = feedforward_layer.backward(dFeedforward);
        auto ff_end = std::chrono::system_clock::now();
        std::chrono::duration<float> ff_cost = ff_end - ff_start;
        printf("[TIME] FeedForward backward cost: %.6fs\n", ff_cost.count());

        Cube dAttentionAddNorm = attention_norm.backward(dFeedforwardAddNorm);
        
        auto mhal_start = std::chrono::system_clock::now();
        Cube dAttention = multi_head_attention.backward(dAttentionAddNorm);
        auto mhal_end = std::chrono::system_clock::now();
        std::chrono::duration<float> mhal_cost = mhal_end - mhal_start;
        printf("[TIME] Multi-Head Attention backward cost: %.6fs\n", mhal_cost.count());

        std::chrono::duration<float> total_cost = mhal_end - b_start;
        printf("[TIME] Total backward cost: %.6fs\n", total_cost.count());

        return dAttention;
    }
};

#endif