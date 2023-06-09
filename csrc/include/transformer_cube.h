#ifndef TRANSFORMER_H_
#define TRANSFORMER_H_

#include <cmath>
#include <chrono>
#include "model.h"
#include "feedforward_cube.h"
#include "matrix.h"
#include "layernorm_cube.h"
#include "multiheadattention_cube.h"

class TransformerEncoderLayer_cube : public Model{
public:
    MultiHeadAttention_cube multi_head_attention; // Multi-Head Attention sublayer
    LayerNorm_cube attention_norm;                // Layer normalization for attention sublayer
    FeedForwardLayer_Cube feedforward_layer;      // Feed-Forward sublayer
    LayerNorm_cube feedforward_norm;              // Layer normalization for feedforward sublayer

    // Constructor with specified input dimension, hidden dimension, number of heads, and output dimension
    TransformerEncoderLayer_cube(int input_dim, int hidden_dim, int num_heads)
        : multi_head_attention(input_dim, input_dim / num_heads, num_heads, 1, 0),
          feedforward_layer(input_dim, hidden_dim),
          feedforward_norm(input_dim, 1e-6f),
          attention_norm(input_dim, 1e-6f) {
#ifdef DEBUG
        std::cout << "[TransformerEncoderLayer_cube constructor] " << this << std::endl;
#endif
          }

    TransformerEncoderLayer_cube(const TransformerEncoderLayer_cube& other)
        : multi_head_attention(other.multi_head_attention),
          feedforward_layer(other.feedforward_layer),
          feedforward_norm(other.feedforward_norm),
          attention_norm(other.attention_norm)
    {
#ifdef DEBUG
        std::cout << "[TransformerEncoderLayer_cube copy constructor] " << this << std::endl;
#endif
    }

    ~TransformerEncoderLayer_cube() {
#ifdef DEBUG
        std::cout << "[TransformerEncoderLayer_cube destructor] " << this <<std::endl;
#endif
    }

    // Forward pass of the transformer encoder layer
    Cube forward(const Cube& input) override {
        auto mhaStart = std::chrono::system_clock::now();

        // Pass input through the multi-head attention sublayer
        Cube attention_output = multi_head_attention.forward(input);
        auto mhaEnd = std::chrono::system_clock::now();
        std::chrono::duration<float> mha_forward_seconds = mhaEnd - mhaStart;
        printf("MHAL forward cost:\t%.6fs\n", mha_forward_seconds.count());

        // Add and normalize (residual connection + layer normalization)
        Cube attention_add_norm = attention_norm.forward(input + attention_output);

        auto ffStart = std::chrono::system_clock::now(); // get the current time
        // Pass the result through the feedforward sublayer
        Cube feedforward_output = feedforward_layer.forward(attention_add_norm);
        auto ffEnd = std::chrono::system_clock::now();
        std::chrono::duration<float> ff_seconds = ffEnd - ffStart;
        printf("FF forward cost:\t%.6fs\n", ff_seconds.count());

        // Add and normalize (residual connection + layer normalization)
        Cube output = feedforward_norm.forward(attention_add_norm + feedforward_output);

        auto ffnEnd = std::chrono::system_clock::now();
        std::chrono::duration<float> total_seconds = ffnEnd - mhaStart;
        printf("Total forward cost:\t%.6fs\n", total_seconds.count());

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
