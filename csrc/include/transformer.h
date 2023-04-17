#ifndef TRANSFORMER_H_
#define TRANSFORMER_H_

#include <cmath>
#include <chrono>
#include "feedforward.h"
#include "dropout.h"
#include "matrix.h"
#include "layernorm.h"
#include "multiheadattention.h"

class TransformerEncoderLayer {
public:
    MultiHeadAttention multi_head_attention; // Multi-Head Attention sublayer
    LayerNorm attention_norm;                // Layer normalization for attention sublayer
    FeedForwardLayer feedforward_layer;      // Feed-Forward sublayer
    LayerNorm feedforward_norm;              // Layer normalization for feedforward sublayer

    // Constructor with specified input dimension, hidden dimension, number of heads, and output dimension
    TransformerEncoderLayer(int input_dim, int hidden_dim, int num_heads)
        : multi_head_attention(input_dim, input_dim / num_heads, num_heads, 1, 0),
          feedforward_layer(input_dim, hidden_dim),
          feedforward_norm(hidden_dim, 1e-6f),
          attention_norm(hidden_dim, 1e-6f) {}

    // Forward pass of the transformer encoder layer
    Matrix forward(const Matrix& input) {
        // Pass input through the multi-head attention sublayer
        Matrix attention_output = multi_head_attention.forward(input);
        std::cout << "attention_output shape: " << attention_output.rows << " " << attention_output.cols << std::endl;

        // Add and normalize (residual connection + layer normalization)
        Matrix attention_add_norm = attention_norm.forward(input + attention_output);

        auto ffStart = std::chrono::system_clock::now(); // get the current time
        // Pass the result through the feedforward sublayer
        Matrix feedforward_output = feedforward_layer.forward(attention_add_norm);
        auto ffEnd = std::chrono::system_clock::now();
        std::chrono::duration<float> elapsed_seconds = ffEnd - ffStart;
        printf("ff cost: %.6fs\n", elapsed_seconds.count());

        // Add and normalize (residual connection + layer normalization)
        Matrix output = feedforward_norm.forward(attention_add_norm + feedforward_output);

        return output;
    }
};

#endif