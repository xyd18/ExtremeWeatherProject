#ifndef MULTIHEADATTENTION_H_
#define MULTIHEADATTENTION_H_

#include "matrix.h"

class MultiHeadAttention {
public:
    int num_heads;
    int d_k;
    Matrix* W_q;
    Matrix* W_k;
    Matrix* W_v;
    Matrix W_o;

    MultiHeadAttention(int d_model, int d_k, int num_heads) : num_heads(num_heads), d_k(d_k), W_o(num_heads * d_k, d_model) {
        W_q = new Matrix[num_heads];
        W_k = new Matrix[num_heads];
        W_v = new Matrix[num_heads];
        for (int i = 0; i < num_heads; ++i) {
            W_q[i] = Matrix(d_model, d_k);
            W_k[i] = Matrix(d_model, d_k);
            W_v[i] = Matrix(d_model, d_k);
        }
    }

    ~MultiHeadAttention() {
        delete[] W_q;
        delete[] W_k;
        delete[] W_v;
    }

    // Forward pass of the multi-head attention layer
    Matrix forward(const Matrix& X) {
        Matrix concat_heads(X.rows, num_heads * d_k);
        for (int h = 0; h < num_heads; ++h) {
            Matrix Q = X * W_q[h];
            Matrix K = X * W_k[h];
            Matrix V = X * W_v[h];

            Matrix attention_scores = Q * K;
            for (int i = 0; i < attention_scores.rows; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < attention_scores.cols; ++j) {
                    attention_scores(i, j) /= std::sqrt(d_k);
                    attention_scores(i, j) = std::exp(attention_scores(i, j));
                    sum += attention_scores(i, j);
                }
                for (int j = 0; j < attention_scores.cols; ++j) {
                    attention_scores(i, j) /= sum;
                }
            }

            Matrix head = attention_scores * V;
            int col_index = h * d_k;
            for (int i = 0; i < X.rows; ++i) {
                for (int j = 0; j < d_k; ++j) {
                    concat_heads(i, col_index + j) = head(i, j);
                }
            }
        }

        // Project concatenated heads to output
        Matrix output = concat_heads * W_o;
        return output;
    }
};

#endif