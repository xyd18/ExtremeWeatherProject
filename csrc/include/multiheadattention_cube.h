#ifndef MULTIHEADATTENTION_H_
#define MULTIHEADATTENTION_H_

#include "matrix.h"
#include "cube.h"
#include <vector>
#include <iostream>

class MultiHeadAttention_cube {
public:
    int num_heads;
    int d_k;
    int d_model;
    int pid;
    int nproc;
    int heads_per_p;
    bool hasWork; // record whether this process has work to do in MHAL
    std::vector<Matrix> W_q;
    std::vector<Matrix> W_k;
    std::vector<Matrix> W_v;
    Matrix W_o;

    MultiHeadAttention_cube(int d_model, int d_k, int num_heads, int nproc, int pid) 
        : d_model(d_model), num_heads(num_heads), d_k(d_k), W_o(d_k * std::max(num_heads / nproc, 1), d_model),
        pid(pid), nproc(nproc), heads_per_p(std::max(num_heads / nproc, 1)) {
        hasWork = pid * heads_per_p <= num_heads - heads_per_p;
        if(hasWork) {
            W_q.reserve(heads_per_p);
            W_k.reserve(heads_per_p);
            W_v.reserve(heads_per_p);
            for (int i = 0; i < heads_per_p; ++i) {
                W_q.emplace_back(d_model, d_k);
                W_k.emplace_back(d_model, d_k);
                W_v.emplace_back(d_model, d_k);
            }
            reset();
#ifdef DEBUG
            std::cout << "[MultiHeadAttention constructor] Worker " << pid << ", heads: " << heads_per_p << std::endl;
#endif
        }else{
#ifdef DEBUG
            std::cout << "[MultiHeadAttention constructor] Worker " << pid << " does not have work in MHAL" << std::endl;
#endif
        }
    }

    MultiHeadAttention_cube(const MultiHeadAttention_cube& other)
    : num_heads(other.num_heads), d_k(other.d_k), d_model(other.d_model), pid(other.pid), nproc(other.nproc),
      heads_per_p(other.heads_per_p), hasWork(other.hasWork), W_q(other.W_q), W_k(other.W_k), W_v(other.W_v), W_o(other.W_o) {}


    ~MultiHeadAttention_cube() {
#ifdef DEBUG
            std::cout << "[MultiHeadAttention destructor]" << std::endl;
#endif
    }

    void reset() {
        if(!hasWork) return;
        for (int i = 0; i < heads_per_p; ++i) {
            W_q[i].reset();
            W_k[i].reset();
            W_v[i].reset();
        }
        W_o.reset();
    }

    // Forward pass of the multi-head attention layer
    Cube forward(const Cube& X) {
        if(!hasWork) return Cube(0, 0, 0); // FIXME: this is errorneous if nproc > num_heads, need a sub group of the world
        Cube concat_heads(X.batch_size, X.rows, d_k * heads_per_p);
        for (int h = 0; h < heads_per_p; ++h) {
            Cube Q = X * W_q[h]; // (batch_size, seq_length, d_model) * (d_model, d_k) = (batch_size, seq_length, d_k)
            Cube K = X * W_k[h]; // n * d_k
            Cube V = X * W_v[h]; // n * d_v 

            Cube attention_scores = Q * K.transpose(); // (batch_size, seq_length, d_k) * (batch_size, d_k, seq_length) = (batch_size, seq_length, seq_length)
            // std::cout << "attention_scores shape: " << attention_scores.rows << " " << attention_scores.cols << std::endl;

            // scaled + softmax
            for (int b = 0; b < attention_scores.batch_size; ++b) {
                for (int i = 0; i < attention_scores.rows; ++i) {
                    float sum = 0.0f;
                    for (int j = 0; j < attention_scores.cols; ++j) {
                        attention_scores(b, i, j) /= std::sqrt(d_k);
                        attention_scores(b, i, j) = std::exp(attention_scores(b, i, j));
                        sum += attention_scores(b, i, j);
                    }
                    for (int j = 0; j < attention_scores.cols; ++j) {
                        attention_scores(b, i, j) /= sum;
                    }
                }
            }
            

            Cube head = attention_scores * V; // (batch_size, seq_length, seq_length) * (batch_size, seq_length, d_v) = (batch_size, seq_length, d_v)

            // concatenate head
            int col_index = h * d_k;
            for (int b = 0; b < X.batch_size; ++b) {
                for (int i = 0; i < X.rows; ++i) { // concate on last dimension
                    for (int j = 0; j < d_k; ++j) {
                        concat_heads(b, i, col_index + j) = head(b, i, j);
                    }
                }
            }
            
        }

        // Project concatenated heads to output
        Cube output = concat_heads * W_o; // (batch_size, seq_length, d_k * heads_per_p) * (d_k * heads_per_p, d_model);
        return output; // (batch_size, seq_length, d_model)
    }

    /** Backward pass of the multi-head attention layer
     * dQ = dY * Wq, dK = dY * Wk, dV = dY * Wv
     * dA_q = softmax_prime(A) * (dQ * K.T), dA_k = softmax_prime(A) * (Q.T * dK), dA_v = softmax_prime(A) * (Q.T * dV)
     * dQ_in = dA_q * K + alpha * dQ, dK_in = dA_k * Q + alpha * dK, dV_in = dA_v * Q + alpha * dV
     * dWq = X_in.T * dQ_in, dWk = X_in.T * dK_in, dWv = X_in.T * dV_in
     */
    Matrix backward(Matrix grad) {
        // std::cout << "MultiHeadAttention backward" << std::endl;
        // std::cout << "grad shape: " << grad.rows << " " << grad.cols << std::endl;
        // for(int h = 0;h < num_heads;h++) {
        //     Matrix grad_slice(grad.rows, d_k);
        //     for (int i = 0; i < grad.rows; ++i) {
        //         for (int j = 0; j < d_k; ++j) {
        //             grad_slice(i, j) = grad(i, h * d_k + j);
        //         }
        //     }
        //     std::cout << "grad_slice shape: " << grad_slice.rows << " " << grad_slice.cols << std::endl;

        //     Matrix dQ = grad_slice * W_q[h].transponse();
        //     Matrix dK = grad_slice * W_k[h].transponse();
        //     Matrix dV = grad_slice * W_v[h].transponse();
        //     /* FIXME: incomplete implementation */
        // }
        // return Matrix(0, 0); // FIXME: for now return nothing
    }
};

#endif