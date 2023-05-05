#ifndef MULTIHEADATTENTION_OPENMP_H_
#define MULTIHEADATTENTION_OPENMP_H_

#include "matrix.h"
#include "cube.h"
#include <vector>
#include <iostream>
#include <omp.h>

class MultiHeadAttention_openmp {
public:
    int num_heads;
    int d_k;
    int d_model;
    int num_workers;
    std::vector<Matrix> W_q;
    std::vector<Matrix> W_k;
    std::vector<Matrix> W_v;
    std::vector<Matrix> W_o;
    // Matrix W_o;

    std::vector<Cube> Q_cache;
    std::vector<Cube> K_cache;
    std::vector<Cube> V_cache;
    std::vector<Cube> QK_softmax_cache;
    Cube X_cache;
    std::vector<Cube> before_projection_cache;

    MultiHeadAttention_openmp(int d_model, int d_k, int num_heads, int num_workers) 
        : d_model(d_model), num_heads(num_heads), d_k(d_k), num_workers(num_workers) {
        // W_o(d_model, d_model) {
        W_q.reserve(num_heads);
        W_k.reserve(num_heads);
        W_v.reserve(num_heads);
        W_o.reserve(num_heads);
        for (int i = 0; i < num_heads; ++i) {
            W_q.emplace_back(d_model, d_k);
            W_k.emplace_back(d_model, d_k);
            W_v.emplace_back(d_model, d_k);
            W_o.emplace_back(d_k, d_model);
        }
        Q_cache.reserve(num_heads);
        K_cache.reserve(num_heads);
        V_cache.reserve(num_heads);
        QK_softmax_cache.reserve(num_heads);
        before_projection_cache.reserve(num_heads);
        reset();
#ifdef DEBUG
        std::cout << "[MultiHeadAttention OpenMP constructor] Initialization Completed." << std::endl;
#endif     
    }

    MultiHeadAttention_openmp(const MultiHeadAttention_openmp& other)
    : num_heads(other.num_heads), d_k(other.d_k), d_model(other.d_model), num_workers(other.num_workers), 
    W_q(other.W_q), W_k(other.W_k), W_v(other.W_v), W_o(other.W_o) {}


    ~MultiHeadAttention_openmp() {
#ifdef DEBUG
            std::cout << "[MultiHeadAttention destructor]" << std::endl;
#endif
    }

    void reset() {
        for (int i = 0; i < num_heads; ++i) {
            W_q[i].reset();
            W_k[i].reset();
            W_v[i].reset();
            W_o[i].reset();
        }
        // W_o.reset();
    }

    // Forward pass of the multi-head attention layer
    Cube forward(const Cube& X) {
        X_cache = X;
        std::vector<Cube> heads_list(num_heads);
        Cube concat_heads(X.batch_size, X.rows, d_model);
        auto start_individual = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for num_threads(num_workers) // schedule(dynamic, num_heads)
        for (int h = 0; h < num_heads; ++h) {
            Cube Q = X * W_q[h]; // (batch_size, seq_length, d_model) * (d_model, d_k) = (batch_size, seq_length, d_k)
            Cube K = X * W_k[h];
            Cube V = X * W_v[h]; 
            Q_cache[h] = Q;
            K_cache[h] = K;
            V_cache[h] = V;

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
            
            QK_softmax_cache[h] = attention_scores;
            Cube head = attention_scores * V; // (batch_size, seq_length, seq_length) * (batch_size, seq_length, d_v) = (batch_size, seq_length, d_v)

            // int col_index = h * d_k;
            // for (int b = 0; b < X.batch_size; ++b) {
            //     for (int i = 0; i < X.rows; ++i) { // concate on last dimension
            //         for (int j = 0; j < d_k; ++j) {
            //             concat_heads(b, i, col_index + j) = head(b, i, j);
            //         }
            //     }
            // }
            before_projection_cache[h] = head;
            heads_list[h] = head * W_o[h];
        }
        auto end_individual = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_individual = end_individual - start_individual;
        std::cout << "MHAL Forward Individual time: " << elapsed_individual.count() << std::endl;

        // concatenate head
        for(int h = 0; h < num_workers; ++h) {
            concat_heads = concat_heads + heads_list[h];
        }
        
        // after_concat_cache = concat_heads;
        // Project concatenated heads to output
        // Cube output = concat_heads * W_o; // (batch_size, seq_length, d_k * heads_per_p) * (d_k * heads_per_p, d_model);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - end_individual;
        std::cout << "MHAL Forward Single time: " << elapsed.count() << std::endl;
        return concat_heads; // (batch_size, seq_length, d_model)
    }

    /** Backward pass of the multi-head attention layer
     * dQ = dY * Wq, dK = dY * Wk, dV = dY * Wv
     * dA_q = softmax_prime(A) * (dQ * K.T), dA_k = softmax_prime(A) * (Q.T * dK), dA_v = softmax_prime(A) * (Q.T * dV)
     * dQ_in = dA_q * K + alpha * dQ, dK_in = dA_k * Q + alpha * dK, dV_in = dA_v * Q + alpha * dV
     * dWq = X_in.T * dQ_in, dWk = X_in.T * dK_in, dWv = X_in.T * dV_in
     */
    Cube backward(const Cube& grad) {
        std::cout << "===================MHAL Backward===================" << std::endl;
        printf("grad input(%d, %d, %d)\n", grad.batch_size, grad.rows, grad.cols);
        // sleep(1);
        // return grad;

        // Compute gradient w.r.t. output H
        int B = grad.batch_size;
        int N = grad.rows;
        int D = grad.cols;
        // Cube dH = grad * W_o.transpose(); // (B, N, h * d_v)
        // printf("dH(%d, %d, %d)\n", dH.batch_size, dH.rows, dH.cols);
        Cube dX_concate(B, N, D);
        // std::vector<Cube> dX_list(num_heads);
        auto start_individual = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for num_threads(num_workers)
        for(int h = 0; h < num_heads;h++) {
            // Cube dH_slice(B, N, d_k);
            // for(int b = 0;b < B;b++) {
            //     for(int i = 0;i < N;i++) {
            //         for(int j = 0;j < d_k;j++) {
            //             dH_slice(b, i, j) = dH(b, i, h * d_k + j);
            //         }
            //     }
            // }
            Cube dH_slice = grad * W_o[h].transpose();
            printf("dH_slice(%d, %d, %d)\n", dH_slice.batch_size, dH_slice.rows, dH_slice.cols);

            // Compute gradient w.r.t. concatenated Q,K,V
            Cube dQ = dH_slice * W_q[h].transpose(); // (B , N, d_model) 
            Cube dK = dH_slice * W_k[h].transpose();
            Cube dV = dH_slice * W_v[h].transpose();
            printf("dQ(%d, %d, %d)\n", dQ.batch_size, dQ.rows, dQ.cols);
            printf("dK(%d, %d, %d)\n", dK.batch_size, dK.rows, dK.cols);
            printf("dV(%d, %d, %d)\n", dV.batch_size, dV.rows, dV.cols);

            // Compute gradient w.r.t. attention scores A
            // Matrix dA = dQ * K_cache[i].transpose();
            // float seq_length = Q_cache[i].rows;
            // for(int j = 0;j < dA.cols * dA.rows;j++) {
            //     dA.data[j] /= seq_length;
            // }
            // Cube softmax_grad = common::softmax_backwar_cube(QK_softmax_cache[h]); // (B, N, N)
            // printf("softmax_grad(%d, %d, %d)\n", softmax_grad.batch_size, softmax_grad.rows, softmax_grad.cols);
            // dA = dA * softmax_grad;
            // Compute gradient w.r.t. Q,K,V weight matrices
            Cube dW_q = X_cache.transpose() * dQ; // B, d_model, d_model
            Cube dW_k = X_cache.transpose() * dK;
            Cube dW_v = X_cache.transpose() * dV;
            printf("dW_q(%d, %d, %d)\n", dW_q.batch_size, dW_q.rows, dW_q.cols);
            printf("dW_k(%d, %d, %d)\n", dW_k.batch_size, dW_k.rows, dW_k.cols);
            printf("dW_v(%d, %d, %d)\n", dW_v.batch_size, dW_v.rows, dW_v.cols);

            // Compute gradient w.r.t. input X
            Cube dX = dQ * W_q[h] + dK * W_k[h] + dV * W_v[h];
            printf("dX(%d, %d, %d)\n", dX.batch_size, dX.rows, dX.cols);
            // dX_list[h] = dX;
            // Update weight gradients FIXME: need learning rate?
            // FIXME: this just take advantage of d_model > d_k
            for(int i = 0;i < d_model;i++) {
                for(int j = 0;j < d_k;j++) {
                    float temp_q = W_q[h](i, j);
                    float temp_k = W_k[h](i, j);
                    float temp_v = W_v[h](i, j);
                    for(int b = 0;b < B;b++) {
                        temp_q += dW_q(b, i, j);
                        temp_k += dW_k(b, i, j);
                        temp_v += dW_v(b, i, j);
                    }
                    W_q[h](i, j) = temp_q;
                    W_k[h](i, j) = temp_k;
                    W_v[h](i, j) = temp_v;
                }
            }

            // update input gradients
            // Matrix dX_slice = dA * K_cache[i];
            // for(int j = 0;j < dX_slice.rows;j++) {
            //     for(int k = 0;k < dX_slice.cols;k++) {
            //         dX_concate(j, i * d_k + k) = dX_slice(j, k);
            //     }
            // }

            for(int b = 0;b < B;b++) {
                for(int i = 0;i < N;i++) {
                    for(int j = 0;j < d_k;j++) {
                        dX_concate(b, i, h * d_k + j) = dX(b, i, j);
                    }
                }
            }

            // update W_o
            Cube dW_o = before_projection_cache[h].transpose() * grad; // (B, d_v, d_model)
            printf("dW_o(%d, %d, %d)\n", dW_o.batch_size, dW_o.rows, dW_o.cols);
            for(int i = 0;i < W_o[h].rows;i++) {
                for(int j = 0;j < W_o[h].cols;j++) {
                    float temp = W_o[h](i, j);
                    for(int b = 0;b < B;b++) {
                        temp += dW_o(b, i, j);
                    }
                    W_o[h](i, j) = temp;
                }
            }
        }

        auto end_individual = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_individual = end_individual - start_individual;
        printf("MHAL Backward individual time: %f\n", duration_individual.count());
        // concate dX
        // for(int h = 0; h < num_heads;h++) {
        //     int col_index = h * d_k;
        //     for(int b = 0;b < B;b++) {
        //         for(int i = 0;i < N;i++) {
        //             for(int j = 0;j < d_k;j++) {
        //                 dX_concate(b, i, col_index + j) = dX_list[h](b, i, j);
        //             }
        //         }
        //     }
        // }
        
        printf("dX_concate(%d, %d, %d)\n", dX_concate.batch_size, dX_concate.rows, dX_concate.cols);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - end_individual;
        printf("MHAL Backward Single time: %f\n", duration.count());
        return dX_concate; // need concatenate
    }
};

#endif