#ifndef MULTIHEADATTENTION_H_
#define MULTIHEADATTENTION_H_

#include "matrix.h"
#include "common.h"
#include <vector>
#include <iostream>
#include <unistd.h>  // FIXME: for sleep

class MultiHeadAttention {
public:
    int d_model;
    int d_k;
    int num_heads;
    int x_rows;
    int pid;
    int nproc;
    int heads_per_p;
    bool hasWork; // record whether this process has work to do in MHAL
    std::vector<Matrix> W_q;
    std::vector<Matrix> W_k;
    std::vector<Matrix> W_v;
    std::vector<Matrix> Q_cache;
    std::vector<Matrix> K_cache;
    std::vector<Matrix> V_cache;
    std::vector<Matrix> QK_softmax_cache;
    Matrix X_cache;
    Matrix after_concat_cache;
    Matrix W_o;

    MultiHeadAttention(int d_model, int d_k, int num_heads, int nproc, int pid) 
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
            Q_cache.reserve(heads_per_p);
            K_cache.reserve(heads_per_p);
            V_cache.reserve(heads_per_p);
            QK_softmax_cache.reserve(heads_per_p);
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

    ~MultiHeadAttention() {}

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
    Matrix forward(const Matrix& X) {
        if(!hasWork) return Matrix(0, 0); // FIXME: this is errorneous if nproc > num_heads, because wrong shape will cause error in reduction, need a sub group of the world
        x_rows = X.rows;
        X_cache = X;
        Matrix concat_heads(x_rows, d_k * heads_per_p);
        for (int h = 0; h < heads_per_p; ++h) {
            Matrix Q = X * W_q[h]; // n * d_k
            Matrix K = X * W_k[h]; // n * d_k
            Matrix V = X * W_v[h]; // n * d_v 
            Q_cache[h] = Q;
            K_cache[h] = K;
            V_cache[h] = V;

            Matrix attention_scores = Q * K.transpose(); // n * n
            // std::cout << "attention_scores shape: " << attention_scores.rows << " " << attention_scores.cols << std::endl;

            // scaled + softmax
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

            QK_softmax_cache[h] = attention_scores;
            Matrix head = attention_scores * V; // n * d_v, d_v = d_k

            // concatenate head
            int col_index = h * d_k;
            for (int i = 0; i < X.rows; ++i) { // concate on last dimension
                for (int j = 0; j < d_k; ++j) {
                    concat_heads(i, col_index + j) = head(i, j);
                }
            }
        }
        after_concat_cache = concat_heads;
        // Project concatenated heads to output
        Matrix output = concat_heads * W_o;
        return output;
    }

    /** Backward pass of the multi-head attention layer
     * dQ = dY * Wq, dK = dY * Wk, dV = dY * Wv
     * dA_q = softmax_prime(A) * (dQ * K.T), dA_k = softmax_prime(A) * (Q.T * dK), dA_v = softmax_prime(A) * (Q.T * dV)
     * dQ_in = dA_q * K + alpha * dQ, dK_in = dA_k * Q + alpha * dK, dV_in = dA_v * Q + alpha * dV
     * dWq = X_in.T * dQ_in, dWk = X_in.T * dK_in, dWv = X_in.T * dV_in
     */
    Matrix backward(Matrix grad) {
        std::cout << "===================MHAL Backward===================" << std::endl;
        std::cout << "grad shape: " << grad.rows << " " << grad.cols << std::endl;
        // FIXME: grad seems need to be sliced here
        // FIXME: implementation is still not correct
        sleep(1);
        return grad;
        // Compute gradient w.r.t. output H
        Matrix dH = grad * W_o.transpose();
        std::cout << "dH shape: " << dH.rows << " " << dH.cols << std::endl;
        Matrix dX_concate = Matrix(x_rows, d_k * heads_per_p);
        
        for(int i = 0; i < heads_per_p;i++) {
            Matrix dH_slice = Matrix(dH.rows, d_k);
            for(int j = 0;j < dH_slice.rows;j++) {
                for(int k = 0;k < dH_slice.cols;k++) {
                    dH_slice(j, k) = dH(j, i * d_k + k);
                }
            }
            std::cout << "dH_slice shape: " << dH_slice.rows << " " << dH_slice.cols << std::endl;
            // Compute gradient w.r.t. concatenated Q,K,V
            Matrix dQ = dH_slice * W_q[i].transpose();
            Matrix dK = dH_slice * W_k[i].transpose();
            Matrix dV = dH_slice * W_v[i].transpose();
            std::cout << "dQ shape: " << dQ.rows << " " << dQ.cols << std::endl;
            std::cout << "dK shape: " << dK.rows << " " << dK.cols << std::endl;
            std::cout << "dV shape: " << dV.rows << " " << dV.cols << std::endl;
            // Compute gradient w.r.t. attention scores A
            Matrix dA = dQ * K_cache[i].transpose();
            float seq_length = Q_cache[i].rows;
            for(int j = 0;j < dA.cols * dA.rows;j++) {
                dA.data[j] /= seq_length;
            }
            std::cout << "dA shape: " << dA.rows << " " << dA.cols << std::endl;
            Matrix softmax_grad = common::softmaxBackward(QK_softmax_cache[i]);
            std::cout << "softmax_grad shape: " << softmax_grad.rows << " " << softmax_grad.cols << std::endl;
            dA = dA * softmax_grad;
            std::cout << "dA' shape: " << dA.rows << " " << dA.cols << std::endl;
            // Compute gradient w.r.t. Q,K,V weight matrices
            Matrix dW_q = X_cache.transpose() * dQ;
            Matrix dW_k = X_cache.transpose() * dK;
            Matrix dW_v = X_cache.transpose() * dV;
            std::cout << "dW_q shape: " << dW_q.rows << " " << dW_q.cols << std::endl;
            std::cout << "dW_k shape: " << dW_k.rows << " " << dW_k.cols << std::endl;
            std::cout << "dW_v shape: " << dW_v.rows << " " << dW_v.cols << std::endl;
            // Compute gradient w.r.t. input X
            Matrix dX = dQ * W_q[i].transpose() + dK * W_k[i].transpose() + dV * W_v[i].transpose();
            std::cout << "dX shape: " << dX.rows << " " << dX.cols << std::endl;
            // Update weight gradients FIXME: need learning rate?
            for(int j = 0;j < dW_q.rows;j++) {
                for(int k = 0;k < dW_q.cols;k++) {
                    W_q[i](j, i * d_k + k) += dW_q(j, k);
                    W_k[i](j, i * d_k + k) += dW_k(j, k);
                    W_v[i](j, i * d_k + k) += dW_v(j, k);
                }
            }

            // update input gradients
            Matrix dX_slice = dA * K_cache[i];
            for(int j = 0;j < dX_slice.rows;j++) {
                for(int k = 0;k < dX_slice.cols;k++) {
                    dX_concate(j, i * d_k + k) = dX_slice(j, k);
                }
            }
        }

        // update W_o
        std::cout << "after_concat_cache shape: " << after_concat_cache.rows << " " << after_concat_cache.cols << std::endl;
        Matrix dW_o = after_concat_cache.transpose() * grad;
        W_o = W_o + dW_o;
        return dX_concate; // need concatenate
    }
};

#endif