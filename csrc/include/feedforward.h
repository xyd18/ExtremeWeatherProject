#ifndef FEEDFORWARDLAYER_H_
#define FEEDFORWARDLAYER_H_

#include <cmath>
#include <iostream>
#include "common.h"
#include "matrix.h"

class FeedForwardLayer {

private:
    int input_size;
    int hidden_size; // default for 1024
    LinearLayer linear1;
    LinearLayer linear2;
    Matrix hidden;

public:
    FeedForwardLayer(int input_size, int hidden_size)
        : input_size(input_size), hidden_size(hidden_size),
        linear1(input_size, hidden_size), linear2(hidden_size, input_size) {
        linear1.reset();
        linear2.reset();
    }

    ~FeedForwardLayer() {}

    // Forward pass through the feed-forward layer
    Matrix forward(const Matrix& input)
    {
        #ifdef DEBUG
        printf("FeedForwardLayer input size: (batch_size=%d, d_model=%d)\n", input.rows, input.cols);
        #endif

        // Pass input through the first linear layer
        hidden = linear1.forward(input);

        #ifdef DEBUG
        printf("FeedForwardLayer hidden size: (batch_size=%d, hidden_size=%d)\n", hidden.rows, hidden.cols);
        #endif

        // Apply activation function (e.g., ReLU) to the output of the first linear layer
        common::relu(hidden);

        // Pass the result through the second linear layer
        Matrix output = linear2.forward(hidden);
        std::cout << "linear output: " << output.rows << " " << output.cols << std::endl;

        return output;
    }

    /**
     * Backward pass of the feedforward layer
     * Z = XW_1 + b_1
     * H = ReLU(Z)
     * Y = HW_2 + b_2
     * Given dL/dY, dL/dH = dL/dY * W_2^T (output of linear backward), 
     * dL/dZ = dL/dH * relu_prime(Z), relu has same function for both forward and backward
     * dL/dX = dL/dZ * W_1^T
     * dL/dW_1 = X^T * dL/dZ
     * dL/db_1 = sum_rows(dL/dZ) similar to linear layer
     * 
     * Y = X * W1 + b1, Z = max(0, Y), O = Z * W2 + b2 
     * Matrix dO = dout;
        db2 = dO.sum_axis(0, 1);
        dW2 = Z.transpose().dot(dO);
        dZ = dO.dot(W2.transpose());
        dY = dZ.relu_backward(Y);
        db1 = dY.sum_axis(0, 1);
        dW1 = X.transpose().dot(dY);
        return dY.dot(W1.transpose());
    */
    // Backward pass through the feed-forward layer
    Matrix backward(const Matrix& grad) {
        std::cout << "===================Feed Forward Backward===================" << std::endl;
        std::cout << "grad shape: " << grad.rows << " " << grad.cols << std::endl;
        Matrix grad_relu = linear2.backward(grad);
        std::cout << "linear2 backward output: " << grad_relu.rows << " " << grad_relu.cols << std::endl;
        std::cout << "hidden layer shape: " << hidden.rows << " " << hidden.cols << std::endl;
        common::reluBackward(grad_relu, hidden);
        std::cout << "relu backward output: " << grad_relu.rows << " " << grad_relu.cols << std::endl;
        return linear1.backward(grad_relu);
    }
};

#endif