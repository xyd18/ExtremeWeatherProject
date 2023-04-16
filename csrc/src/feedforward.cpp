#include <cmath>
#include <iostream>
#include "../include/feedforward.h"
#include "../include/common.h"

FeedForwardLayer::FeedForwardLayer(int input_size, int hidden_size)
    : input_size(input_size), hidden_size(hidden_size),
    linear1(input_size, hidden_size), linear2(hidden_size, input_size){

    linear1.reset();
    linear2.reset();

    std::cout << "FeedForwardLayer::FeedForwardLayer()" << std::endl;
}

FeedForwardLayer::~FeedForwardLayer()
{
}

Matrix FeedForwardLayer::forward(const Matrix &input)
{
    // Pass input through the first linear layer
    std::cout << "linear input: " << input.rows << " " << input.cols << std::endl;
    hidden = linear1.forward(input);
    std::cout << "linear hidden: " << hidden.rows << " " << hidden.cols << std::endl;

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
*/
Matrix FeedForwardLayer::backward(const Matrix &grad) {
    Matrix grad_relu = linear2.backward(grad);
    std::cout << "linear2 backward output: " << grad_relu.rows << " " << grad_relu.cols << std::endl;
    std::cout << "hidden layer shape: " << hidden.rows << " " << hidden.cols << std::endl;
    for (int i = 0; i < hidden.rows; ++i)
    {
        for (int j = 0; j < hidden.cols; ++j)
        {
            grad_relu(i, j) = hidden(i, j) > 0 ? grad_relu(i, j) : 0;
        }
    }
    return linear1.backward(grad_relu);
}

// int main()
// {
//     // Instantiate the FeedForwardLayer
//     FeedForwardLayer model(512, 2048);

//     // Define batch size and input size
//     int batch_size = 4;
//     int input_size = 512;

//     // Create a sample input array (batch size: 4, input size: 512)
//     Matrix input(batch_size, input_size);
//     for (int i = 0; i < batch_size * input_size; ++i)
//     {
//         input.data[i] = static_cast<float>(rand()) / RAND_MAX; // Random values between 0 and 1
//     }

//     // Print a few input values for demonstration
//     std::cout << "Input values (first 5):" << std::endl;
//     for (int i = 0; i < 5; ++i)
//     {
//         std::cout << input.data[i] << " ";
//     }
//     std::cout << std::endl;

//     // Perform the forward pass
//     Matrix output = model.forward(input); // 4 * 512

//     // Print a few output values for demonstration
//     std::cout << "Output values (first 5):" << std::endl;
//     for (int i = 0; i < 5; ++i)
//     {
//         std::cout << output.data[i] << " ";
//     }
//     std::cout << std::endl;

//     return 0;
// }
