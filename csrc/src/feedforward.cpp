#include <cmath>
#include <iostream>
#include <random>

#include "../include/feedforward.h"

FeedForwardLayer::FeedForwardLayer(int input_size, int hidden_size)
    : input_size(input_size), hidden_size(hidden_size),
    linear1(input_size, hidden_size), linear2(hidden_size, input_size){

    // Random number engine and distribution
    std::default_random_engine generator;                            // You can seed this with a fixed value or time-based seed
    std::uniform_real_distribution<float> distribution(-0.1f, 0.1f); // Uniform distribution in the range [-0.1, 0.1]

    for (int i = 0; i < input_size; ++i)
    {
        for (int j = 0; j < hidden_size; ++j)
        {
            linear1.weight(i, j) = distribution(generator); // Initialize with random value
        }
    }
    for (int i = 0; i < hidden_size; ++i)
    {
        linear1.bias[i] = distribution(generator); // Initialize with random value
    }

    for (int i = 0; i < hidden_size; ++i)
    {
        for (int j = 0; j < input_size; ++j)
        {
            linear2.weight(i, j) = distribution(generator); // Initialize with random value
        }
    }
    for (int i = 0; i < input_size; ++i)
    {
        linear2.bias[i] = distribution(generator); // Initialize with random value
    }
}

FeedForwardLayer::~FeedForwardLayer()
{
}

Matrix FeedForwardLayer::forward(const Matrix &input)
{
    // Pass input through the first linear layer
    Matrix hidden = linear1.forward(input);

    // Apply activation function (e.g., ReLU) to the output of the first linear layer
    for (int i = 0; i < hidden.rows; ++i)
    {
        for (int j = 0; j < hidden.cols; ++j)
        {
            hidden(i, j) = std::max(0.0f, hidden(i, j));
        }
    }

    // Pass the result through the second linear layer
    Matrix output = linear2.forward(hidden);

    return output;
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
