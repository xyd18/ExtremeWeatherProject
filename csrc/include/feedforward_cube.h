#ifndef FEEDFORWARDLAYER_H_
#define FEEDFORWARDLAYER_H_

#include "matrix.h"
#include "linearlayer_cube.h"
#include "cube.h"
#include "common.h"

class FeedForwardLayer_Cube {

private:
    int input_size; // default for 512
    int hidden_size; // default for 1024
    LinearLayer_cube linear1;
    LinearLayer_cube linear2;
    Cube hidden;

public:
    FeedForwardLayer_Cube(int input_size=512, int hidden_size=1024)
        : input_size(input_size), hidden_size(hidden_size),
        linear1(input_size, hidden_size), linear2(hidden_size, input_size){

        linear1.reset();
        linear2.reset();

        std::cout << "FeedForwardLayer::FeedForwardLayer()" << std::endl;
    }

    ~FeedForwardLayer_Cube(){
    }

    // Forward pass through the feed-forward layer
    // The input matrix is now of shape [batch_size, seq_length, input_size]
    Cube forward(const Cube& input) {
        // Pass input through the first linear layer
        std::cout << "batch_size: " << input.batch_size << std::endl;
        std::cout << "linear input: " << input.rows << " " << input.cols << std::endl;
        hidden = linear1.forward(input);
        std::cout << "linear hidden: " << hidden.rows << " " << hidden.cols << std::endl;

        // Apply activation function (e.g., ReLU) to the output of the first linear layer
        common::relu_cube(hidden);

        // Pass the result through the second linear layer
        Cube output = linear2.forward(hidden);
        std::cout << "linear output: " << output.rows << " " << output.cols << std::endl;

        return output;
    }

    // Backward pass through the feed-forward layer
    // The gradient matrix is now of shape [batch_size, seq_length, hidden_size]
    Cube backward(const Cube& grad) {

    }
};

#endif