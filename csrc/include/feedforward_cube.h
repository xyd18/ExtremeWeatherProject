#ifndef FEEDFORWARDLAYER_H_
#define FEEDFORWARDLAYER_H_

#include "model.h"
#include "matrix.h"
#include "linearlayer_cube.h"
#include "cube.h"
#include "common.h"

class FeedForwardLayer_Cube : public Model {

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

    }

    // Copy constructor
    FeedForwardLayer_Cube(const FeedForwardLayer_Cube& other)
        : input_size(other.input_size), hidden_size(other.hidden_size),
        linear1(other.linear1), linear2(other.linear2), hidden(other.hidden) {}

    ~FeedForwardLayer_Cube(){
    }

    // Forward pass through the feed-forward layer
    // The input matrix is now of shape [batch_size, seq_length, input_size]
    Cube forward(const Cube& input) override {

        #ifdef DEBUG
        printf("FeedForwardLayer input size: (batch_size=%d, seq_length=%d, d_model=%d)\n", input.batch_size, input.rows, input.cols);
        #endif

        // Pass input through the first linear layer
        hidden = linear1.forward(input);

        #ifdef DEBUG
        printf("FeedForwardLayer hidden size: (batch_size=%d, seq_length=%d, hidden_size=%d)\n", hidden.batch_size, hidden.rows, hidden.cols);
        #endif

        // Apply activation function (e.g., ReLU) to the output of the first linear layer
        common::relu_cube(hidden);

        // Pass the result through the second linear layer
        Cube output = linear2.forward(hidden);

        #ifdef DEBUG
        printf("FeedForwardLayer output size: (batch_size=%d, seq_length=%d, d_model=%d)\n", output.batch_size, output.rows, output.cols);
        #endif

        return output;
    }

    // Backward pass through the feed-forward layer
    // The gradient matrix is now of shape [batch_size, seq_length, hidden_size]
    Cube backward(const Cube& grad) {
        return Cube(grad);
    }
};

#endif