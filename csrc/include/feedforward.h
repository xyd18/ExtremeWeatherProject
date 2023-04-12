#ifndef FEEDFORWARDLAYER_H_
#define FEEDFORWARDLAYER_H_

#include "matrix.h"

class FeedForwardLayer {

private:
    int input_size;
    int hidden_size; // default for 1024
    LinearLayer linear1;
    LinearLayer linear2;

public:
    FeedForwardLayer(int input_size, int hidden_size);

    ~FeedForwardLayer();

    // Forward pass through the feed-forward layer
    Matrix forward(const Matrix& input);
};

#endif