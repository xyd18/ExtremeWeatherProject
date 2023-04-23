#ifndef DROPOUT_H_
#define DROPOUT_H_

#include <iostream>
#include <random>
#include "matrix.h"

class Dropout {
private:
    float _dropout_prob;
    int _seed;

public:
    Dropout(float dropout_prob, int seed = 0) {
        _dropout_prob = dropout_prob;
        _seed = seed;
    }

    ~Dropout() {}

    Matrix forward(const Matrix& input, int batch_size, int hidden_size) {
        Matrix output(batch_size, hidden_size);

        if (_dropout_prob == 0.0) {
            // Copy the input to output without any modification
            std::copy(&input(0,0), &input(0,0) + batch_size * hidden_size, &output(0,0));
            return output;
        }

        std::default_random_engine generator(_seed);
        std::uniform_real_distribution<float> distribution(0.0, 1.0);

        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < hidden_size; j++) {
                float rand_num = distribution(generator);
                if (rand_num < _dropout_prob) {
                    output(i,j) = 0.0;
                } else {
                    output(i,j) = input(i,j) / (1.0 - _dropout_prob);
                }
            }
        }
        return output;
    }
};

#endif