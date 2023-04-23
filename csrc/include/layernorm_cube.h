#ifndef LAYERNORM_H_
#define LAYERNORM_H_

#include "matrix.h"
#include "cube.h"

class LayerNorm_cube {
private:
    int hidden_size;
    float* gamma;
    float* beta;
    float epsilon;

public:
    LayerNorm_cube(int hidden_size, float epsilon=1e-5) :
    hidden_size(hidden_size), epsilon(epsilon) {
        gamma = new float[hidden_size];
        beta = new float[hidden_size];
        // initialize gamma and beta to ones and zeros, respectively
        for (int i = 0; i < hidden_size; i++) {
            gamma[i] = 1.0;
            beta[i] = 0.0;
        }
#ifdef DEBUG
        printf("LayerNorm_cube::LayerNorm_cube(hidden_size=%d)\n", hidden_size);
#endif
    }

    ~LayerNorm_cube() {
        delete[] gamma;
        delete[] beta;
    }

    Cube forward(Cube input) {
    Cube output(input.batch_size, input.rows, input.cols);
    // compute mean and variance of input for each sample in the batch
    for (int b = 0; b < output.batch_size; b++) {
        for (int i = 0; i < output.rows; i++) {
            float mean = 0.0;
            float variance = 0.0;
            for (int j = 0; j < output.cols; j++) {
                mean += input(b,i,j);
                variance += input(b,i,j) * input(b,i,j);
            }
            mean /= output.cols;
            variance = variance / output.cols - mean * mean;

            // normalize input using mean and variance
            for (int j = 0; j < output.cols; j++) {
                output(b,i,j) = (input(b,i,j) - mean) / sqrt(variance + epsilon);
            }

            // apply scaling and shifting using gamma and beta
            for (int j = 0; j < output.cols; j++) {
                output(b,i,j) = gamma[j] * input(b,i,j) + beta[j];
            }
        }
    }

    return output;
}

    Matrix backward(Matrix grad) {

    }
};

#endif
