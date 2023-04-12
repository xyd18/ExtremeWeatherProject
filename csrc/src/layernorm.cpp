#include <cmath>

#include "../include/layernorm.h"

LayerNorm::LayerNorm(int hidden_size, float epsilon) {
    hidden_size = hidden_size;
    epsilon = epsilon; // 1e-5 would be a good default value
    gamma = new float[hidden_size];
    beta = new float[hidden_size];
    // initialize gamma and beta to ones and zeros, respectively
    for (int i = 0; i < hidden_size; i++) {
        gamma[i] = 1.0;
        beta[i] = 0.0;
    }
}

LayerNorm::~LayerNorm() {
    delete[] gamma;
    delete[] beta;
}

Matrix LayerNorm::forward(Matrix input) {
    Matrix output(input.rows, input.cols);
    // compute mean and variance of input for each sample in the batch
    for (int i = 0; i < output.rows; i++) {
        float mean = 0.0;
        float variance = 0.0;
        for (int j = 0; j < hidden_size; j++) {
            mean += input(i,j);
            variance += input(i,j) * input(i,j);
        }
        mean /= hidden_size;
        variance = variance / hidden_size - mean * mean;

        // normalize input using mean and variance
        for (int j = 0; j < output.cols; j++) {
            output(i,j) = (input(i,j) - mean) / sqrt(variance + epsilon);
        }

        // apply scaling and shifting using gamma and beta
        for (int j = 0; j < hidden_size; j++) {
            output(i,j) = gamma[j] * input(i,j) + beta[j];
        }
    }
    return output;
}