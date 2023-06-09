#ifndef MATRIX_H_
#define MATRIX_H_

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <random>

class Matrix {
public:
    int rows;
    int cols;
    float* data;

    // Default constructor
    Matrix() : rows(0), cols(0), data(nullptr) {}

    Matrix(int rows, int cols) : rows(rows), cols(cols) {
        data = new float[rows * cols];
        for (int i = 0; i < rows * cols; ++i) {
            data[i] = 0.0f;
        }
    }

    Matrix(const Matrix& other) {
        rows = other.rows;
        cols = other.cols;
        data = new float[rows * cols];

        // Copy the data from the input object
        for (int i = 0; i < rows * cols; ++i) {
            data[i] = other.data[i];
        }
    }

    ~Matrix() {
        // delete[] data; // FIXME: this line will cause double free, comment it out for now
    }

    void reset() {
        // Random number engine and distribution
        std::default_random_engine generator;                            // You can seed this with a fixed value or time-based seed
        std::uniform_real_distribution<float> distribution(-0.1f, 0.1f); // Uniform distribution in the range [-0.1, 0.1]

        for (int i = 0; i < rows * cols; ++i) {
            data[i] = distribution(generator);
        }
    }

    float& operator()(int row, int col) {
        return data[row * cols + col];
    }

    const float& operator()(int row, int col) const {
        return data[row * cols + col];
    }

    // Multiply two matrices and return the result
    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::runtime_error("Matrix dimensions do not match for multiplication.");
        }
        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                for (int k = 0; k < cols; ++k) {
                    result(i, j) += (*this)(i, k) * other(k, j);
                }
            }
        }
        return result;
    }

    // Element-wise addition of two matrices
    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::runtime_error("Matrix dimensions do not match for addition.");
        }
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(i, j) = (*this)(i, j) + other(i, j);
            }
        }
        return result;
    }

    // Element-wise minus of two matrices
    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::runtime_error("Matrix dimensions do not match for addition.");
        }
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(i, j) = (*this)(i, j) - other(i, j);
            }
        }
        return result;
    }

    Matrix transpose() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }
};

class LinearLayer {
public:
    Matrix weight;  // Weight matrix
    Matrix inputCopy;  // Copy of input matrix for backward pass
    float* bias;    // Bias vector
    float learning_rate = 0.01f;

    // Constructor with specified input dimension and output dimension
    LinearLayer(int input_dim, int output_dim) : weight(input_dim, output_dim) {
        bias = new float[output_dim];
        for (int i = 0; i < output_dim; ++i) {
            bias[i] = 0.0f;
        }
    }

    ~LinearLayer() {
        delete[] bias;
    }

    void reset() {
        // Random number engine and distribution
        std::default_random_engine generator;                            // You can seed this with a fixed value or time-based seed
        std::uniform_real_distribution<float> distribution(-0.1f, 0.1f); // Uniform distribution in the range [-0.1, 0.1]

        for (int i = 0; i < weight.rows * weight.cols; ++i) {
            weight.data[i] = distribution(generator);
        }
        for (int i = 0; i < weight.cols; ++i) {
            bias[i] = distribution(generator);
        }
    }

    // Forward pass of the linear layer
    Matrix forward(const Matrix& input) {
        // Copy input matrix for backward pass
        inputCopy = input;

        // Perform matrix multiplication (affine transformation)
        Matrix output = input * weight;

        // Add bias term to each output row
        for (int i = 0; i < output.rows; ++i) {
            for (int j = 0; j < output.cols; ++j) {
                output(i, j) += bias[j];
            }
        }

        return output;
    }

    /** Backward pass of the linear layer
     * dL/dX = dL/dY * dY/dX = dL/dY * W
     * dL/dW = dL/dY * dY/dW = dL/dY * X = X^T * dL/dY
     * dL/db = dL/dY * dY/db = dL/dY * 1 (vector of batch size of 1)
    */
    Matrix backward(const Matrix& grad) {
        // Compute gradient w.r.t. weight, dL/dW = X^T * dL/dY
        std::cout << "Linear Backward" << std::endl;
        std::cout << "Input cache: " << inputCopy.rows << "x" << inputCopy.cols << std::endl;
        std::cout << "input Grad: " << grad.rows << "x" << grad.cols << std::endl;
        Matrix grad_weight = inputCopy.transpose() * grad;

        // Compute gradient w.r.t. bias
        std::vector<float> grad_bias(grad.cols, 0.f);
        for (int i = 0; i < grad.rows; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < grad.cols; ++j) {
                sum += grad(i, j);
            }
            grad_bias[i] = sum;
        }

        // Compute gradient w.r.t. input
        Matrix grad_x = grad * weight.transpose();

        // Update weights and biases
        for(int i = 0;i < weight.rows; i++) {
            for(int j = 0;j < weight.cols;j++) {
                weight(i, j) -= learning_rate * grad_weight(i, j);
            }
        }
        for (int i = 0; i < grad.cols; ++i) {
            bias[i] += grad_bias[i]; // FIXME: += ?
        }
        std::cout << "output grad: " << grad_x.rows << "x" << grad_x.cols << std::endl;
        return grad_x;
    }
};

#endif