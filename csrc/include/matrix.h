#ifndef MATRIX_H_
#define MATRIX_H_

#include <iostream>
#include <cmath>
#include <stdexcept>

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

    ~Matrix() {
        delete[] data;
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

    Matrix transponse() const {
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
    float* bias;    // Bias vector

    // Constructor with specified input dimension and output dimension
    LinearLayer(int input_dim, int output_dim) : weight(input_dim, output_dim) {
        bias = new float[output_dim];
        for (int i = 0; i < output_dim; ++i) {
            bias[i] = 0.0f;
        }
        std::cout << "Linear layer initialized with input dimension " << input_dim << " and output dimension " << output_dim << std::endl;
    }

    ~LinearLayer() {
        delete[] bias;
    }

    // Forward pass of the linear layer
    Matrix forward(const Matrix& input) {
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
};

#endif