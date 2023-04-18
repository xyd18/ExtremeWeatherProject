#ifndef CUBE_H_
#define CUBE_H_

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <random>

class Cube {
public:
    int batch_size;
    int rows;
    int cols;
    float* data;

    // Default constructor
    Cube() : batch_size(0), rows(0), cols(0), data(nullptr) {}

    Cube(int batch_size, int rows, int cols) : batch_size(batch_size), rows(rows), cols(cols) {
        data = new float[batch_size * rows * cols];
    }

    ~Cube() {
        // delete[] data; // FIXME: this line will cause double free, comment it out for now
    }

    void reset() {
        // Random number engine and distribution
        std::default_random_engine generator;                            // You can seed this with a fixed value or time-based seed
        std::uniform_real_distribution<float> distribution(-0.1f, 0.1f); // Uniform distribution in the range [-0.1, 0.1]

        for (int i = 0; i < batch_size * rows * cols; ++i) {
            data[i] = distribution(generator);
        }
    }

    void setZero() {
        for (int i = 0; i < batch_size * rows * cols; ++i) {
            data[i] = 0.0f;
        }
    }

    float& operator()(int batch, int row, int col) {
        return data[batch * rows * cols + row * cols + col];
    }

    const float& operator()(int row, int col) const {
        return data[batch * rows * cols + row * cols + col];
    }

    // Multiply two matrices within the cube along the rows and columns
    Cube operator*(const Cube& other) const {
        if (batch_size != other.batch_size || cols != other.rows) {
            throw std::runtime_error("Cube dimensions do not match for multiplication.");
        }

        Cube result(batch_size, rows, other.cols);
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < other.cols; ++j) {
                    for (int k = 0; k < cols; ++k) {
                        result(b, i, j) += (*this)(b, i, k) * other(b, k, j);
                    }
                }
            }
        }
        return result;
    }

    // Element-wise addition of two matrices
    Cube operator+(const Cube& other) const {
        if (batch_size != other.batch_size || rows != other.rows || cols != other.cols) {
            throw std::runtime_error("Cube dimensions do not match for element-wise addition.");
        }
        Cube result(batch_size, rows, cols);
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    result(b, i, j) = (*this)(b, i, j) + other(b, i, j);
                }
            }
        }
        return result;
    }

    // Transpose each matrix within the cube along the rows and columns
    Cube transpose() const {
        Cube result(batch_size, cols, rows);
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    result(b, j, i) = (*this)(b, i, j);
                }
            }
        }
        return result;
    }
};

#endif