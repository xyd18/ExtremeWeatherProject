#ifndef CUBE_H_
#define CUBE_H_

#include "matrix.h"
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <random>
#include <fstream>

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

    Cube(const Cube& other) {
        batch_size = other.batch_size;
        rows = other.rows;
        cols = other.cols;
        data = new float[batch_size * rows * cols];

        // Copy the data from the input object
        for (int i = 0; i < batch_size * rows * cols; ++i) {
            data[i] = other.data[i];
        }
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
        std::fill(data, data + batch_size * rows * cols, 0.0f);
        // for (int i = 0; i < batch_size * rows * cols; ++i) {
        //     data[i] = 0.0f;
        // }
    }

    float& operator()(int batch, int row, int col) {
        return data[batch * rows * cols + row * cols + col];
    }

    const float& operator()(int batch, int row, int col) const {
        return data[batch * rows * cols + row * cols + col];
    }

    /* Matrix multiplication along the last dimension
     * Input shape: (batch_size, seq_length, input_size) = (batch_size, rows, cols)
     * Weight shape: (input_size, output_size) = (other.rows, other.cols)
     * Output shape: (batch_size, seq_length, output_size) = (batch_size, rows, other.cols)
     */
    Cube operator*(const Matrix& other) const {

        if (cols != other.rows) {
            throw std::runtime_error("Cube dimensions do not match for multiplication.");
        }

        Cube result(batch_size, rows, other.cols);
        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < rows; ++s) {
                for (int i = 0; i < cols; ++i) {
                    for (int o = 0; o < other.cols; ++o) {
                        result(b,s,o) += (*this)(b, s, i) * other(i, o);
                    }
                }
            }
        }
        return result;
    }

    // Multiply two matrices within the cube along the rows and columns
    // Input shape: (batch_size, seq_length_a, input_size)
    // Other shape: (batch_size, input_size, seq_length_b)
    // Output shape: (batch_size, seq_length_a, seq_length_b)
    Cube operator*(const Cube& other) const {
        if (batch_size != other.batch_size || cols != other.rows) {
            throw std::runtime_error("Cube dimensions do not match for multiplication.");
        }

        Cube result(batch_size, rows, other.cols);
        // Perform batched matrix multiplication
        for (int batch = 0; batch < batch_size; ++batch) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < other.cols; ++j) {
                    float sum = 0.0;
                    for (int k = 0; k < cols; ++k) {
                        sum += (*this)(batch, i, k) * other(batch,k,j);
                    }
                    result(batch,i,j) = sum;
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

    // Serialize the Cube object to a file
    void save(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (file.is_open()) {
            // Write the object's data members to the file
            file.write(reinterpret_cast<const char*>(&batch_size), sizeof(int));
            file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
            file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
            file.write(reinterpret_cast<const char*>(data), batch_size * rows * cols * sizeof(float));
            file.close();
        }
        else {
            // Handle error opening the file
        }
    }

    // Deserialize the Cube object from a file
    void load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (file.is_open()) {
            // Read the object's data members from the file
            file.read(reinterpret_cast<char*>(&batch_size), sizeof(int));
            file.read(reinterpret_cast<char*>(&rows), sizeof(int));
            file.read(reinterpret_cast<char*>(&cols), sizeof(int));

            // Allocate memory for the data array
            data = new float[batch_size * rows * cols];

            // Read the data array from the file
            file.read(reinterpret_cast<char*>(data), batch_size * rows * cols * sizeof(float));
            file.close();
        }
        else {
            // Handle error opening the file
        }
    }
};

#endif