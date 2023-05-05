#ifndef LINEAR_H_
#define LINEAR_H_

#include "model.h"
#include "matrix.h"
#include "cube.h"

class LinearLayer_cube : public Model {
public:
    Matrix weight;  // Weight matrix
    Cube inputCopy;  // Copy of input matrix for backward pass
    float* bias;    // Bias vector
    float learning_rate = 0.00001f;

    // Constructor with specified input dimension and output dimension
    LinearLayer_cube(int input_dim, int output_dim) : weight(input_dim, output_dim) {
        bias = new float[output_dim];
        for (int i = 0; i < output_dim; ++i) {
            bias[i] = 0.0f;
        }
    }

    LinearLayer_cube(const LinearLayer_cube& other) :
        weight(other.weight.rows, other.weight.cols),
        inputCopy(other.inputCopy),
        bias(new float[other.weight.cols]),
        learning_rate(other.learning_rate)
    {
        // Copy bias vector
        for (int i = 0; i < weight.cols; ++i) {
            bias[i] = other.bias[i];
        }
    }

    ~LinearLayer_cube() {
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

    /* Forward pass of the linear layer
     * Input shape: (batch_size, seq_length, input_size)
     * Output shape: (batch_size, seq_length, output_size)
     */
    Cube forward(const Cube& input) override {
        if (input.cols != weight.rows) {
            throw std::runtime_error("Cube dimensions do not match for LinearLayer weight.");
        }

        // Copy input matrix for backward pass
        inputCopy = input;

        // Perform matrix multiplication (affine transformation)
        Cube output = input * weight;

        // Add bias term to each output row
        for (int b = 0; b < output.batch_size; ++b) {
            for (int i = 0; i < output.rows; ++i) {
                for (int j = 0; j < output.cols; ++j) {
                    output(b, i, j) += bias[j];
                }
            }
        }

        return output;
    }

    /** Backward pass of the linear layer
     * dL/dX = dL/dY * dY/dX = dL/dY * W
     * dL/dW = dL/dY * dY/dW = dL/dY * X = X^T * dL/dY
     * dL/db = dL/dY * dY/db = dL/dY * 1 (vector of batch size of 1)
    */
    Cube backward(const Cube& grad) {
        // Compute gradient w.r.t. weight
        std::cout << "===================Linear Backward===================" << std::endl;
        printf("Input Copy(%d, %d, %d)\n", inputCopy.batch_size, inputCopy.rows, inputCopy.cols);
        Cube grad_weight = inputCopy.transpose() * grad;
        printf("grad_weight(%d, %d, %d)\n", grad_weight.batch_size, grad_weight.rows, grad_weight.cols);

        // Compute gradient w.r.t. bias
        std::vector<std::vector<float>> grad_bias(grad.batch_size, std::vector<float>(grad.rows, 0.f));
        for(int b = 0;b < grad.batch_size;b++) {
            for(int i = 0;i < grad.rows;i++) {
                float sum = 0.0f;
                for(int j = 0;j < grad.cols;j++) {
                    sum += grad(b, i, j);
                }
                grad_bias[b][i] = sum;
            }
        }

        // Compute gradient w.r.t. input
        Cube grad_x = grad * weight.transpose();
        printf("grad_x(%d, %d, %d)\n", grad_x.batch_size, grad_x.rows, grad_x.cols);

        // Update weights and biases
        for(int i = 0;i < weight.rows; i++) {
            for(int j = 0;j < weight.cols;j++) {
                for(int b = 0;b < grad.batch_size;b++) {
                    weight(i, j) -= learning_rate * grad_weight(b, i, j);
                }
            }
        }
        for (int i = 0; i < grad.cols; ++i) {
            for(int b = 0;b < grad.batch_size;b++) {
                bias[i] += grad_bias[b][i];
            }
        }

        return grad_x;
    }
};

#endif