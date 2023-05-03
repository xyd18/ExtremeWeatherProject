#ifndef FEEDFORWARDLAYER_TMP_H_
#define FEEDFORWARDLAYER_TMP_H_

#include "matrix.h"
#include "linearlayer_cube.h"
#include "cube.h"
#include "common.h"

class FeedForwardLayerTMP_cube {
    private:
        int input_size; // default for 512
        int hidden_size; // default for 1024
        LinearLayer_cube linear1;
        LinearLayer_cube linear2;
        Cube hidden;
        // MPI settting
        int nproc;
        int pid;
    
    public:
        FeedForwardLayerTMP_cube(int input_size, int hidden_size, int pid, int nproc)
            : input_size(input_size), hidden_size(hidden_size),
              linear1(input_size, hidden_size / nproc), linear2(hidden_size / nproc, input_size),
              pid(pid), nproc(nproc) {
            
            linear1.reset();
            linear2.reset();
            if(pid == 0) {
                std::cout << "FeedForwardLayerTMP linear1 W shape " << linear1.weight.rows << "," << linear1.weight.cols  << std::endl;
                std::cout << "FeedForwardLayerTMP linear2 W shape " << linear2.weight.rows << "," << linear2.weight.cols  << std::endl;
            }
        }

        Cube forward(const Cube& input) {
            // forward pass
            // std::cout << "[FeedForwardLayerTMP, W" << pid << "] input shape: " << input.rows << " " << input.cols << std::endl;
            hidden = linear1.forward(input);
            // std::cout << "[FeedForwardLayerTMP, W" << pid << "] hidden shape: " << hidden.rows << " " << hidden.cols << std::endl;

            // Apply activation function (e.g., ReLU) to the output of the first linear layer
            common::relu_cube(hidden);

            // Pass the result through the second linear layer
            Cube output = linear2.forward(hidden);
            // std::cout << "[FeedForwardLayerTMP, W" << pid << "] output shape: " << output.rows << " " << output.cols << std::endl;
            return output;
        }

        Cube backward(const Cube& grad) {
            std::cout << "===================Feed Forward Backward===================" << std::endl;
            printf("grad(%d, %d, %d)\n", grad.batch_size, grad.rows, grad.cols);
            Cube grad_relu = linear2.backward(grad);
            printf("grad_relu(%d, %d, %d)\n", grad_relu.batch_size, grad_relu.rows, grad_relu.cols);
            printf("hidden(%d, %d, %d)\n", hidden.batch_size, hidden.rows, hidden.cols);
            common::relu_backward_cube(grad_relu, hidden);
            Cube output = linear1.backward(grad_relu);
            printf("output(%d, %d, %d)\n", output.batch_size, output.rows, output.cols);
            return output;
        }
};

#endif