#ifndef FEEDFORWARDLAYER_TMP_H_
#define FEEDFORWARDLAYER_TMP_H_

#include "matrix.h"
#include "common.h"

class FeedForwardLayerTMP {
    private:
        int input_size;
        int hidden_size; // default for 1024
        LinearLayer linear1;
        LinearLayer linear2;
        Matrix hidden;
        // MPI settting
        int nproc;
        int pid;
    
    public:
        FeedForwardLayerTMP(int input_size, int hidden_size, int pid, int nproc)
            : input_size(input_size), hidden_size(hidden_size),
              linear1(input_size, hidden_size / nproc), linear2(hidden_size / nproc, input_size),
              pid(pid), nproc(nproc) {
            
            linear1.reset();
            linear2.reset();
            std::cout << "FeedForwardLayerTMP linear1 W shape " << linear1.weight.rows << "," << linear1.weight.cols  << std::endl;
            std::cout << "FeedForwardLayerTMP linear2 W shape " << linear2.weight.rows << "," << linear2.weight.cols  << std::endl;
        }

        Matrix forward(const Matrix& input) {
            // forward pass
            std::cout << "[FeedForwardLayerTMP, W" << pid << "] input shape: " << input.rows << " " << input.cols << std::endl;
            hidden = linear1.forward(input);
            std::cout << "[FeedForwardLayerTMP, W" << pid << "] hidden shape: " << hidden.rows << " " << hidden.cols << std::endl;

            // Apply activation function (e.g., ReLU) to the output of the first linear layer
            common::relu(hidden);

            // Pass the result through the second linear layer
            Matrix output = linear2.forward(hidden);
            std::cout << "linear output: " << output.rows << " " << output.cols << std::endl;
            return output;
        }
};

#endif