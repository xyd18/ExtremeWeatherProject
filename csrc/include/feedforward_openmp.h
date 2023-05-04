#ifndef FEEDFORWARDLAYER_OPENMP_H_
#define FEEDFORWARDLAYER_OPENMP_H_

#include "matrix.h"
#include "linearlayer_cube.h"
#include "cube.h"
#include "common.h"
#include <omp.h>

class FeedForwardLayer_openmp {
    private:
        int input_size; // default for 512
        int hidden_size; // default for 1024
        int num_workers;
        std::vector<LinearLayer_cube> linear1;
        std::vector<LinearLayer_cube> linear2;
        std::vector<Cube> hidden;
    
    public:
        FeedForwardLayer_openmp(int input_size, int hidden_size, int num_workers)
            : input_size(input_size), hidden_size(hidden_size), num_workers(num_workers) {
            linear1.reserve(num_workers);
            linear2.reserve(num_workers);
            hidden.reserve(num_workers);
            for (int i = 0; i < num_workers; ++i) {
                linear1.emplace_back(input_size, hidden_size / num_workers); 
                linear2.emplace_back(hidden_size / num_workers, input_size);
                linear1[i].reset();
                linear2[i].reset();
            }
#ifdef DEBUG
        std::cout << "[FeedForward OpenMP constructor] Initialization Completed." << std::endl;
#endif  
        }

        Cube forward(const Cube& input) {
            std::vector<Cube> output_list(num_workers);
            #pragma omp parallel for num_threads(num_workers)
            for(int h = 0;h < num_workers;h++) {
                hidden[h] = linear1[h].forward(input);

                // Apply activation function (e.g., ReLU) to the output of the first linear layer
                common::relu_cube(hidden[h]);

                // Pass the result through the second linear layer
                output_list[h] = linear2[h].forward(hidden[h]);
            }

            // Merge the results from different workers
            Cube output(input.batch_size, input.rows, input.cols);
            for(int b = 0;b < input.batch_size;b++) {
                for(int i = 0;i < input.rows;i++) {
                    for(int j = 0;j < input.cols;j++) {
                        float temp = 0.f;
                        for(int h = 0;h < num_workers;h++) {
                            temp += output_list[h](b, i, j);
                        }
                        output(b, i, j) = temp;
                    }
                }
            }
            return output;
        }

        Cube backward(const Cube& grad) {
            std::cout << "===================Feed Forward Backward===================" << std::endl;
            printf("grad(%d, %d, %d)\n", grad.batch_size, grad.rows, grad.cols);
            std::vector<Cube> output_list(num_workers);

            #pragma omp parallel for num_threads(num_workers)
            for(int h = 0;h < num_workers;h++) {
                Cube grad_relu = linear2[h].backward(grad);
                printf("grad_relu(%d, %d, %d)\n", grad_relu.batch_size, grad_relu.rows, grad_relu.cols);
                printf("hidden(%d, %d, %d)\n", hidden[h].batch_size, hidden[h].rows, hidden[h].cols);
                common::relu_backward_cube(grad_relu, hidden[h]);
                output_list[h] = linear1[h].backward(grad_relu);
                printf("output slice(%d, %d, %d)\n", output_list[h].batch_size, output_list[h].rows, output_list[h].cols);
            }

            // Merge the results from different workers
            Cube output(grad.batch_size, grad.rows, grad.cols);
            for(int b = 0;b < grad.batch_size;b++) {
                for(int i = 0;i < grad.rows;i++) {
                    for(int j = 0;j < grad.cols;j++) {
                        float temp = 0.f;
                        for(int h = 0;h < num_workers;h++) {
                            temp += output_list[h](b, i, j);
                        }
                        output(b, i, j) = temp;
                    }
                }
            }
            return output;
        }
};

#endif