#ifndef COMMON_H_
#define COMMON_H_

#include <iostream>
#include <cmath>
#include <vector>
#include <assert.h>  
#include "matrix.h"
#include "cube.h"

namespace common {

/**
 * Lce = - allSamplesSUM( allClassesSUM( label * log( probablity ) ) ) / batchSize
 * Here, a sample is a token in the sequence, input size is n * d_model
 * our label is that for each position, there is a correct token here. We expand it to n * d_model,
 * by using 0 or 1 to indicate whether the token is correct.
 * So here, labels need to be one-hot to the same shape as output.
*/
inline float crossEntropyLoss(Matrix output, std::vector<int> &labels) {
    int samples_size = output.rows;
    int numer_of_classes = output.cols;
    Matrix labels_one_hot = Matrix(samples_size, numer_of_classes);
    for(int i = 0;i < samples_size; i++) {
        labels_one_hot(i, labels[i]) = 1.f;
    }
    float cross_entropy_loss = 0.f;
    for(int i = 0; i < samples_size; i++) {
        float loss = 0.f;
        for(int j = 0; j < numer_of_classes;j++) {
            loss += labels_one_hot(i, j) * std::log(output(i, j));
        }
        cross_entropy_loss -= loss;
    }
    return cross_entropy_loss / samples_size;
}

inline void softmax(Matrix input) {
    for (int i = 0; i < input.rows; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < input.cols; ++j) {
            input(i, j) = std::exp(input(i, j));
            sum += input(i, j);
        }
        for (int j = 0; j < input.cols; ++j) {
            input(i, j) /= sum;
        }
    }
}

/**
 * Backward pass of softmax + cross entropy loss.For a given output token at position t, 
 * the gradient of the loss with respect to the output token is simply the difference 
 * between the predicted probability distribution and the target probability distribution
 * at that position. Details of derivation: https://deepnotes.io/softmax-crossentropy
*/
inline Matrix softMaxCrossEntropyBackward(Matrix output, std::vector<int> &labels) {
    std::cout << "===================SoftmaxCrossEntropyBackward===================" << std::endl;
    std::cout << "output: " << output.rows << " " << output.cols << std::endl;
    std::cout << "labels: " << labels.size() << std::endl;
    int samples_size = output.rows;
    int numer_of_classes = output.cols;
    Matrix grad(samples_size, numer_of_classes);

    Matrix labels_one_hot = Matrix(samples_size, numer_of_classes);
    for(int i = 0;i < samples_size; i++) {
        labels_one_hot(i, labels[i]) = 1.f;
    }
    
    for(int i = 0; i < samples_size; i++) {
        for(int j = 0; j < numer_of_classes;j++) {
            grad(i, j) = output(i, j) - labels_one_hot(i, j);
        }
    }
    std::cout << "output grad: " << grad.rows << " " << grad.cols << std::endl;
    return grad;
}

/**
 * A modified version of softMaxCrossEntropyBackward for Cube data structure.
*/
inline Cube softMaxCrossEntropyBackwardCube(Cube output, Cube labels) {
    std::cout << "===================SoftmaxCrossEntropyBackward===================" << std::endl;
    printf("output(%d, %d, %d)\n", output.batch_size, output.rows, output.cols);
    printf("labels(%d, %d, %d)\n", labels.batch_size, labels.rows, labels.cols);
    Cube grad(output.batch_size, output.rows, output.cols);
    for(int b = 0;b < output.batch_size;b++) {
        for(int n = 0;n < output.rows;n++) {
            for(int c = 0;c < output.cols;c++) {
                grad(b, n, c) = output(b, n, c) - labels(b, n, c);
            }
        }
    }
    printf("output grad(%d, %d, %d)\n", grad.batch_size, grad.rows, grad.cols);
    return grad;
}

inline Matrix softmaxBackward(Matrix input) {
    Matrix grad(input.rows, input.cols);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            grad(i, j) = input(i, j) * (1 - input(i, j));
        }
    }
    return grad;
}

inline Cube softmax_backwar_cube(Cube input) {
    Cube grad(input.batch_size, input.rows, input.cols);
    for(int b = 0;b < input.batch_size;b++) {
        for(int n = 0;n < input.rows;n++) {
            for(int c = 0;c < input.cols;c++) {
                grad(b, n, c) = input(b, n, c) * (1 - input(b, n, c));
            }
        }
    }
    return grad;
}

inline void relu(Matrix input) {
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input(i, j) = std::max(0.0f, input(i, j));
        }
    }
}

inline void reluBackward(Matrix grad, Matrix input) {
    for (int i = 0; i < input.rows; ++i)
    {
        for (int j = 0; j < input.cols; ++j)
        {
            grad(i, j) = input(i, j) > 0 ? grad(i, j) : 0;
        }
    }
}


inline void relu_cube(Cube input) {
    for (int b = 0; b < input.batch_size; ++b) {
        for (int i = 0; i < input.rows; ++i) {
            for (int j = 0; j < input.cols; ++j) {
                input(b, i, j) = std::max(0.0f, input(b, i, j));
            }
        }
    }
    
}

inline void relu_backward_cube(Cube grad, Cube input) {
    for(int b = 0;b < input.batch_size;b++) {
        for (int i = 0; i < input.rows; ++i) {
            for (int j = 0; j < input.cols; ++j) {
                grad(b, i, j) = input(b, i, j) > 0 ? grad(b, i, j) : 0;
            }
        }
    }
}

// inline void concate_cube(std::vector<Cube> input_list, int dimension) {
//     switch (dimension) {
//         case 2
//             Cube output(input_list[0].batch_size, input_list[0].rows, input_list[0].cols * input_list.size());
//             for(int b = 0;b < input_list[0].batch_size;b++) {
//                 for(int i = 0;i < input_list[0].rows;i++) {
//                     for(int j = 0;j < input_list[0].cols;j++) {
//                         for(int i = 0;i < input_list.size();i++) {
//                             output(b, n, c * input_list.size() + i) = input_list[i](b, n, c);
//                         }
//                     }
//                 }
//             }
//         default:
//             break;
//     }
// }

} // namespace common
#endif
