#ifndef COMMON_H_
#define COMMON_H_

#include <iostream>
#include <cmath>
#include <vector>
#include "matrix.h"

/**
 * Lce = - allSamplesSUM( allClassesSUM( label * log( probablity ) ) ) / batchSize
 * Here, a sample is a token in the sequence, input size is n * d_model
 * our label is that for each position, there is a correct token here. We expand it to n * d_model,
 * by using 0 or 1 to indicate whether the token is correct.
 * So here, labels has the same shape as output.
*/
float crossEntropyLoss(Matrix output, Matrix labels) {
    int samples_size = output.rows;
    int numer_of_classes = output.cols;
    float cross_entropy_loss = 0.f;
    for(int i = 0; i < samples_size; i++) {
        float loss = 0.f;
        for(int j = 0; j < numer_of_classes;j++) {
            loss += labels(i, j) * std::log(output(i, j));
        }
        cross_entropy_loss -= loss;
    }
    return cross_entropy_loss / samples_size;
}

void softmax(Matrix input) {
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
 * Backward pass of the cross entropy loss.For a given output token at position t, 
 * the gradient of the loss with respect to the output token is simply the difference 
 * between the predicted probability distribution and the target probability distribution
 * at that position FIXME: not entirely sure about this
 * 
 * output {x1, x2, x3}
 * CE = - (y1 * log(x1) + y2 * log(x2) + y3 * log(x3))
*/
void gradientCrossEntropy(Matrix output, std::vector<float> labels, Matrix grad) {
    int samples_size = output.rows;
    int numer_of_classes = output.cols;
    assert(grad.rows == samples_size && grad.cols == numer_of_classes);
    /* FIXME: this is not correct, depend on our sentence input */
    Matrix labelExtend(samples_size, numer_of_classes);
    for(int i = 0;i < samples_size;i++) {
        labelExtend(i, labels[i]) = 1.f;
    }
    
    for(int i = 0; i < samples_size; i++) {
        for(int j = 0; j < numer_of_classes;j++) {
            grad(i, j) = output(i, j) - labelExtend(i, j);
        }
    }
    return;
}

#endif