#ifndef DROPOUT_H_
#define DROPOUT_H_

#include "matrix.h"

class Dropout {
private:
    float _dropout_prob;
    int _seed;

public:
    Dropout(float dropout_prob, int seed);

    ~Dropout();

    Matrix forward(const Matrix& input, int batch_size, int hidden_size);
};

#endif