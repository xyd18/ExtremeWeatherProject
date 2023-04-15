#ifndef LAYERNORM_H_
#define LAYERNORM_H_

#include "matrix.h"

class LayerNorm {
private:
    int hidden_size;
    float* gamma;
    float* beta;
    float epsilon;

public:
    LayerNorm(int hidden_size, float epsilon);

    ~LayerNorm();

    Matrix forward(Matrix input);

    Matrix backward(Matrix grad);
};

#endif
