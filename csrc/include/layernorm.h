#ifndef LAYERNORM_H_
#define LAYERNORM_H_

#include "matrix.h"

class LayerNorm {
private:
    int hidden_size;
    float epsilon;
    float* gamma;
    float* beta;
    Matrix xmu_cache;
    Matrix var_cache;
    Matrix ivar_cache;
    Matrix sqrtvar_cache;
    Matrix xhat_cache;
    Matrix x_cache;

public:
    LayerNorm(int hidden_size, float epsilon);

    ~LayerNorm();

    Matrix forward(Matrix input);

    Matrix backward(Matrix grad);
};

#endif
