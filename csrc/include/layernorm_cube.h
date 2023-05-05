#ifndef LAYERNORM_H_
#define LAYERNORM_H_

#include "model.h"
#include "matrix.h"
#include "cube.h"

class LayerNorm_cube : public Model {
private:
    int hidden_size;
    float* gamma;
    float* beta;
    float epsilon;
    Cube xmu_cache;
    Cube var_cache;
    Cube ivar_cache;
    Cube sqrtvar_cache;
    Cube xhat_cache;
    Cube x_cache;

public:
    LayerNorm_cube(int hidden_size, float epsilon=1e-5) :
    hidden_size(hidden_size), epsilon(epsilon) {
        gamma = new float[hidden_size];
        beta = new float[hidden_size];
        // initialize gamma and beta to ones and zeros, respectively
        for (int i = 0; i < hidden_size; i++) {
            gamma[i] = 1.0;
            beta[i] = 0.0;
        }
    }

    // Copy constructor
    LayerNorm_cube(const LayerNorm_cube& other) :
    hidden_size(other.hidden_size), epsilon(other.epsilon) {
        gamma = new float[hidden_size];
        beta = new float[hidden_size];
        // Copy the data from the other object
        for (int i = 0; i < hidden_size; i++) {
            gamma[i] = other.gamma[i];
            beta[i] = other.beta[i];
        }
    }

    ~LayerNorm_cube() {
        delete[] gamma;
        delete[] beta;
    }

    Cube forward(const Cube& input) override {
        if (input.cols != hidden_size)
             throw std::runtime_error("LayerNorm_cube::forward(): input.cols != hidden_size");
        x_cache = Cube(input); 
        xmu_cache = Cube(input.batch_size, input.rows, input.cols);
        var_cache =Cube(input.batch_size, input.rows, 1);
        ivar_cache = Cube(input.batch_size, input.rows, 1);
        sqrtvar_cache = Cube(input.batch_size, input.rows, 1);
        xhat_cache = Cube(input.batch_size, input.rows, input.cols);
        Cube output(input.batch_size, input.rows, input.cols);
        // compute mean and variance of input for each sample in the batch
        for (int b = 0; b < output.batch_size; b++) {
            for (int i = 0; i < output.rows; i++) {
                float mean = 0.0;
                float var = 0.0;
                for (int j = 0; j < output.cols; j++) {
                    mean += input(b,i,j);
                }
                mean /= output.cols;
                for (int j = 0; j < output.cols; j++)
                {
                    float temp = input(b,i,j) - mean;
                    xmu_cache(b,i,j) = temp;
                    var += pow(temp, 2);
                }
                var /= output.cols;
                var_cache(b, i, 0) = var;
                float sqrtvar = sqrt(var + epsilon);
                sqrtvar_cache(b, i, 0) = sqrtvar;
                ivar_cache(b, i, 0) = 1.0 / sqrtvar;
                for (int j = 0; j < output.cols; j++) {
                    xhat_cache(b,i,j) = xmu_cache(b,i,j) / sqrtvar; // xhat = xmu * ivar
                    output(b,i,j) = gamma[j] * xhat_cache(b,i,j) + beta[j];
                }  
            }
            // for (int i = 0; i < output.rows; i++) {
            //     float mean = 0.0;
            //     float variance = 0.0;
            //     for (int j = 0; j < output.cols; j++) {
            //         mean += input(b,i,j);
            //         variance += input(b,i,j) * input(b,i,j);
            //     }
            //     mean /= output.cols;
            //     variance = variance / output.cols - mean * mean;

            //     // normalize input using mean and variance
            //     for (int j = 0; j < output.cols; j++) {
            //         output(b,i,j) = (input(b,i,j) - mean) / sqrt(variance + epsilon);
            //     }

            //     // apply scaling and shifting using gamma and beta
            //     for (int j = 0; j < output.cols; j++) {
            //         output(b,i,j) = gamma[j] * input(b,i,j) + beta[j];
            //     }
            // }
        }

        return output;
    }

    /**
     * A modified version of the backward pass of the layer normalization layer, for Batch inputs
     * Reference: https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
     * Adjusted for Layer Normalization
    */
    Cube backward(Cube grad) {
        // Compute the gradient w.r.t. the mean and variance
        std::cout << "====================Layer Norm Backward=====================" << std::endl;
        printf("grad(%d, %d, %d)\n", grad.batch_size, grad.rows, grad.cols);
        printf("x_cache(%d, %d, %d)\n", x_cache.batch_size, x_cache.rows, x_cache.cols);
        printf("var_cache(%d, %d, %d)\n", var_cache.batch_size, var_cache.rows, var_cache.cols);
        int B = grad.batch_size;
        int N = grad.rows;
        int D = grad.cols;

        Cube d_b = Cube(B, N, 1);
        Cube d_gamma = Cube(B, N, 1);
        Cube d_xhat = Cube(B, N, D);
        Cube d_ivar = Cube(B, N, 1);
        Cube d_xmu1 = Cube(B, N, D);
        // Matrix d_b = Matrix(N, 1);
        // Matrix d_gamma = Matrix(N, 1);
        // Matrix d_xhat = Matrix(N, D);
        // Matrix d_ivar = Matrix(N, 1);
        // Matrix d_xmu1 = Matrix(N, D);
        for(int b = 0;b < B; b++) {
            for(int i = 0;i < N; i++) {
                float sum = 0.f;
                float sum2 = 0.f;
                float sum3 = 0.f;
                for(int j = 0; j < D; j++) {
                    float p = grad(b,i,j);
                    sum += p; // dbeta = np.sum(dout, axis=1)
                    sum2 += p * xhat_cache(b, i, j); // dgamma = np.sum(dout * xhat, axis=1)
                    sum3 += p * gamma[j] * xmu_cache(b, i, j); // divar = dxhat * xmu
                    d_xhat(b, i, j) = p * gamma[j]; // dxhat = dgammax * gamma, dgammax = dout
                }
                d_b(b, i, 0) = sum;
                d_gamma(b, i, 0) = sum2;
                d_ivar(b, i, 0) = sum3;
                for(int j = 0;j < D;j++) {
                    d_xmu1(b, i, j) =d_xhat(b, i, j) * sum3; // dxmu1 = dxhat * ivar
                }
            }
        }
        Cube d_sqrtvar = Cube(B, N, 1);
        Cube d_var = Cube(B, N, 1);
        // Matrix d_sqrtvar = Matrix(N, 1);
        // Matrix d_var = Matrix(N, 1);
        for(int b = 0;b < B;b++) {
            for(int i = 0;i < N;i++) {
                d_sqrtvar(b, i, 0) = -1.0 / (sqrtvar_cache(b, i, 0) * sqrtvar_cache(b, i, 0)) * d_ivar(b, i, 0); // dsqrtvar = -1. / (sqrtvar ** 2) * divar
                d_var(b, i, 0) = 0.5 * 1.0 / sqrtvar_cache(b, i, 0) * d_sqrtvar(b, i, 0); // dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar
            }
        }
        Cube dsq = Cube(B, N, D);
        Cube d_xmu2 = Cube(B, N, D);
        // Matrix dsq = Matrix(N, D);
        // Matrix d_xmu2 = Matrix(N, D);
        for(int b = 0;b < B;b++) {
            for(int i = 0;i < N;i++) {
                for(int j = 0;j < D;j++) {
                    float temp = 1.0 * d_var(b, i, 0) / N; // dsq = 1. /N * np.ones((N,D)) * dvar
                    dsq(b, i, j) = temp;
                    d_xmu2(b, i, j) = 2.0 * xmu_cache(b, i, j) * temp + d_xmu1(b, i, j); // dxmu2 = 2 * xmu * dsq, dx1 = dxmu1 + dxmu2, dxmu2 is now essentially dx1
                }
            }
        }   
        for(int b = 0;b < B;b++) {
            for(int i = 0;i < N;i++) {
                float sum = 0.f;
                for(int j = 0;j < D;j++) {
                    sum -= d_xmu2(b, i, j); // dmu = -1 * np.sum(dx1, axis=1)
                }
                sum /= N; // dx2 = 1. /N * np.ones((N,D)) * dmu, and then dx = dx1 + dx2
                for(int j = 0;j < D;j++) {
                    d_xmu2(b, i, j) += sum;
                }
            }
        }
        std::cout << "Layer Norm Backward Completed" << std::endl;
        return d_xmu2;
    }
};

#endif
