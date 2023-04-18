#include <cmath>

#include "../include/layernorm.h"

LayerNorm::LayerNorm(int hidden_size, float epsilon) {
    hidden_size = hidden_size;
    epsilon = epsilon; // 1e-5 would be a good default value
    gamma = new float[hidden_size];
    beta = new float[hidden_size];
    // initialize gamma and beta to ones and zeros, respectively
    for (int i = 0; i < hidden_size; i++) {
        gamma[i] = 1.0;
        beta[i] = 0.0;
    }

    std::cout << "LayerNorm::LayerNorm() " << hidden_size << std::endl;
}

LayerNorm::~LayerNorm() {
    delete[] gamma;
    delete[] beta;
}

Matrix LayerNorm::forward(Matrix input) {
    x_cache = input;
    xmu_cache = Matrix(input.rows, input.cols);
    var_cache = Matrix(input.rows, 1);
    ivar_cache = Matrix(input.rows, 1);
    sqrtvar_cache = Matrix(input.rows, 1);
    xhat_cache = Matrix(input.rows, input.cols);
    Matrix output(input.rows, input.cols);
    // compute mean and variance of input for each sample in the batch
    for (int i = 0; i < output.rows; i++) {
        float mean = 0.0;
        float var = 0.0;
        for (int j = 0; j < output.cols; j++) {
            mean += input(i,j);
        }
        mean /= output.cols;
        for (int j = 0; j < output.cols; j++)
        {
            float temp = input(i,j) - mean;
            xmu_cache(i,j) = temp;
            var += pow(temp, 2);
        }
        var /= output.cols;
        var_cache(i, 0) = var;
        float sqrtvar = sqrt(var + epsilon);
        sqrtvar_cache(i, 0) = sqrtvar;
        ivar_cache(i, 0) = 1.0 / sqrtvar;
        for (int j = 0; j < output.cols; j++) {
            xhat_cache(i,j) = xmu_cache(i,j) / sqrtvar; // xhat = xmu * ivar
            output(i,j) = xhat_cache(i,j) * gamma[j] + beta[j];
        }  
    }
    return output;
}

/**
 * Reference: https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
 * Adjusted for Layer Normalization
*/
Matrix LayerNorm::backward(Matrix grad) {
    // Compute the gradient w.r.t. the mean and variance
    std::cout << "====================Layer Norm Backward=====================" << std::endl;
    std::cout << "grad shape: " << grad.rows << " " << grad.cols << std::endl;
    std::cout << "x_cache shape: " << x_cache.rows << " " << x_cache.cols << std::endl;
    std::cout << "var_cache shape: " << var_cache.rows << " " << var_cache.cols << std::endl;
    int N = grad.rows;
    int D = grad.cols;

    Matrix d_b = Matrix(N, 1);
    Matrix d_gamma = Matrix(N, 1);
    Matrix d_xhat = Matrix(N, D);
    Matrix d_ivar = Matrix(N, 1);
    Matrix d_xmu1 = Matrix(N, D);
    for(int i = 0;i < N; i++) {
        float sum = 0.f;
        float sum2 = 0.f;
        float sum3 = 0.f;
        for(int j = 0; j < D; j++) {
            sum += grad(i,j); // dbeta = np.sum(dout, axis=1)
            sum2 += grad(i,j) * xhat_cache(i,j); // dgamma = np.sum(dout * xhat, axis=1)
            sum3 += grad(i,j) * gamma[j] * xmu_cache(i, j); // divar = dxhat * xmu
            d_xhat(i,j) = grad(i,j) * gamma[j]; // dxhat = dgammax * gamma, dgammax = dout
        }
        d_b(i,0) = sum;
        d_gamma(i,0) = sum2;
        d_ivar(i,0) = sum3;
        for(int j = 0;j < D;j++) {
            d_xmu1(i,j) =d_xhat(i, j) * sum3; // dxmu1 = dxhat * ivar
        }
    }

    Matrix d_sqrtvar = Matrix(N, 1);
    Matrix d_var = Matrix(N, 1);
    for(int i = 0;i < N;i++) {
        d_sqrtvar(i,0) = -1.0 / (sqrtvar_cache(i,0) * sqrtvar_cache(i,0)) * d_ivar(i,0); // dsqrtvar = -1. / (sqrtvar ** 2) * divar
        d_var(i,0) = 0.5 * 1.0 / sqrtvar_cache(i,0) * d_sqrtvar(i,0); // dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar
    }

    Matrix dsq = Matrix(N, D);
    Matrix d_xmu2 = Matrix(N, D);
    for(int i = 0;i < N;i++) {
        for(int j = 0;j < D;j++) {
            float temp = 1.0 * d_var(i,0) / N; // dsq = 1. /N * np.ones((N,D)) * dvar
            dsq(i,j) = temp;
            d_xmu2(i,j) = 2.0 * xmu_cache(i,j) * temp + d_xmu1(i, j); // dxmu2 = 2 * xmu * dsq, dx1 = dxmu1 + dxmu2, dxmu2 is now essentially dx1
        }
    }

    for(int i = 0;i < N;i++) {
        float sum = 0.f;
        for(int j = 0;j < D;j++) {
            sum -= d_xmu2(i,j); // dmu = -1 * np.sum(dx1, axis=1)
        }
        sum /= N; // dx2 = 1. /N * np.ones((N,D)) * dmu, and then dx = dx1 + dx2
        for(int j = 0;j < D;j++) {
            d_xmu2(i,j) += sum;
        }
    }

    return d_xmu2;


    // Matrix x_centered = x_cache - mean_cache;
    // Matrix x_centered_sum(x_centered.rows, x_centered.cols);
    // Matrix x_centered_pow = Matrix(x_centered.rows, x_centered.cols);
    // Matrix var_cache_pow = Matrix(var_cache.rows, var_cache.cols);
    // Matrix var_cache_sqrt_negative = Matrix(var_cache.rows, var_cache.cols);
    // Matrix var_cache_sqrt_positive = Matrix(var_cache.rows, var_cache.cols);
    // for(int i = 0; i < x_centered_sum.rows; i++) {
    //     float sum1 = 0.f;
    //     float sum2 = 0.f;
    //     for(int j = 0; j < x_centered_sum.cols; j++) {
    //         sum1 += x_centered(i,j);
    //         sum2 += x_centered(i,j) * x_centered(i,j) * (-0.5);
    //         var_cache_pow(i,j) = pow(var_cache(i,j) + epsilon, -1.5);
    //         var_cache_sqrt_negative(i,j) = -1.0 / sqrt(var_cache(i,j) + epsilon);
    //         var_cache_sqrt_positive(i,j) = 1.0 / sqrt(var_cache(i,j) + epsilon);
    //     }
    //     for(int j = 0; j < x_centered_sum.cols; j++) {
    //         x_centered_sum(i,j) = sum1;
    //         x_centered_pow(i,j) = sum2;
    //     }
    // }

    // Matrix grad_var = grad * x_centered_pow * var_cache_pow;
    // std::cout << "grad_var shape: " << grad_var.rows << " " << grad_var.cols << std::endl;
    // Matrix grad_var_shift_positive(grad_var.rows, grad_var.cols);
    // Matrix grad_var_shift_negative(grad_var.rows, grad_var.cols);
    // float seq_length = x_cache.rows;
    // for(int i = 0; i < grad_var_shift_positive.rows; i++) {
    //     for(int j = 0; j < grad_var_shift_positive.cols; j++) {
    //         grad_var_shift_negative(i,j) = (-2.0 / seq_length) * grad_var(i,j);
    //         grad_var_shift_positive(i,j) = (2.0 / seq_length) * grad_var(i,j);
    //     }
    // }
    // Matrix grad_mean = grad * var_cache_sqrt_negative;
    // for(int i = 0; i < grad_mean.rows; i++) {
    //     float sum = 0.f;
    //     for(int j = 0; j < grad_mean.cols; j++) {
    //         sum += grad_mean(i,j);
    //     }
    //     for(int j = 0; j < grad_mean.cols; j++) {
    //         grad_mean(i,j) = sum;
    //     }
    // }
    // grad_mean = grad_mean + grad_var_shift_negative * x_centered_sum;

    // // Compute the gradient w.r.t. the input
    // Matrix gamma_broadcast(mean_cache.rows, mean_cache.cols);
    // for(int i = 0;i < gamma_broadcast.rows; i++) {
    //     for(int j = 0; j < gamma_broadcast.cols; j++) {
    //         gamma_broadcast(i,j) = gamma[j];
    //     }
    // }
    // Matrix dX = grad * gamma_broadcast * var_cache_sqrt_positive + grad_mean + grad_var_shift_positive * x_centered_sum;
    // std::cout << "dX shape: " << dX.rows << " " << dX.cols << std::endl;
    // return dX;
}