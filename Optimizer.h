#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Matrix.h"
#include <memory>
#include <cmath>

enum class OptimizerType{
    SGD,
    MOMENTUM,
    RMSPROP,
    ADAM
};

class Optimizer{
public:
    virtual ~Optimizer() = default;
    virtual Matrix update(const Matrix &weights, const Matrix &gradients) = 0;
    virtual std::unique_ptr<Optimizer> clone() const = 0;
};

class SGD : public Optimizer{
    double learning_rate;

public:
    SGD(double lr = 0.01) : learning_rate(lr) {}
    Matrix update(const Matrix &weights, const Matrix &gradients) override
    {
        Matrix delta = gradients.multiply(learning_rate);
        return weights.add(delta);
    }
    std::unique_ptr<Optimizer> clone() const override
    {
        return std::make_unique<SGD>(*this);
    }
};

class Momentum : public Optimizer
{
    double learning_rate;
    double beta;
    Matrix velocity;
    bool initialized = false;

public:
    Momentum(double lr = 0.01, double b = 0.9) : learning_rate(lr), beta(b), velocity(0, 0) {}
    Matrix update(const Matrix &gradients, const Matrix &weights) override
    {
        if (!initialized)
        {
            velocity = Matrix(weights.rows, weights.cols);
            initialized = true;
        }
        Matrix temp1 = velocity.multiply(beta);
        Matrix temp2 = gradients.multiply(learning_rate);
        velocity = temp1.add(temp2);
        return weights.add(velocity);
    }
    std::unique_ptr<Optimizer> clone() const override
    {
        return std::make_unique<Momentum>(*this);
    }
};

class AdaGrad : public Optimizer
{
    double learning_rate;
    double epsilon = 1e-8;
    Matrix G;
    bool initialized = false;

public:
    AdaGrad(double lr = 0.01) : learning_rate(lr) {}
    Matrix update(const Matrix &weights, const Matrix &gradients) override
    {
        if (!initialized)
        {
            G = Matrix(weights.rows, weights.cols);
            initialized = true;
        }
        G = G.add(gradients.square());
        Matrix denominator = G.sqrt().add(epsilon);
        Matrix update_term = gradients.divide(denominator).multiply(learning_rate);
        return weights.add(update_term);
    }
    std::unique_ptr<Optimizer> clone() const override
    {
        return std::make_unique<AdaGrad>(*this);
    }
};

class RMSProp : public Optimizer
{
    double learning_rate;
    double beta;
    double epsilon = 1e-8;
    Matrix G;
    bool initialized = false;

public:
    RMSProp(double lr = 0.01, double b = 0.9) : learning_rate(lr), beta(b) {}
    Matrix update(const Matrix &weights, const Matrix &gradients) override
    {
        if (!initialized)
        {
            G = Matrix(weights.rows, weights.cols);
            initialized = true;
        }
        Matrix g_sqr = gradients.square();
        Matrix temp1 = G.multiply(beta);
        Matrix temp2 = g_sqr.multiply(1.0 - beta);
        G = temp1.add(temp2);
        Matrix denominator = G.sqrt().add(epsilon);
        Matrix update_term = gradients.divide(denominator).multiply(learning_rate);
        return weights.add(update_term);
    }
    std::unique_ptr<Optimizer> clone() const override
    {
        return std::make_unique<RMSProp>(*this);
    }
};

class Adam : public Optimizer
{
    double learning_rate;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    int t = 0;
    Matrix m, v;
    bool initialized = false;

public:
    Adam(double lr = 0.01) : learning_rate(lr), m(0, 0), v(0, 0) {}
    Matrix update(const Matrix &weights, const Matrix &gradients)
    {
        if (!initialized)
        {
            m = Matrix(weights.rows, weights.cols);
            v = Matrix(weights.rows, weights.cols);
            initialized = true;
        }
        t++;

        Matrix m_temp1 = m.multiply(beta1);
        Matrix m_temp2 = gradients.multiply(1.0 - beta1);
        m = m_temp1.add(m_temp2);

        Matrix g_sqr = gradients.square();
        Matrix v_temp1 = v.multiply(beta2);
        Matrix v_temp2 = g_sqr.multiply(1.0 - beta2);
        v = v_temp1.add(v_temp2);

        double beta1_t = std::pow(beta1, t);
        Matrix m_hat = m.multiply(1.0 / (1.0 - beta1_t));

        double beta2_t = std::pow(beta2, t);
        Matrix v_hat = v.multiply(1.0 / (1.0 - beta2_t));
        Matrix v_hat_sqrt = v_hat.sqrt().add(epsilon);

        Matrix update_term = m_hat.divide(v_hat_sqrt).multiply(learning_rate);
        return weights.add(update_term);
    }
    std::unique_ptr<Optimizer> clone() const override
    {
        return std::make_unique<Adam>(*this);
    }
};

#endif