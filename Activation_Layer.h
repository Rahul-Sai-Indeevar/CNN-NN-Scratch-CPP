#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include "Layer.h"
#include <functional>

enum class ActivationType
{
    RELU,
    SIGMOID,
    SOFTMAX
};

class ActivationLayer : public Layer
{
private:
    ActivationType type;
    std::vector<Matrix> input_cache;

public:
    ActivationLayer(ActivationType t) : type(t) {}

    static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
    static double sigmoidPrime(double x)
    {
        double s = sigmoid(x);
        return s * (1.0 - s);
    }
    static double relu(double x) { return (x > 0) ? x : 0.0; }
    static double reluPrime(double x) { return (x > 0) ? 1.0 : 0.0; }

    std::vector<Matrix> forward(const std::vector<Matrix> &input) override
    {
        input_cache = input;
        std::vector<Matrix> output;
        for (const auto &mat : input)
        {
            if (type == ActivationType::SOFTMAX)
            {
                Matrix res(mat.rows, mat.cols);
                for (int i = 0; i < mat.rows; i++)
                {
                    double max_val = -1e9;
                    for (int j = 0; j < mat.cols; j++)
                    {
                        if (mat.at(i, j) > max_val)
                            max_val = mat.at(i, j);
                    }
                    double sum = 0.0;
                    for (int j = 0; j < mat.cols; j++)
                    {
                        res.at(i, j) = std::exp(mat.at(i, j) - max_val);
                        sum += res.at(i, j);
                    }
                    for (int j = 0; j < mat.cols; j++)
                    {
                        res.at(i, j) /= sum;
                    }
                }
                output.push_back(res);
            }
            else
            {
                Matrix res = mat;
                std::function<double(double)> func;
                if (type == ActivationType::SIGMOID)
                    func = sigmoid;
                else if (type == ActivationType::RELU)
                    func = relu;
                output.push_back(res.map(func));
            }
        }
        return output;
    }

    std::vector<Matrix> backward(const std::vector<Matrix> &output_gradient) override
    {
        std::vector<Matrix> input_gradient;
        for (size_t i = 0; i < output_gradient.size(); i++)
        {
            const Matrix &grad = output_gradient[i];
            const Matrix &input = input_cache[i];
            if (type == ActivationType::SOFTMAX)
                input_gradient.push_back(grad);
            else
            {
                std::function<double(double)> func;
                if (type == ActivationType::SIGMOID)
                    func = sigmoidPrime;
                else if (type == ActivationType::RELU)
                    func = reluPrime;
                Matrix derv = input.map(func);
                input_gradient.push_back(grad.multiply(derv));
            }
        }
        return input_gradient;
    }

    void update(Optimizer *opt) override {}

    std::string getType() const override { return "Activation"; }
    void save(std::ofstream &file) const override
    {
        file.write((char *)&type, sizeof(ActivationType));
    }
    void load(std::ifstream &file) override
    {
        file.read((char *)&type, sizeof(ActivationType));
    }
};

#endif