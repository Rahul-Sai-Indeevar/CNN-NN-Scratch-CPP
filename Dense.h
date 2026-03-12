#ifndef DENSE_H
#define DENSE_H

#include "Layer.h"
#include <memory>

class Dense : public Layer
{
private:
    Matrix weights;
    Matrix bias;
    std::unique_ptr<Optimizer> opt_w;
    std::unique_ptr<Optimizer> opt_b;
    Matrix input_cache;
    Matrix grad_weights;
    Matrix grad_bias;

public:
    Dense(int input_size, int output_size, const Optimizer &opt_proto) : weights(input_size, output_size), bias(1, output_size), grad_weights(0, 0), grad_bias(0, 0)
    {
        weights.randomize();
        double scale = std::sqrt(2.0 / input_size); // He/Xavier Init
        for (double &v : weights.data)  v *= scale;
        for (double &v : bias.data) v = 0.01;
        opt_w = opt_proto.clone();
        opt_b = opt_proto.clone();
    }

    std::vector<Matrix> forward(const std::vector<Matrix> &input) override
    {
        input_cache = input[0];
        Matrix output = input_cache.dot(weights);
        output = output.add(bias);
        std::vector<Matrix> output_vec;
        output_vec.push_back(output);
        return output_vec;
    }

    std::vector<Matrix> backward(const std::vector<Matrix> &output_gradient) override
    {
        Matrix grad_output = output_gradient[0];
        Matrix input_T = input_cache.transpose();

        // Calculate gradients
        grad_weights = input_T.dot(grad_output);
        grad_bias = grad_output.sumAxis0();

        Matrix weights_T = weights.transpose();
        Matrix grad_input = grad_output.dot(weights_T);

        weights = opt_w->update(weights, grad_weights);
        bias = opt_b->update(bias, grad_bias);

        std::vector<Matrix> output_vec;
        output_vec.push_back(grad_input);
        return output_vec;
    }

    void update(Optimizer *opt) override{ }

    std::string getType() const override { return "Dense"; }

    void save(std::ofstream &file) const override
    {
        weights.save(file);
        bias.save(file);
    }

    void load(std::ifstream &file) override
    {
        weights.load(file);
        bias.load(file);
    }
};

#endif