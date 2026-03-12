#ifndef CONV2D_H
#define CONV2D_H

#include "Layer.h"
#include <vector>
#include <cmath>
#include <omp.h>

class Conv2D : public Layer
{
    int input_depth, num_filters, kernel_size, stride, padding;

    std::vector<std::vector<Matrix>> kernels;
    std::vector<Matrix> biases;
    std::vector<Matrix> input_cache;

    // Gradients
    std::vector<std::vector<Matrix>> kernel_grads;
    std::vector<Matrix> bias_grads;

    // Optimizers
    std::vector<std::vector<std::unique_ptr<Optimizer>>> opt_kernels;
    std::vector<std::unique_ptr<Optimizer>> opt_biases;
    bool optimizers_initialized = false;

public:
    // Inside Conv2D.h

    Conv2D(int in_depth, int n_filters, int k_size, int s = 1, int p = 0)
        : input_depth(in_depth), num_filters(n_filters), kernel_size(k_size), stride(s), padding(p)
    {
        kernels.resize(num_filters);
        kernel_grads.resize(num_filters);

        // AGGRESSIVE SCALING:
        // Standard Xavier is sqrt(2 / n). We multiply by 0.1 extra safety factor.
        double scale = 0.1 * std::sqrt(2.0 / (kernel_size * kernel_size * input_depth));

        for (int i = 0; i < num_filters; i++)
        {
            kernels[i].resize(input_depth);
            kernel_grads[i].resize(input_depth);

            // Init bias to 0
            biases.push_back(Matrix(1, 1));
            bias_grads.push_back(Matrix(1, 1));

            for (int j = 0; j < input_depth; j++)
            {
                kernels[i][j] = Matrix(kernel_size, kernel_size);
                kernels[i][j].randomize(); // -1 to 1

                // Apply Scale
                for (double &v : kernels[i][j].data) v *= scale;
            }
        }
        std::cout << "Conv2D Init (" << n_filters << "x" << k_size << "x" << k_size << ") Sample Weight: " << kernels[0][0].at(0, 0) << std::endl;
    }

    std::vector<Matrix> forward(const std::vector<Matrix> &input) override
    {
        input_cache.clear();
        std::vector<Matrix> output(num_filters);

        // Cache padded input
        for (const auto &mat : input) input_cache.push_back(mat.pad(padding));

#pragma omp parallel for
        for (int f = 0; f < num_filters; f++){
            Matrix sum_mat(0, 0);
            bool first = true;

            for (int c = 0; c < input_depth; c++)
            {
                Matrix corr = Matrix::correlate(input_cache[c], kernels[f][c]);
                if (first)
                {
                    sum_mat = corr;
                    first = false;
                }
                else
                    sum_mat = sum_mat.add(corr);
            }
            output[f] = sum_mat.add(biases[f].at(0, 0));
        }
        return output;
    }

    std::vector<Matrix> backward(const std::vector<Matrix> &output_gradient) override
    {
        std::vector<Matrix> input_gradient(input_depth);

        // Initialize with zeros
        for (int c = 0; c < input_depth; c++)
            input_gradient[c] = Matrix(input_cache[0].rows, input_cache[0].cols);

#pragma omp parallel for
        for (int f = 0; f < num_filters; f++)
        {
            // Handle Stride
            Matrix dilated_grad = (stride > 1) ? output_gradient[f].dilate(stride) : output_gradient[f];

            // 1. Bias Gradient
            double b_sum = 0;
            for (double v : dilated_grad.data)
                b_sum += v;
            bias_grads[f].at(0, 0) = b_sum;

            for (int c = 0; c < input_depth; c++)
            {
                // 2. Kernel Gradient (Valid Correlation)
                kernel_grads[f][c] = Matrix::correlate(input_cache[c], dilated_grad);

                // 3. Input Gradient (Full Convolution)
                Matrix rot_kernel = kernels[f][c].rotate180();
                Matrix dX = Matrix::convFull(dilated_grad, rot_kernel);

#pragma omp critical
                {
                    input_gradient[c] = input_gradient[c].add(dX);
                }
            }
        }

        // Crop padding to return to original size
        std::vector<Matrix> final_grads;
        for (const auto &g : input_gradient)
        {
            final_grads.push_back(g.crop(padding));
        }
        return final_grads;
    }

    void update(Optimizer *opt) override
    {
        if (!optimizers_initialized)
        {
            opt_kernels.resize(num_filters);
            opt_biases.reserve(num_filters);
            for (int f = 0; f < num_filters; f++)
            {
                opt_biases.push_back(opt->clone());
                opt_kernels[f].resize(input_depth);
                for (int c = 0; c < input_depth; c++)
                {
                    opt_kernels[f][c] = opt->clone();
                }
            }
            optimizers_initialized = true;
        }

#pragma omp parallel for
        for (int f = 0; f < num_filters; f++)
        {
            biases[f] = opt_biases[f]->update(biases[f], bias_grads[f]);
            for (int c = 0; c < input_depth; c++)
            {
                kernels[f][c] = opt_kernels[f][c]->update(kernels[f][c], kernel_grads[f][c]);
            }
        }
    }

    std::string getType() const override { return "Conv2D"; }

    void save(std::ofstream &file) const override
    {
        // 1. Save Config
        file.write((char *)&input_depth, sizeof(int));
        file.write((char *)&num_filters, sizeof(int));
        file.write((char *)&kernel_size, sizeof(int));
        file.write((char *)&stride, sizeof(int));
        file.write((char *)&padding, sizeof(int));

        // 2. Save Weights
        for (int f = 0; f < num_filters; f++)
        {
            biases[f].save(file);
            for (int c = 0; c < input_depth; c++)
            {
                kernels[f][c].save(file);
            }
        }
    }

    void load(std::ifstream &file) override
    {
        // 1. Load Config
        file.read((char *)&input_depth, sizeof(int));
        file.read((char *)&num_filters, sizeof(int));
        file.read((char *)&kernel_size, sizeof(int));
        file.read((char *)&stride, sizeof(int));
        file.read((char *)&padding, sizeof(int));

        // 2. Resize structures
        kernels.assign(num_filters, std::vector<Matrix>(input_depth, Matrix(1, 1)));
        biases.assign(num_filters, Matrix(1, 1));

        // Resize Gradients & Optimizers
        kernel_grads.assign(num_filters, std::vector<Matrix>(input_depth, Matrix(1, 1)));
        bias_grads.assign(num_filters, Matrix(1, 1));
        optimizers_initialized = false; // Force re-init of optimizers

        // 3. Load Weights
        for (int f = 0; f < num_filters; f++)
        {
            biases[f].load(file);
            for (int c = 0; c < input_depth; c++)
            {
                kernels[f][c].load(file);
            }
        }
    }
};

#endif