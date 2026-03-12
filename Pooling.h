#ifndef POOLING_H
#define POOLING_H

#include "Layer.h"
#include <vector>
#include <cfloat>

enum class PoolType{
    AVG,
    MAX,
    GLOBAL_AVG
};

class Pooling : public Layer
{
    int size;
    int stride;
    PoolType type;
    std::vector<Matrix> input_cache;
    std::vector<Matrix> max_mask;

public:
    Pooling(int pool_size, int pool_stride, PoolType t = PoolType::MAX) : size(pool_size), stride(pool_stride), type(t) {}
    Pooling(PoolType t = PoolType::GLOBAL_AVG) : size(0), stride(0), type(t) {}

    std::vector<Matrix> forward(const std::vector<Matrix> &input) override
    {
        input_cache = input;
        std::vector<Matrix> output;
        if (type == PoolType::MAX)
            max_mask.clear();
        for (const auto &in : input)
        {
            int out_h, out_w, pool_h, pool_w, step;
            if (type == PoolType::GLOBAL_AVG)
            {
                out_h = 1;
                out_w = 1;
                pool_h = in.rows;
                pool_w = in.cols;
                step = 1;
            }
            else
            {
                out_h = (in.rows - size) / stride + 1;
                out_w = (in.cols - size) / stride + 1;
                pool_h = size;
                pool_w = size;
                step = size;
            }
            Matrix out_mat(out_h, out_w);
            Matrix mask(in.rows, in.cols);
            for (int i = 0; i < out_h; i++)
            {
                for (int j = 0; j < out_w; j++)
                {
                    int start_r = i * step;
                    int start_c = j * step;
                    double val = (type == PoolType::MAX) ? -DBL_MAX : 0.0;
                    int max_r = -1, max_c = -1;
                    for (int m = 0; m < pool_h; m++)
                    {
                        for (int n = 0; n < pool_w; n++)
                        {
                            double curr = in.at(start_r + m, start_c + n);
                            if (type == PoolType::MAX)
                            {
                                if (curr > val)
                                {
                                    val = curr;
                                    max_r = start_r + m;
                                    max_c = start_c + n;
                                }
                                else
                                    val += curr;
                            }
                        }
                    }
                    if (type == PoolType::MAX)
                    {
                        out_mat.at(i, j) = val;
                        mask.at(max_r, max_c) = 1.0;
                    }
                    else
                        out_mat.at(i, j) = val / (pool_h * pool_w);
                }
            }
            output.push_back(out_mat);
            if (type == PoolType::MAX)
                max_mask.push_back(mask);
        }
        return output;
    }

    std::vector<Matrix> backward(const std::vector<Matrix> &output_gradient) override
    {
        std::vector<Matrix> input_gradient;
        for (size_t k = 0; k < output_gradient.size(); k++)
        {
            const Matrix &grad = output_gradient[k];
            Matrix in_grad(input_cache[k].rows, input_cache[k].cols);
            int pool_h, pool_w, step;
            if (type == PoolType::GLOBAL_AVG)
            {
                pool_h = input_cache[k].rows;
                pool_w = input_cache[k].cols;
                step = 0;
                double val = grad.at(0, 0) / (pool_h * pool_w);
                for (int i = 0; i < pool_h; i++)
                {
                    for (int j = 0; j < pool_w; j++)
                    {
                        in_grad.at(i, j) = val;
                    }
                }
            }
            else
            {
                pool_h = size, pool_w = size, step = stride;
                for (int i = 0; i < grad.rows; i++)
                {
                    for (int j = 0; j < grad.cols; j++)
                    {
                        double g = grad.at(i, j);
                        int start_r = i * step;
                        int start_c = j * step;
                        if (type == PoolType::MAX)
                        {
                            for (int m = 0; m < pool_h; m++)
                            {
                                for (int n = 0; n < pool_w; n++)
                                {
                                    if (max_mask[k].at(start_r + m, start_c + n) == 1.0)
                                        in_grad.at(start_r + m, start_c + n) += g;
                                }
                            }
                        }
                        else
                        {
                            double dist_g = g / (pool_h * pool_w);
                            for (int m = 0; m < pool_h; m++)
                            {
                                for (int n = 0; n < pool_w; n++)
                                {
                                    in_grad.at(start_r + m, start_c + n) += dist_g;
                                }
                            }
                        }
                    }
                }
            }
            input_gradient.push_back(in_grad);
        }
        return input_gradient;
    }

    void update(Optimizer *opt) override {}

    std::string getType() const override { return "Pooling"; }

    void save(std::ofstream &file) const override
    {
        file.write((char *)&size, sizeof(int));
        file.write((char *)&stride, sizeof(int));
        file.write((char *)&type, sizeof(PoolType));
    }

    void load(std::ifstream &file) override
    {
        file.read((char *)&size, sizeof(int));
        file.read((char *)&stride, sizeof(int));
        file.read((char *)&type, sizeof(PoolType));
    }
};

#endif