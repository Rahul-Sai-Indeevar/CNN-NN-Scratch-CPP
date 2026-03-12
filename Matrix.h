#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <random>
#include <cassert>
#include <stdexcept>

class Matrix
{
public:
    int rows, cols;
    std::vector<double> data;
    Matrix() : rows(0), cols(0) {}

    Matrix(int r, int c) : rows(r), cols(c)
    {
        data.resize(r * c, 0.0);
    }

    double &at(int r, int c)
    {
        return data[r * cols + c];
    }

    const double &at(int r, int c) const
    {
        return data[r * cols + c];
    }

    void randomize()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(-1.0, 1.0);
        for (auto &val : data)
            val = dist(gen);
    }

    Matrix add(const Matrix &other) const
    {
        if (rows == other.rows && cols == other.cols)
        {
            Matrix result(rows, cols);
            for (size_t i = 0; i < data.size(); i++)
                result.data[i] = data[i] + other.data[i];
            return result;
        }

        if (other.rows == 1 && cols == other.cols)
        {
            Matrix result(rows, cols);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result.at(i, j) = this->at(i, j) + other.at(0, j);
                }
            }
            return result;
        }
        throw std::invalid_argument("Matrix dimensions incompatible for addition.");
    }

    Matrix subtract(const Matrix &other) const
    {
        assert(rows == other.rows && cols == other.cols && "Dimension mismatch for Subtract");
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); i++)
        {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    Matrix dot(const Matrix &other) const
    {
        assert(cols == other.rows && "Dimension mismatch for Dot Product");
        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; i++)
        {
            for (int k = 0; k < cols; k++)
            {
                double r_val = this->at(i, k);
                for (int j = 0; j < other.cols; j++)
                {
                    result.at(i, j) += r_val * other.at(k, j);
                }
            }
        }
        return result;
    }

    Matrix multiply(const Matrix &other) const
    {
        assert(rows == other.rows && cols == other.cols && "Dimension mismatch for Hadamard");
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); i++)
        {
            result.data[i] = data[i] * other.data[i];
        }
        return result;
    }

    Matrix multiply(double scalar) const
    {
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); i++)
        {
            result.data[i] = scalar * data[i];
        }
        return result;
    }

    Matrix transpose() const
    {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result.at(j, i) = this->at(i, j);
            }
        }
        return result;
    }

    Matrix map(std::function<double(double)> func) const
    {
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); i++)
        {
            result.data[i] = func(data[i]);
        }
        return result;
    }

    void print() const
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                std::cout << at(i, j) << "\t";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    Matrix divide(const Matrix &other) const
    {
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); i++)
        {
            if (other.data[i] < 1e-8)
                result.data[i] = 0.0;
            else
                result.data[i] = data[i] / other.data[i];
        }
        return result;
    }

    Matrix sumAxis0() const
    {
        Matrix result(1, cols);
        for (int j = 0; j < cols; j++)
        {
            double sum = 0.0;
            for (int i = 0; i < rows; i++)
            {
                sum += at(i, j);
            }
            result.at(0, j) = sum;
        }
        return result;
    }

    Matrix square() const
    {
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); i++)
        {
            result.data[i] = data[i] * data[i];
        }
        return result;
    }

    Matrix sqrt() const
    {
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); i++)
        {
            result.data[i] = std::sqrt(data[i]);
        }
        return result;
    }

    Matrix add(double scalar) const
    {
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); i++)
        {
            result.data[i] = data[i] + scalar;
        }
        return result;
    }

    static Matrix correlate(const Matrix &input, const Matrix &kernel)
    {
        int out_h = input.rows - kernel.rows + 1;
        int out_w = input.cols - kernel.cols + 1;
        Matrix output(out_h, out_w);
        for (int i = 0; i < out_h; i++)
        {
            for (int j = 0; j < out_w; j++)
            {
                double sum = 0.0;
                for (int ki = 0; ki < kernel.rows; ki++)
                {
                    for (int kj = 0; kj < kernel.cols; kj++)
                    {
                        sum += input.at(i + ki, j + kj) * kernel.at(ki, kj);
                    }
                }
                output.at(i, j) = sum;
            }
        }
        return output;
    }

    static Matrix convFull(const Matrix &input, const Matrix &kernel)
    {
        int out_h = input.rows + kernel.rows - 1; // Full padding p = k-1
        int out_w = input.cols + kernel.cols - 1;
        Matrix output(out_h, out_w);
        for (int i = 0; i < out_h; i++)
        {
            for (int j = 0; j < out_w; j++)
            {
                double sum = 0.0;
                for (int ki = 0; ki < kernel.rows; ki++)
                {
                    for (int kj = 0; kj < kernel.cols; kj++)
                    {
                        int r = i - ki, c = j - kj;
                        if (r >= 0 && r < input.rows && c >= 0 && c < input.cols)
                        {
                            sum += input.at(r, c) * kernel.at(ki, kj);
                        }
                    }
                }
                output.at(i, j) = sum;
            }
        }
        return output;
    }

    Matrix rotate180() const
    {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result.at(rows - i - 1, cols - j - 1) = at(i, j);
            }
        }
        return result;
    }

    Matrix pad(int p) const
    {
        if (p == 0)
            return *this;
        Matrix res(rows + 2 * p, cols + 2 * p);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                res.at(i + p, j + p) = at(i, j);
            }
        }
        return res;
    }

    Matrix crop(int p) const
    {
        if (p == 0)
            return *this;
        Matrix res(rows - 2 * p, cols - 2 * p);
        for (int i = 0; i < res.rows; i++)
        {
            for (int j = 0; j < res.cols; j++)
            {
                res.at(i, j) = at(i + p, j + p);
            }
        }
        return res;
    }

    Matrix dilate(int stride) const
    {
        if (stride == 1)
            return *this;
        int new_rows = (rows - 1) * stride + 1;
        int new_cols = (cols - 1) * stride + 1;
        Matrix res(new_rows, new_cols);
        for (int i = 0; i < rows; i++){
            for (int j = 0; j < cols; j++){
                res.at(i * stride, j * stride) = at(i, j);
            }
        }
        return res;
    }

    void save(std::ofstream &file) const
    {
        file.write((char *)&rows, sizeof(int));
        file.write((char *)&cols, sizeof(int));
        file.write((char *)data.data(), data.size() * sizeof(double));
    }

    void load(std::ifstream &file)
    {
        file.read((char *)&rows, sizeof(int));
        file.read((char *)&cols, sizeof(int));
        data.resize(rows * cols);
        file.read((char *)data.data(), data.size() * sizeof(double));
    }
};

#endif