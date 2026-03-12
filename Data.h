#ifndef DATA_H
#define DATA_H

#include "Matrix.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // for std::shuffle
#include <random>

class Data
{
public:
    std::vector<std::vector<double>> raw_data;

    void readCSV(const std::string &filename)
    {
        raw_data.clear(); // Clear old data from memory
        std::ifstream file(filename);
        std::string line;

        while (std::getline(file, line))
        {
            std::stringstream ss(line);
            std::string val_str;
            std::vector<double> row;

            while (std::getline(ss, val_str, ','))
            {
                if (!val_str.empty())
                { // Skip empty strings
                    try
                    {
                        row.push_back(std::stod(val_str));
                    }
                    catch (...)
                    {
                        // Ignore parsing errors (like trailing newlines)
                    }
                }
            }
            if (!row.empty())
                raw_data.push_back(row);
        }
        std::cout << "Loaded " << raw_data.size() << " rows from " << filename << "\n";
    }

    void normalize()
    {
        if (raw_data.empty())
            return;
        int rows = raw_data.size();
        int cols = raw_data[0].size();
        for (int j = 0; j < cols; j++)
        {
            double max_val, min_val = raw_data[0][j];
            for (size_t i = 0; i < rows; i++)
            {
                if (raw_data[i][j] < min_val)
                    min_val = raw_data[i][j];
                if (raw_data[i][j] > max_val)
                    max_val = raw_data[i][j];
            }
            double range = max_val - min_val;
            if (std::abs(range) < 1e-8)
                continue;
            for (size_t i = 0; i < rows; i++)
            {
                raw_data[i][j] = (raw_data[i][j] - min_val) / range;
            }
        }
    }

    void shuffle()
    {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(raw_data.begin(), raw_data.end(), g);
    }

    void getBatch(int start_row, int batch_size, Matrix &input_batch, Matrix &target_batch, int target_col_idx, int num_classes = 1)
    {
        int rows = std::min(int(raw_data.size() - start_row), batch_size);
        int cols = raw_data[0].size();
        if (target_col_idx < 0)
            target_col_idx = cols + target_col_idx;
        int input_cols = cols - 1;

        input_batch = Matrix(rows, input_cols);
        target_batch = Matrix(rows, num_classes);

        for (int i = 0; i < rows; i++)
        {
            int r = start_row + i;
            int c_in = 0;
            for (int c = 0; c < cols; c++)
            {
                if (c == target_col_idx)
                {
                    double val = raw_data[r][c];
                    if (num_classes > 1)
                    {
                        int class_idx = (int)std::round(val * (num_classes - 1));
                        if (class_idx >= 0 && class_idx < num_classes)
                            target_batch.at(i, class_idx) = 1.0;
                        else
                            target_batch.at(i, 0) = val;
                    }
                }
                else
                    input_batch.at(i, c_in++) = raw_data[r][c];
            }
        }
    }
};

#endif