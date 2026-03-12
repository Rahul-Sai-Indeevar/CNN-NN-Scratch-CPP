#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <iostream>
#include <vector>
#include <cstdio> // For C-style I/O (fopen, fread)
#include "Matrix.h"

class MnistLoader
{
    // Helper: Flip integer bytes (Big Endian -> Little Endian)
    static uint32_t swap_endian(uint32_t val)
    {
        val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
        return (val << 16) | (val >> 16);
    }

public:
    static void load(const std::string &image_file, const std::string &label_file,
                     std::vector<Matrix> &images, std::vector<Matrix> &labels, int max_items = 0)
    {

        std::cout << "Reading: " << image_file << "..." << std::endl;

        FILE *f_img = fopen(image_file.c_str(), "rb");
        FILE *f_lbl = fopen(label_file.c_str(), "rb");

        if (!f_img)
        {
            std::cerr << "CRITICAL ERROR: Could not open '" << image_file << "'. File not found!" << std::endl;
            if (f_lbl)
                fclose(f_lbl);
            return;
        }
        if (!f_lbl)
        {
            std::cerr << "CRITICAL ERROR: Could not open '" << label_file << "'. File not found!" << std::endl;
            if (f_img)
                fclose(f_img);
            return;
        }

        // 1. Read Headers
        uint32_t magic, num_items, rows, cols;
        uint32_t magic_l, num_labels;

        fread(&magic, 4, 1, f_img);
        magic = swap_endian(magic);
        fread(&num_items, 4, 1, f_img);
        num_items = swap_endian(num_items);
        fread(&rows, 4, 1, f_img);
        rows = swap_endian(rows);
        fread(&cols, 4, 1, f_img);
        cols = swap_endian(cols);

        fread(&magic_l, 4, 1, f_lbl);
        magic_l = swap_endian(magic_l);
        fread(&num_labels, 4, 1, f_lbl);
        num_labels = swap_endian(num_labels);

        std::cout << "--- Header Info ---" << std::endl;
        std::cout << "Magic: " << magic << " (Exp: 2051)" << std::endl;
        std::cout << "Items: " << num_items << std::endl;
        std::cout << "Rows:  " << rows << " Cols: " << cols << std::endl;

        // Validation
        if (magic != 2051 || magic_l != 2049)
        {
            std::cerr << "ERROR: Invalid Magic Number! Files corrupted." << std::endl;
            fclose(f_img);
            fclose(f_lbl);
            return;
        }

        if (max_items > 0)
            num_items = (uint32_t)max_items;

        // 2. Read Data
        std::cout << "Allocating memory..." << std::endl;

        // Buffer for one image (28x28 = 784 bytes)
        std::vector<unsigned char> img_buffer(rows * cols);
        unsigned char lbl_buffer;

        for (uint32_t i = 0; i < num_items; ++i)
        {
            // Read Image pixels
            fread(img_buffer.data(), 1, rows * cols, f_img);

            Matrix img(rows, cols);
            for (uint32_t p = 0; p < rows * cols; ++p)
            {
                img.data[p] = (double)img_buffer[p] / 255.0;
            }
            images.push_back(img);

            // Read Label
            fread(&lbl_buffer, 1, 1, f_lbl);

            Matrix one_hot(1, 10);
            if (lbl_buffer < 10)
                one_hot.at(0, (int)lbl_buffer) = 1.0;
            labels.push_back(one_hot);
        }

        fclose(f_img);
        fclose(f_lbl);
        std::cout << "SUCCESS: Loaded " << images.size() << " images." << std::endl;
    }
};

#endif