#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>

#include "Layer.h"
#include "Matrix.h"
#include "Dense.h"
#include "Conv2D.h"
#include "Flatten.h"
#include "Pooling.h"
#include "Activation_Layer.h"

enum class LossType
{
    MSE,
    CROSS_ENTROPY
};

class NeuralNetwork
{
private:
    std::vector<Layer *> layers;

public:
    ~NeuralNetwork()
    {
        for (auto layer : layers)
        {
            delete layer;
        }
    }

    void add(Layer *layer)
    {
        layers.push_back(layer);
    }

    Matrix feedForward(Matrix input)
    {
        std::vector<Matrix> output;
        output.push_back(input);
        for (auto layer : layers)
        {
            output = layer->forward(output);
        }
        return output[0];
    }

    double calcLoss(const Matrix &output, const Matrix &target, LossType type)
    {
        double total_loss = 0.0;
        int N = target.rows * target.cols;
        if (type == LossType::MSE)
        {
            for (int i = 0; i < N; i++)
            {
                double diff = target.data[i] - output.data[i];
                total_loss += diff * diff;
            }
            return total_loss / N;
        }
        else if (type == LossType::CROSS_ENTROPY)
        {
            for (int i = 0; i < N; i++)
            {
                double pred = std::max(1e-15, std::min(1.0 - 1e-15, output.data[i]));
                total_loss += target.data[i] * std::log(pred);
            }
            return -total_loss / target.rows;
        }
        return 0.0;
    }

    void train(Matrix input, Matrix target){
        // 1. Forward
        std::vector<Matrix> activation;
        activation.push_back(input);

        for (auto layer : layers)
        {
            activation = layer->forward(activation);
        }

        // 2. Error Calculation
        Matrix final_output = activation[0];
        Matrix error_gradient = target.subtract(final_output);

        std::vector<Matrix> gradient;
        gradient.push_back(error_gradient);

        // 3. Backward
        for (int i = layers.size() - 1; i >= 0; i--)
        {
            gradient = layers[i]->backward(gradient);
        }
        // Note: Updates happen INSIDE backward() for Conv2D/Dense.
    }

    void save(const std::string &filename)
    {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "Error saving model to " << filename << "\n";
            return;
        }

        // 1. Write number of layers
        int num_layers = layers.size();
        file.write((char *)&num_layers, sizeof(int));

        // 2. Loop and save each layer
        for (auto layer : layers)
        {
            std::string type = layer->getType();
            int type_len = type.length();
            file.write((char *)&type_len, sizeof(int));
            file.write(type.c_str(), type_len);
            layer->save(file);
        }

        file.close();
        std::cout << "Model saved to " << filename << "\n";
    }

    // Load a model from file
    void load(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "Error loading model from " << filename << "\n";
            return;
        }

        // Clear existing architecture
        for (auto l : layers)
            delete l;
        layers.clear();

        int num_layers;
        file.read((char *)&num_layers, sizeof(int));

        for (int i = 0; i < num_layers; i++)
        {
            // Read Type String
            int type_len;
            file.read((char *)&type_len, sizeof(int));
            std::string type(type_len, ' ');
            file.read(&type[0], type_len);

            Layer *layer = nullptr;

            if (type == "Dense")
            {
                // Pass dummy values, load() will overwrite weights/biases
                layer = new Dense(1, 1, SGD());
            }
            else if (type == "Conv2D")
            {
                layer = new Conv2D(1, 1, 1);
            }
            else if (type == "Flatten")
            {
                layer = new Flatten();
            }
            else if (type == "Pooling")
            {
                layer = new Pooling(1, 1);
            }
            else if (type == "Activation")
            {
                layer = new ActivationLayer(ActivationType::SIGMOID);
            }

            if (layer)
            {
                layer->load(file); // Load actual weights/config
                add(layer);
            }
            else
            {
                std::cerr << "Unknown layer type: " << type << "\n";
            }
        }
        file.close();
        std::cout << "Model loaded from " << filename << "\n";
    }
};

#endif