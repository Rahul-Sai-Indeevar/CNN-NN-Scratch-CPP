#include <iostream>
#include <vector>
#include <algorithm> // For std::shuffle
#include <random>    // For std::mt19937

#include "Neural_Network.h"
#include "Conv2D.h"
#include "Pooling.h"
#include "Flatten.h"
#include "Dense.h"
#include "Activation_Layer.h"
#include "Mnist_Loader.h"
#include "Optimizer.h"

int main()
{
    // ---------------------------------------------------------
    // 1. SETUP & DATA LOADING
    // ---------------------------------------------------------
    std::vector<Matrix> train_x, train_y;
    int num_samples = 5000;

    MnistLoader::load("train-images.idx3-ubyte", "train-labels.idx1-ubyte", train_x, train_y, num_samples);

    if (train_x.empty())
    {
        std::cerr << "Data load failed. Exiting.\n";
        return -1;
    }

    // ---------------------------------------------------------
    // 2. BUILD ARCHITECTURE (LeNet-5)
    // ---------------------------------------------------------

    NeuralNetwork nn;
    Adam opt(0.0001);

    std::cout << "Building Tuned LeNet-5...\n";

    // Layer 1: Conv2D
    nn.add(new Conv2D(1, 6, 5, 1, 2));
    nn.add(new ActivationLayer(ActivationType::RELU));

    // TWEAK 1: Use MAX Pooling instead of AVG Pooling
    nn.add(new Pooling(2, 2, PoolType::MAX));

    // Layer 3: Conv2D
    nn.add(new Conv2D(6, 16, 5, 1, 0));
    nn.add(new ActivationLayer(ActivationType::RELU));

    // TWEAK 1: Use MAX Pooling
    nn.add(new Pooling(2, 2, PoolType::MAX));

    // Layer 5: Flatten (16 * 5 * 5 = 400)
    nn.add(new Flatten());

    // Layer 6: Dense (400 -> 120)
    nn.add(new Dense(400, 120, opt));
    nn.add(new ActivationLayer(ActivationType::RELU));

    // Layer 7: Dense (120 -> 84)
    nn.add(new Dense(120, 84, opt));
    nn.add(new ActivationLayer(ActivationType::RELU));

    // Layer 8: Output (84 -> 10)
    nn.add(new Dense(84, 10, opt));
    nn.add(new ActivationLayer(ActivationType::SOFTMAX));

    // ---------------------------------------------------------
    // 3. TRAINING LOOP
    // ---------------------------------------------------------
    int epochs = 20;
    std::cout << "Starting Training (" << epochs << " epochs)...\n";

    // Index vector for shuffling
    std::vector<int> indices(train_x.size());
    for (size_t i = 0; i < indices.size(); i++)
        indices[i] = i;

    for (int e = 0; e < epochs; e++)
    {
        // Shuffle Data
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        double total_loss = 0;
        int correct = 0;

        // Progress Bar Helper
        int print_every = train_x.size() / 10;
        if (print_every == 0)
            print_every = 1;

        for (size_t i = 0; i < train_x.size(); i++)
        {
            int idx = indices[i];

            // 1. Train
            nn.train(train_x[idx], train_y[idx]);

            // 2. Statistics
            Matrix pred = nn.feedForward(train_x[idx]);
            total_loss += nn.calcLoss(train_y[idx], pred, LossType::CROSS_ENTROPY);

            // 3. Accuracy
            int p_max = 0, t_max = 0;
            for (int k = 0; k < 10; k++)
            {
                if (pred.at(0, k) > pred.at(0, p_max))
                    p_max = k;
                if (train_y[idx].at(0, k) > train_y[idx].at(0, t_max))
                    t_max = k;
            }
            if (p_max == t_max)
                correct++;

            if (i % print_every == 0)
                std::cout << ".";
        }

        double avg_loss = total_loss / train_x.size();
        double acc = (double)correct / train_x.size() * 100.0;

        std::cout << "\nEpoch " << e << " | Loss: " << avg_loss << " | Acc: " << acc << "%\n";
    }

    // ---------------------------------------------------------
    // 4. SAVE MODEL
    // ---------------------------------------------------------
    std::cout << "Saving model to lenet5_mnist.model...\n";
    nn.save("lenet5_mnist.model");

    return 0;
}