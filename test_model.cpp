#include <iostream>
#include "Neural_Network.h"
#include "Mnist_Loader.h"
#include "Dense.h"
#include "Conv2D.h"
#include "Flatten.h"
#include "Pooling.h"
#include "Activation_Layer.h"

int main()
{
    NeuralNetwork nn;
    std::cout << "Loading model...\n";

    // 1. Load the saved model
    nn.load("lenet5_mnist.model");

    // 2. Load one image to test
    std::vector<Matrix> x, y;
    MnistLoader::load("train-images.idx3-ubyte", "train-labels.idx1-ubyte", x, y, 10);

    // 3. Predict
    Matrix pred = nn.feedForward(x[0]);
    std::cout << "Prediction for first image: ";
    pred.print();

    return 0;
}
