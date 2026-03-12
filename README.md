# Deep Learning Framework from Scratch (C++)

![C++](https://img.shields.io/badge/C++-17-blue.svg)
![Dependencies](https://img.shields.io/badge/Dependencies-None-brightgreen.svg)
![OpenMP](https://img.shields.io/badge/Optimization-OpenMP-orange.svg)

A high-performance, object-oriented Deep Learning framework built entirely from scratch in C++. This project does **not** rely on any external ML libraries (like TensorFlow, PyTorch, or Eigen). It implements backpropagation, multi-dimensional tensor convolutions, and gradient descent algorithms purely from first principles.

## 🚀 Features

* **Modular Architecture:** Designed with OOP principles (Polymorphism/Inheritance) allowing layers to be stacked dynamically (e.g., `nn.add(new Conv2D(...))`).
* **Computer Vision:** Full support for Convolutional Neural Networks (CNNs), including `Conv2D`, `MaxPooling`, `AveragePooling`, and `Flatten` layers.
* **Math Kernel:** Custom Linear Algebra engine supporting Matrix Broadcasting, Cross-Correlation, and Strided Tensors.
* **Optimizers:** Custom implementations of **Adam**, **RMSProp**, and **Momentum** optimizations.
* **Performance:** Leverages **OpenMP** for multi-threaded CPU matrix operations, drastically reducing training time.
* **Persistence:** Polymorphic serialization system to save and load trained models to binary files.

## 🧠 Model Benchmark: LeNet-5 on MNIST

To prove the framework's mathematical accuracy, I recreated the classic **LeNet-5** architecture to classify the MNIST handwritten digit dataset. 

**Architecture:**
1. `Conv2D` (6 filters, 5x5, padding: 2) + `ReLU`
2. `MaxPooling` (2x2)
3. `Conv2D` (16 filters, 5x5) + `ReLU`
4. `MaxPooling` (2x2)
5. `Flatten`
6. `Dense` (120 nodes) + `ReLU`
7. `Dense` (84 nodes) + `ReLU`
8. `Dense` (10 nodes) + `Softmax`

**Results:**
The custom engine successfully trained the network, avoiding vanishing gradients and dead ReLUs via He/Xavier initialization, reaching **~94.6% Accuracy** in just 20 epochs.

*(Insert your screenshot of the terminal output here)*

## 🛠️ How to Build and Run

**1. Clone the repository**
```bash
git clone https://github.com/YourUsername/DeepLearning-CPP-Scratch.git
cd DeepLearning-CPP-Scratch
```
**2. Download MNIST Dataset**
Download the binary dataset files from Yann [LeCun's website](http://yann.lecun.com/exdb/mnist/) and extract them into the root directory:
- train-images.idx3-ubyte
- train-labels.idx1-ubyte

**3. Compile with GCC & OpenMP**
```bash
g++ main.cpp -o mnist -fopenmp -O3
```
**4. Run**
```bash
./mnist
```
