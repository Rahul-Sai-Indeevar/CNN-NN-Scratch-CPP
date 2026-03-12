#ifndef LAYER_H
#define LAYER_H

#include "Matrix.h"
#include "Optimizer.h"
#include <vector>
#include <string>
#include <fstream>

class Layer{
public:
    virtual ~Layer() = default;
    virtual std::vector<Matrix> forward(const std::vector<Matrix> &input) = 0;
    virtual std::vector<Matrix> backward(const std::vector<Matrix> &output_gradient) = 0;
    virtual void update(Optimizer *opt) = 0;

    virtual std::string getType() const = 0;
    virtual void save(std::ofstream &file) const = 0;
    virtual void load(std::ifstream &file) = 0;
};
#endif
