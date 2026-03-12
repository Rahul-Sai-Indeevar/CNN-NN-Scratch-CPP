#ifndef FLATTEN_H
#define FLATTEN_H

#include "Layer.h"

class Flatten : public Layer{
    int input_rows, input_cols, input_depth;
public:
    std::vector<Matrix> forward(const std::vector<Matrix>& input) override{
        input_depth = input.size();
        input_rows = input[0].rows;
        input_cols = input[0].cols;

        int total_size = input_depth * input_rows * input_cols;
        Matrix flat(1,total_size);

        int index = 0;
        for(const auto& mat : input){
            for(double val : mat.data){
                flat.data[index++] = val;
            }
        }

        std::vector<Matrix> output;
        output.push_back(flat);
        return output;
    }

    std::vector<Matrix> backward(const std::vector<Matrix>& output_gradient) override{
        std::vector<Matrix> input_gradient;
        const Matrix& grad = output_gradient[0];
        int index = 0;
        for(int d=0;d<input_depth;d++){
            Matrix m(input_rows,input_cols);
            for(int i=0;i<input_rows;i++){
                for(int j=0;j<input_cols;j++){
                    m.at(i,j) = grad.data[index++];
                }
            }
            input_gradient.push_back(m);
        }
        return input_gradient;
    }

    void update (Optimizer* opt) override {}
    std::string getType() const override { return "Flatten"; }
    void save(std::ofstream &file) const override {}
    void load(std::ifstream &file) override {}
};

#endif