#ifndef SEQUENTIAL_H_
#define SEQUENTIAL_H_

#include <cmath>
#include <chrono>
#include <vector>
#include "transformer_cube.h"
#include "cube.h"


class Sequential {
public:
    std::vector<TransformerEncoderLayer_cube> layers;

    Sequential() {
#ifdef DEBUG
            std::cout << "[Sequential constructor]" << std::endl;
#endif
    }

    ~Sequential() {
#ifdef DEBUG
            std::cout << "[Sequential destructor]" << std::endl;
#endif
    }

    Cube forward(const Cube& input) {

        Cube output = input;

        for (auto& layer : layers) {
            output = layer.forward(output);
        }

        return output;
    }

};

#endif