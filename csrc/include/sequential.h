#ifndef SEQUENTIAL_H_
#define SEQUENTIAL_H_

#include "model.h"
#include <cmath>
#include <chrono>
#include <vector>
#include "cube.h"


class Sequential {
public:
    std::vector<Model*> layers;

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
            output = layer->forward(output);
        }

        return output;
    }

};

#endif