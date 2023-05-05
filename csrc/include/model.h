#ifndef MODEL_H_
#define MODEL_H_

#include "cube.h"

class Model {
public:
    virtual ~Model() {}
    virtual Cube forward(const Cube& input) = 0;
};

#endif // MODEL_H_