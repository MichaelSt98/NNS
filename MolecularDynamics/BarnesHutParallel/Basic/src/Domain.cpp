//
// Created by Michael Staneker on 15.03.21.
//

#include "../include/Domain.h"

float getSystemSize(Box *b) {
    float systemSize = 0.0;
    float temp;
    for (int i=0; i<DIM; i++) {
        //float temp = 0.0;
        if (abs(b->lower[i]) > abs(b->upper[i])) {
            temp = abs(b->lower[i]);
        }
        else {
            temp = abs(b->upper[i]);
        }
        if (temp > systemSize) {
            systemSize = temp;
        }
    }
    return systemSize;
}

bool particleWithinBox(Particle &p, Box &b) {
    for (int i=0; i<DIM; i++) {
        if (p.x[i] > b.upper[i] || p.x[i] < b.lower[i]) {
            return false;
        }
    }
    return true;
}
