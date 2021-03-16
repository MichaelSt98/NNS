//
// Created by Michael Staneker on 15.03.21.
//

#ifndef BARNESHUTSERIAL_DOMAIN_H
#define BARNESHUTSERIAL_DOMAIN_H

#include "Constants.h"
#include "Particle.h"
#include <cmath>

typedef struct Box {
    float lower[DIM];
    float upper[DIM];
} Box;

float getSystemSize(Box *b);

bool particleWithinBox(Particle &p, Box &b);

#endif //BARNESHUTSERIAL_DOMAIN_H
