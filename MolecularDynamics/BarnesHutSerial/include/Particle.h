//
// Created by Michael Staneker on 15.03.21.
//

#ifndef BARNESHUTSERIAL_PARTICLE_H
#define BARNESHUTSERIAL_PARTICLE_H

#include "Constants.h"
#include <cmath>

typedef struct {
    float m;
    float x[DIM];
    float v[DIM];
    float F[DIM];
    bool moved = false;
    bool todelete = false;
} Particle;


typedef struct ParticleList {
    Particle p;
    struct ParticleList *next;
} ParticleList;


void force(Particle *i, Particle *j);

void updateX(Particle *p, float delta_t);

void updateV(Particle *p, float delta_t);

#endif //BARNESHUTSERIAL_PARTICLE_H
