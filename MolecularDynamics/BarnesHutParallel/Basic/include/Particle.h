//
// Created by Michael Staneker on 15.03.21.
//

#ifndef BARNESHUTSERIAL_PARTICLE_H
#define BARNESHUTSERIAL_PARTICLE_H

#include "Constants.h"
#include <cmath>

struct Particle {
    float m;
    float x[DIM];
    float v[DIM];
    float F[DIM];
    bool moved;
    bool todelete;

    Particle();
};

struct ParticleList {
    Particle p;
    ParticleList *next;

    ParticleList();
    //~ParticleList();
};

void deleteParticleList(ParticleList * pLst);

void force(Particle *i, Particle *j);

void updateX(Particle *p, float delta_t);

void updateV(Particle *p, float delta_t);

#endif //BARNESHUTSERIAL_PARTICLE_H
