//
// Created by Michael Staneker on 15.03.21.
//

#ifndef BARNESHUTSERIAL_PARTICLE_H
#define BARNESHUTSERIAL_PARTICLE_H

#include "Constants.h"
#include <cmath>
#include <mpi.h>

typedef struct {
    float m;
    float x[DIM];
    float v[DIM];
    float F[DIM];
    bool moved = false;
    bool todelete = false;
} Particle;

/*
MPI_Datatype mpiParticle;
int mpiParticleLengths[6] = {1, DIM, DIM, DIM, 1, 1};
const MPI_Aint mpiParticleDisplacements[6] ={ 0, sizeof(float), 2*sizeof(float), 3*sizeof(float), 4*sizeof(float), 4*sizeof(float) + sizeof(bool) };
MPI_Datatype mpiParticleTypes[6] = { MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_C_BOOL, MPI_C_BOOL }; // MPI_C_BOOL ?
*/

typedef struct ParticleList {
    Particle p;
    struct ParticleList *next;
} ParticleList;


void force(Particle *i, Particle *j);

void updateX(Particle *p, float delta_t);

void updateV(Particle *p, float delta_t);

#endif //BARNESHUTSERIAL_PARTICLE_H
