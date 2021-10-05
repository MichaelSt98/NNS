//
// Created by Michael Staneker on 15.03.21.
//

#include "../include/Particle.h"

void force(Particle *i, Particle *j) {
    float r = 0;
    for (int d=0; d<DIM; d++)
        r += (j->x[d] - i->x[d]) * (j->x[d] - i->x[d]);
    float f = i->m * j->m /(sqrt(r) * r);
    for (int d=0; d<DIM; d++)
        i->F[d] += f * (j->x[d] - i->x[d]);
}

void updateX(Particle *p, float delta_t) {
    //std::cout << "Force = (" << p->F[0] << ", " << p->F[1] << ", " << p->F[2] << ")" << std::endl;
    float a = delta_t * .5 / p->m;
    for (int d=0; d<DIM; d++) {
        p->x[d] += delta_t * (p->v[d] + a * p->F[d]); // according to (3.22)
        p->F_old[d] = p->F[d];
        //p->F[d] = 0;
    }
}

void updateV(Particle *p, float delta_t) {
    float a = delta_t * .5 / p->m;
    for (int d=0; d<DIM; d++)
        p->v[d] += a * (p->F[d] + p->F_old[d]); //+ p->F_old[d]); // according to (3.24)
}
