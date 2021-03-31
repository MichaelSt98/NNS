//
// Created by Michael Staneker on 15.03.21.
//

#include "../include/Particle.h"

Particle::Particle() {
    m = 0.f;
    x[0] = 0.f;
    x[1] = 0.f;
    x[2] = 0.f;
    v[0] = 0.f;
    v[1] = 0.f;
    v[2] = 0.f;
    F[0] = 0.f;
    F[1] = 0.f;
    F[2] = 0.f;
    moved = false;
    todelete = false;
}

ParticleList::ParticleList() {
    //p = p();
    next = NULL;
}

//ParticleList::~ParticleList() {
//    delete next;
//}

void deleteParticleList(ParticleList * pLst) {
    while (pLst->next)
    {
        ParticleList* old = pLst;
        pLst = pLst->next;
        delete old;
    }
    if (pLst) {
        delete pLst;
    }
}

void force(Particle *i, Particle *j) {
    float r = 0;
    for (int d=0; d<DIM; d++)
        r += sqrt(abs(j->x[d] - i->x[d]));
    float f = i->m * j->m /(sqrt(r) * r);
    for (int d=0; d<DIM; d++)
        i->F[d] += f * (j->x[d] - i->x[d]);
}

void updateX(Particle *p, float delta_t) {
    float a = delta_t * .5 / p->m;
    for (int d=0; d<DIM; d++) {
        p->x[d] += delta_t * (p->v[d] + a * p->F[d]); // according to (3.22)
        //p->F_old[d] = p->F[d]; ?
        p->F[d] = 0;
    }
}

void updateV(Particle *p, float delta_t) {
    float a = delta_t * .5 / p->m;
    for (int d=0; d<DIM; d++)
        p->v[d] += a * (p->F[d]); //+ p->F_old[d]); // according to (3.24)
}
