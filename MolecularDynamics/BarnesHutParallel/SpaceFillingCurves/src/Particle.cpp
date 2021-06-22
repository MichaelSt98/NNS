//
// Created by Michael Staneker on 15.03.21.
//

#include "../include/Particle.h"

//float smoothing = 1e-5;

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
    F_old[0] = 0.f;
    F_old[1] = 0.f;
    F_old[2] = 0.f;
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
    double r = 0;
    for (int d=0; d<DIM; d++)
        //r += sqrt(abs(j->x[d] - i->x[d]));
        r += (j->x[d] - i->x[d]) * (j->x[d] - i->x[d]);
    double f = G * i->m * j->m /(sqrt(r) * r); // + smoothing);
    if (r < 1e-20){
        Logger(WARN) << "In force: encountered very low value of r";
    }
    for (int d=0; d<DIM; d++) {
        i->F[d] += f * (j->x[d] - i->x[d]);
    }
    //Logger(INFO) << "force(): Force = (" << i->F[0] << ", " << i->F[1] << ", " << i->F[2] << ")";
}

void updateX(Particle *p, float delta_t) {
    //Logger(INFO) << "updateX(): Force = (" << p->F[0] << ", " << p->F[1] << ", " << p->F[2] << ")";
    float a = delta_t * .5 / p->m;
    for (int d=0; d<DIM; d++) {
        p->x[d] += delta_t * (p->v[d] + a * p->F[d]); // according to (3.22)
        p->F_old[d] = p->F[d]; //?
        //p->F[d] = 0;
    }
}

void updateV(Particle *p, float delta_t) {
    float a = delta_t * .5 / p->m;
    for (int d=0; d<DIM; d++) {
        p->v[d] += a * (p->F[d] + p->F_old[d]); //+ p->F_old[d]); // according to (3.24)
    }
}

std::string p2str(const Particle &p) {
    return "x = (" + std::to_string(p.x[0]) // x
                    + std::to_string(p.x[1]) // y
                    + std::to_string(p.x[2]) // z
                    + "), m = " + std::to_string(p.m);
}
