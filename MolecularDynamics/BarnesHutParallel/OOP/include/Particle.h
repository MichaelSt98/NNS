//
// Created by Michael Staneker on 12.04.21.
//

#ifndef OOP_PARTICLE_H
#define OOP_PARTICLE_H

#include "Vector3.h"

typedef float pFloat;

class Particle {

public:

    pFloat m;
    Vector3<pFloat> x;
    Vector3<pFloat> v;
    Vector3<pFloat> F;
    Vector3<pFloat> oldF;

    bool moved;
    bool toDelete;

    Particle();
    Particle(Vector3<pFloat> x);
    Particle(Vector3<pFloat> x, Vector3<pFloat> v);

    void force(Particle *j);
    void force(Particle &j);

    void updateX(float deltaT);
    void updateV(float deltaT);

    friend std::ostream &operator << (std::ostream &os, const Particle &p);
};

inline std::ostream &operator<<(std::ostream &os, const Particle &p)
{
    os << "\tm = " << p.m << std::endl;
    os << "\tx = " << p.x << std::endl;
    os << "\tv = " << p.v << std::endl;
    os << "\tF = " << p.v << std::endl;
    os << "\tmoved    = " << (p.moved ? "true " : "false") << std::endl;
    os << "\ttoDelete = " << (p.toDelete ? "true" : "false");
    return os;
}

void force(Particle *i, Particle *j);

void force(Particle &i, Particle &j);


#endif //OOP_PARTICLE_H
