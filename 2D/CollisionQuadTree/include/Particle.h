#ifndef QUADTREE_PARTICLE_H
#define QUADTREE_PARTICLE_H

#include "Rectangle.h"
#include "QuadTree.h"

#include <any>

class QuadTree;

class Particle {
    friend class QuadTree;
public:
    Rectangle bound;
    std::any data;

    Particle(const Rectangle &_bounds = {}, std::any _data = {});

private:
    QuadTree *qt = nullptr;
    Particle(const Particle&) = delete;
};


#endif //QUADTREE_PARTICLE_H
