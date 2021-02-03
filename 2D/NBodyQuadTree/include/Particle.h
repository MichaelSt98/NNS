#ifndef QUADTREE_PARTICLE_H
#define QUADTREE_PARTICLE_H

#include "Rectangle.h"
#include "QuadTree.h"

#include <iostream>
#include <any>
#include <math.h>

class QuadTree;

class Particle {
    friend class QuadTree;

public:

    double x;   //! x-coordinate
    double y;   //! y-coordinate
    double dx;  //! x-velocity
    double dy;  //! y-velocity
    double m;   //! mass
    double ax;
    double ay;
    double fx;
    double fy;

    /**!
     *
     * @param _x
     * @param _y
     * @param _dx
     * @param _dy
     * @param _m
     */
    Particle(double _x, double _y, double _dx, double _dy, double _m);

    Particle();

    Particle(double _x, double _y, double _m);

    /**!
     *
     */
    void move(const Rectangle &map_bounds, float timeStep);

    void advance(const Rectangle &map_bounds, float timeStep);

    float getDistance(const Particle &otherParticle);

    void accelerate(const Particle &interactingParticle);

    void resetForce();

    void calculateForce(const Particle &interactingParticle);

    bool identical(const Particle &otherParticle);

private:
    /**!
     *
     */
    QuadTree *qt = nullptr;

    /**!
     *
     */
    //Particle(const Particle&) = delete;
};


#endif //QUADTREE_PARTICLE_H
