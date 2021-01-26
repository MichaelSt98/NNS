#ifndef QUADTREE_PARTICLE_H
#define QUADTREE_PARTICLE_H

#include "Rectangle.h"
#include "QuadTree.h"

#include <any>

class QuadTree;

class Particle {
    friend class QuadTree;

public:

    double x;   //! x-coordinate
    double y;   //! y-coordinate
    double dx;  //! x-velocity
    double dy;  //! y-velocity
    double m;   //! mass

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

    /**!
     *
     */
    void move(const Rectangle &map_bounds);

private:
    /**!
     *
     */
    QuadTree *qt = nullptr;

    /**!
     *
     */
    Particle(const Particle&) = delete;
};


#endif //QUADTREE_PARTICLE_H
