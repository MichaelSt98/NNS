//
// Created by Michael Staneker on 25.01.21.
//

#ifndef NBODY_OCTANT_H
#define NBODY_OCTANT_H

#include "Vector3D.h"
#include <iostream>
#include <utility>

class Tree;

class Octant {

private:

    double length;
    Vector3D center;

public:

    /**!
     * Constructor for Octant class.
     *
     * @param _x x coordinate for center
     * @param _y y coordinate for center
     * @param _z z coordinate for center
     * @param _length length of octant instance
     */
    Octant(double _x, double _y, double _z, double _length);

    /**!
     * Move constructor for Octant class.
     *
     * @param otherOctant other Octant instance
     */
    Octant(Octant&& otherOctant);

    /**!
     * Copy constructor for Octant class.
     *
     * @param otherOctant other Octant istance
     */
    Octant(const Octant& otherOctant);

    /**!
     * Overwritten stream operator to print Octant instances.
     */
    friend std::ostream &operator<<(std::ostream &os, const Octant &octant);

    /**!
     * get length of the octant instance
     *
     * @return length of the octant instance as double
     */
    double getLength() const;

    /**!
     * Check if particle is within the octant instance.
     *
     * @param particle Body instance
     * @return bool, whether particle within the Octant
     */
    bool contains(const Vector3D &particle) const;

    /**!
     * Get the corresponding sub-octant (when subdividing is required) in dependence of the particle to be insert
     *
     * * UNW -> 0
     * * UNE -> 1
     * * USW -> 2
     * * USE -> 3
     * * LNW -> 4
     * * LNE -> 5
     * * LSW -> 6
     * * LSE -> 7
     *
     * @param particle Body instance
     * @return integer, sub-octant
     */
    int getSubOctant(const Vector3D &particle) const;

    /**!
     * (- + +) or upper-north-west
     *
     * @return upper-north-west octant
     */
    Octant getUNW() const;

    /**!
     * (+ + +) or upper-north-east
     *
     * @return upper-north-east octant
     */
    Octant getUNE() const;

    /**!
     * (- - +) or upper-south-west
     *
     * @return upper-south-west octant
     */
    Octant getUSW() const;

    /**!
     * (+ - +) or upper-south-east
     *
     * @return upper-south-east octant
     */
    Octant getUSE() const;

    /**!
     * (- + -) or lower-north-west
     *
     * @return lower-north-west octant
     */
    Octant getLNW() const;

    /**!
     * (+ + -) or lower-north-east
     *
     * @return lower-north-east octant
     */
    Octant getLNE() const;

    /**!
     * (- - -) or lower-south-west
     *
     * @return lower-south-west octant
     */
    Octant getLSW() const;

    /**!
     * (+ - -) or lower-south-east
     *
     * @return lower-south-east
     */
    Octant getLSE() const;

};


#endif //NBODY_OCTANT_H
