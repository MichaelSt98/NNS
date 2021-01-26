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

public:

    double length;
    Vector3D center;

    Octant(double _x, double _y, double _z, double _length);

    Octant(Octant&& otherOctant);

    Octant(const Octant& otherOctant);

    friend std::ostream &operator<<(std::ostream &os, const Octant &octant);

    double getLength() const;

    bool contains(const Vector3D &particle) const;

    int getSubOctant(const Vector3D &particle) const; //, Tree *tree) const;

    // UNW; //0 // UNE; //1 // USW; //2 // USE; //3
    // LNW; //4 // LNE; //5 // LSW; //6 // LSE; //7
    //void getSubOctant(const Vector3D &particle,
    //                  Octant *unw, Octant *une, Octant *usw, Octant *use,
    //                  Octant *lnw, Octant *lne, Octant *lsw, Octant *lse);

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
