//
// Created by Michael Staneker on 25.01.21.
//

#ifndef NBODY_VECTOR3D_H
#define NBODY_VECTOR3D_H

#include <cmath>

class Vector3D {

public:
    double x;
    double y;
    double z;

    Vector3D();

    Vector3D(double _x, double _y, double _z);

    double magnitude();

    static double magnitude(double _x, double _y, double _z);
};


#endif //NBODY_VECTOR3D_H
