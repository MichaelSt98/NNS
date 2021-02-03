//
// Created by Michael Staneker on 25.01.21.
//

#ifndef NBODY_BODY_H
#define NBODY_BODY_H

#include "Vector3D.h"

class Body {

public:
    double mass;
    Vector3D position;
    Vector3D velocity;
    Vector3D acceleration;

    Body();
    Body(double _mass);
};


#endif //NBODY_BODY_H
