//
// Created by Michael Staneker on 25.01.21.
//

#ifndef NBODY_BODY_H
#define NBODY_BODY_H

#include "Vector3D.h"
#include <boost/mpi.hpp>

class Body {

public:

    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & mass;
        ar & position;
        ar & velocity;
        ar & acceleration;
    }

    double mass;
    Vector3D position;
    Vector3D velocity;
    Vector3D acceleration;

    Body();
    Body(double _mass);
};


#endif //NBODY_BODY_H
