//
// Created by Michael Staneker on 25.01.21.
//

#ifndef NBODY_BODY_H
#define NBODY_BODY_H

#include <cstdint>
#include <bitset>

#include "Vector3D.h"

class Body {

public:
    double mass;
    Vector3D position;
    Vector3D velocity;
    Vector3D acceleration;

    Body();
    Body(double _mass);

    static constexpr std::uint_fast32_t MASK_COORD2KEY { 0x001FFFFF };

    std::uint64_t getKey();

    // preparation function for generating a new key for the hashed tree position
    void discretizePosition(double maxSpan, Vector3D origin);

private:
    std::uint_fast32_t i_x, i_y, i_z;
};


#endif //NBODY_BODY_H
