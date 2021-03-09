//
// Created by Michael Staneker on 25.01.21.
//

#include "../include/Body.h"

Body::Body() : mass { 0.0 }, position(), velocity(), acceleration() {}

Body::Body(double _mass) : mass { _mass }, position(), velocity(), acceleration() {}

std::uint64_t Body::getKey(){
    uint64_t key_ { 1 };
    key_ <<= 63; // fill place-holder bit as we use 21 bits of each coordinate (i.e. (3x21=63) bits)

    // map coordinate variables to 21 length bitmap
    std::bitset<21> i_xb { i_x & MASK_COORD2KEY },
            i_yb { i_y & MASK_COORD2KEY },
            i_zb { i_z & MASK_COORD2KEY };

    for (int j=0; j<21; ++j){
        key_ += (static_cast<std::uint_fast64_t>(1) << 3*j) * (i_xb[j] + 2*i_yb[j] + 4*i_zb[j]);
    }
    return key_;
}