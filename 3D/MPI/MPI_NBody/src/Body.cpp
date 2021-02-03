//
// Created by Michael Staneker on 25.01.21.
//

#include "../include/Body.h"

Body::Body() : mass { 0.0 }, position(), velocity(), acceleration() {}

Body::Body(double _mass) : mass { _mass }, position(), velocity(), acceleration() {}