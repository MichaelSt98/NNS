#include "../include/Particle.h"

Particle::Particle(const Rectangle &_bounds, std::any _data) :
                bound { _bounds }, data { _data} { }
