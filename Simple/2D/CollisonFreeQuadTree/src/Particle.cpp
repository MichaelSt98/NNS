#include "../include/Particle.h"

Particle::Particle(double _x, double _y, double _dx, double _dy, double _m) :
                        x { _x }, y { _y }, dx { _dx }, dy { _dy }, m { _m } { }

Particle::Particle() : x { 0.0f }, y { 0.0f }, dx { 0.0f }, dy { 0.0f }, m { 0.0f } {}

void Particle::move(const Rectangle &map_bounds) {
    if (x + dx < 0 || x + dx > map_bounds.width)
        dx = -dx;
    if (y + dy < 0 || y + dy > map_bounds.height)
        dy = -dy;
    x += dx;
    y += dy;
}