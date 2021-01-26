#include "../include/Particle.h"

Particle::Particle(double _x, double _y, double _dx, double _dy, double _m) :
                        x { _x }, y { _y }, dx { _dx }, dy { _dy }, m { _m } {
    ax = 0.0;
    ay = 0.0;
    fx = 0.0;
    fy = 0.0;
}

Particle::Particle() : x { 0.0 }, y { 0.0 }, dx { 0.0 }, dy { 0.0 }, m { 0.0 } {
    ax = 0.0;
    ay = 0.0;
    fx = 0.0;
    fy = 0.0;
}

Particle::Particle(double _x, double _y, double _m) :
            x {_x }, y { _y }, dx { 0.0 }, dy { 0.0 } , m { _m } {
    ax = 0.0;
    ay = 0.0;
    fx = 0.0;
    fy = 0.0;
}

void Particle::move(const Rectangle &map_bounds, float timeStep) {
    if (x + dx < 0 || x + dx > map_bounds.width)
        dx = -dx;
    if (y + dy < 0 || y + dy > map_bounds.height)
        dy = -dy;
    x += dx * timeStep;
    y += dy * timeStep;
}

void Particle::advance(const Rectangle &map_bounds, float timeStep) {

    dx += timeStep * fx / m;
    dy += timeStep * fy / m;

    move(map_bounds, timeStep);

    resetForce();
}

float Particle::getDistance(const Particle &otherParticle) {
    double x_diff = (x-otherParticle.x);
    double y_diff = (y-otherParticle.y);
    double distanceSquared = x_diff*x_diff+y_diff*y_diff;

    if (distanceSquared < 100.0) {
        distanceSquared = 100.0;
    }

    std::cout << "x = " << x << " other.x = " << otherParticle.x << std::endl;
    std::cout << "x_diff: " << x_diff << std::endl;
    std::cout << "y_diff: " << y_diff << std::endl;
    std::cout << "distance^2: " << distanceSquared << std::endl;

    return (float)sqrt(distanceSquared);
}

void Particle::accelerate(const Particle& interactingParticle) {

    float distance = getDistance(interactingParticle);
    float acceleration = (9.81f / (distance*distance)) * interactingParticle.m;

    //dx /= distance;
    //dy /= distance;

    std::cout << "dx: " << dx << std::endl;

    ax -= acceleration * dx;
    ay -= acceleration * dy;
}

void Particle::resetForce() {
    fx = 0;
    fy = 0;
}

void Particle::calculateForce(const Particle &interactingParticle) {
    double EPS = 50.0f;//3e-4; //3E4;      // softening parameter (just to avoid infinities)
    double diffX = interactingParticle.x - x;
    double diffY = interactingParticle.y - y;
    double distance = sqrt(diffX*diffX + diffY*diffY);
    //std::cout << "distance: " << distance << std::endl;
    double force = (9.81 * m * interactingParticle.m) / (distance*distance + EPS*EPS);
    fx += force * diffX / distance;
    fy += force * diffY / distance;
}

bool Particle::identical(const Particle &otherParticle) {
    double delta = 1.0;
    if (x == otherParticle.x && y == otherParticle.y) {
        return true;
    }
    if (abs(otherParticle.x - x) < delta && abs(otherParticle.y - y) < delta) {
        return true;
    }
    return false;
}