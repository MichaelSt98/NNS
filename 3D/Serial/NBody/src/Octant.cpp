//
// Created by Michael Staneker on 25.01.21.
//

#include "../include/Octant.h"

Octant::Octant(double _x, double _y, double _z, double _length) : center(_x, _y, _z) {
    length = _length;
}

Octant::Octant(Octant&& otherOctant) :
                center { std::move(otherOctant.center) }, length {std::move(otherOctant.length)} { }

Octant::Octant(const Octant& otherOctant) :
                center (otherOctant.center), length (otherOctant.length) { }


std::ostream &operator<<(std::ostream &os, const Octant &octant) {
    os  << "center: (x = " << octant.center.x << ", y = " << octant.center.y
        << ", z = " << octant.center.z << ") length: " << octant.length << std::endl;
    return os;
}

double Octant::getLength() const {
    return length;
}

bool Octant::contains(const Vector3D &particle) const {
    return (particle.x <= (center.x + length/2.0) &&
            particle.x >= (center.x - length/2.0) &&
            particle.y <= (center.y + length/2.0) &&
            particle.y >= (center.y - length/2.0) &&
            particle.z <= (center.z + length/2.0) &&
            particle.z >= (center.z - length/2.0));
}

// UNW = 0 // UNE = 1 // USW = 2 // USE = 3
// LNW = 4 // LNE = 5 // LSW = 6 // LSE = 7
int Octant::getSubOctant(const Vector3D &particle) const {
    if (particle.z <= center.z) { // Lower
        if (particle.y <= center.y) { // South
            if (particle.x <= center.x) { return 6; } // West
            else { return 7; } // (East) return getLSW(); }
        }
        else {
            if (particle.x <= center.x) { return 4; } // East
            else { return 5; } // South
        }
    }
    else { // Upper
        if (particle.y <= center.y) { // West
            if (particle.x <= center.x) { return 2; } // North
            else { return 3; } // South
        }
        else {
            if (particle.x <= center.x) { return 0; } // East
            else { return 1; } // South
        }
    }
}


Octant Octant::getUNW() const {
    //std::cout << "UNW" << std::endl;
    double newLength = length/4.0;
    return Octant{center.x - newLength,
                  center.y + newLength,
                  center.z + newLength,
                  length/2.0};
}

Octant Octant::getUNE() const {
    //std::cout << "UNE" << std::endl;
    double newLength = length/4.0;
    return Octant{center.x + newLength,
                  center.y + newLength,
                  center.z + newLength,
                  length/2.0};
}

Octant Octant::getUSW() const {
    //std::cout << "USW" << std::endl;
    double newLength = length/4.0;
    return Octant{center.x - newLength,
                  center.y - newLength,
                  center.z + newLength,
                  length/2.0};
}

Octant Octant::getUSE() const {
    //std::cout << "USE" << std::endl;
    double newLength = length/4.0;
    return Octant{center.x + newLength,
                  center.y - newLength,
                  center.z + newLength,
                  length/2.0};
}

Octant Octant::getLNW() const {
    //std::cout << "LNW" << std::endl;
    double newLength = length/4.0;
    return Octant{center.x - newLength,
                  center.y + newLength,
                  center.z - newLength,
                  length/2.0};
}

Octant Octant::getLNE() const {
    //std::cout << "LNE" << std::endl;
    double newLength = length/4.0;
    return Octant{center.x + newLength,
                  center.y + newLength,
                  center.z - newLength,
                  length/2.0};
}

Octant Octant::getLSW() const {
    //std::cout << "LSW" << std::endl;
    double newLength = length/4.0;
    return Octant{center.x - newLength,
                  center.y - newLength,
                  center.z - newLength,
                  length/2.0};
}

Octant Octant::getLSE() const {
    //std::cout << "LSE" << std::endl;
    double newLength = length/4.0;
    return Octant{center.x + newLength,
                  center.y - newLength,
                  center.z - newLength,
                  length/2.0};
}
