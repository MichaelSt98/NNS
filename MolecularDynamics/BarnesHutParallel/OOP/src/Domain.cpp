//
// Created by Michael Staneker on 12.04.21.
//

#include "../include/Domain.h"

Domain::Domain() {

}

Domain::Domain(dFloat lowerX, dFloat lowerY, dFloat lowerZ, dFloat upperX, dFloat upperY, dFloat upperZ) {
    lower = { lowerX, lowerY, lowerZ };
    upper = { upperX, upperY, upperZ };
}

Domain::Domain(dFloat size) {
    Domain(size, size, size, size, size, size);
}

Domain::Domain(Vector3<dFloat> lowerVec, Vector3<dFloat> upperVec) : lower{ lowerVec }, upper{ upperVec } {

}

Domain::Domain(Domain &domain) : lower { domain.lower }, upper { domain.upper } {

}

const Domain& Domain::operator=(const Domain& rhs) {
    //std::cout << "rhs.lower = " << rhs.lower << std::endl;
    //std::cout << "lower = " << lower << std::endl;
    lower = rhs.lower;
    upper = rhs.upper;
    return (*this);
}

dFloat Domain::getSystemSizeX() {
    return upper.x - lower.y;
}

dFloat Domain::getSystemSizeY() {
    return upper.y - lower.y;
}

dFloat Domain::getSystemSizeZ() {
    return upper.z - lower.z;
}

void Domain::getSystemSize(Vector3<dFloat> &systemSize) {
    systemSize = upper - lower;
}

dFloat Domain::getCenterX() {
    return 0.5 * (upper.x + lower.y);
}

dFloat Domain::getCenterY() {
    return 0.5 * (upper.y + lower.y);
}

dFloat Domain::getCenterZ() {
    return 0.5 * (upper.z + lower.z);
}

void Domain::getCenter(Vector3<dFloat> &center) {
    center = 0.5 * (upper + lower);
}

bool Domain::withinDomain(Vector3<dFloat> &vec) {
    return (vec < upper && vec > lower);
}