//
// Created by Michael Staneker on 12.04.21.
//

#ifndef OOP_DOMAIN_H
#define OOP_DOMAIN_H

#include "Vector3.h"
#include "Logger.h"

typedef float dFloat;

class Domain {
public:
    Domain();
    Domain(dFloat lowerX, dFloat lowerY, dFloat lowerZ, dFloat upperX, dFloat upperY, dFloat upperZ);
    Domain(dFloat size);
    Domain(Vector3<dFloat> lowerVec, Vector3<dFloat> upperVec);
    Domain(Domain &domain);

    const Domain& operator=(const Domain& rhs);

    friend std::ostream &operator << (std::ostream &os, const Domain &domain);

    Vector3<dFloat> lower;
    Vector3<dFloat> upper;

    dFloat getSystemSizeX();
    dFloat getSystemSizeY();
    dFloat getSystemSizeZ();
    void getSystemSize(Vector3<dFloat> &systemSize);

    dFloat getCenterX();
    dFloat getCenterY();
    dFloat getCenterZ();
    void getCenter(Vector3<dFloat> &center);

    bool withinDomain(Vector3<dFloat> &vec);
};

inline std::ostream &operator << (std::ostream &os, const Domain &domain) {
    os << "lower: " << domain.lower << "    upper: " << domain.upper;
    return os;
}


#endif //OOP_DOMAIN_H
