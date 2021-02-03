//
// Created by Michael Staneker on 25.01.21.
//

#ifndef NBODY_VECTOR3D_H
#define NBODY_VECTOR3D_H

#include <cmath>
#include <boost/mpi.hpp>

class Vector3D {

public:
    friend class boost::serialization::access;

    template<class Archive>
            void serialize(Archive &ar, const unsigned int version) {
                ar & x;
                ar & y;
                ar & z;
            }

    double x;
    double y;
    double z;

    /**!
     * Standard constructor for Vector3D class.
     */
    Vector3D();

    /**!
     * Constructor for Vector3D class.
     *
     * @param _x x coordinate
     * @param _y y coordinate
     * @param _z z coordinate
     */
    Vector3D(double _x, double _y, double _z);

    /**!
     * Calculate magnitude of the Vector3D instance.
     *
     * @return double, the magnitude
     */
    double magnitude();

    /**!
     * Calculate magnitude (static method).
     *
     * @param _x x coordinate
     * @param _y y coordinate
     * @param _z y coordinate
     * @return double, the magnitude
     */
    static double magnitude(double _x, double _y, double _z);
};


#endif //NBODY_VECTOR3D_H
