/**
 * Wrapping CUDA Kernel functions.
 */

#ifndef CUDA_NBODY_KERNELSWRAPPER_H
#define CUDA_NBODY_KERNELSWRAPPER_H

#include <iostream>
#include <cuda.h>

#include "Kernels.cuh"


namespace kernel {

    float resetArrays(int *mutex, float *x, float *y, float *z, float *mass, int *count,
                      int *start, int *sorted, int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                      float *minZ, float *maxZ, int n, int m, bool timing=false);

    float computeBoundingBox(int *mutex, float *x, float *y, float *z, float *minX,
                             float *maxX, float *minY, float *maxY, float *minZ, float *maxZ, int n, bool timing=false);

    float buildTree(float *x, float *y, float *z, float *mass, int *count, int *start,
                    int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                    float *minZ, float *maxZ, int n, int m, bool timing=false);

    float centreOfMass(float *x, float *y, float *z, float *mass, int *index, int n, bool timing=false);

    float sort(int *count, int *start, int *sorted, int *child, int *index, int n, bool timing=false);

    float computeForces(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                        float *ax, float *ay, float *az, float *mass, int *sorted, int *child,
                        float *minX, float *maxX, int n, float g, bool timing=false);

    float update(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                 float *ax, float *ay, float *az, int n, float dt, float d, bool timing=false);

}

#endif //CUDA_NBODY_KERNELSWRAPPER_H
