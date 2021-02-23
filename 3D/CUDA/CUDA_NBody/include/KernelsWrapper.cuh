//
// Created by Michael Staneker on 23.02.21.
//

#ifndef CUDA_NBODY_KERNELSWRAPPER_H
#define CUDA_NBODY_KERNELSWRAPPER_H

#include <iostream>
#include <cuda.h>

#include "Kernels.cuh"

dim3 gridSize  = 512;
dim3 blockSize = 256;

namespace kernel {

    float setDrawArray(float *ptr, float *x, float *y, float *z, int n);

    float resetArrays(int *mutex, float *x, float *y, float *z, float *mass, int *count,
                      int *start, int *sorted, int *child, int *index, float *maxX, float *minY, float *maxY,
                      float *minZ, float *maxZ,  float *top, int n, int m);

    float computeBoundingBox(int *mutex, float *x, float *y, float *z,
                             float *maxX, float *minY, float *maxY, float *minZ, float *maxZ, float *top, int n);

    float buildTree(float *x, float *y, float *z, float *mass, int *count, int *start,
                    int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                    float *minZ, float *maxZ, int n, int m);

    float centreOfMass(float *x, float *y, float *z, float *mass, int *count, int *start,
                       int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                       float *minZ, float *maxZ, int n, int m);

    float sort(int *count, int *start, int *sorted, int *child, int *index, int n);

    float computeForces(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                        float *ax, float *ay, float *az, float *mass, int *sorted, int *child,
                        float *minX, float *maxX, int n, float g);

    float update(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                 float *ax, float *ay, float *az, int n, float dt, float d);

    float copy(float *x, float *y, float *z, float *out, int n);

}

#endif //CUDA_NBODY_KERNELSWRAPPER_H
