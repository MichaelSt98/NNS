//
// Created by Michael Staneker on 23.02.21.
//

#ifndef CUDA_NBODY_KERNELSWRAPPER_H
#define CUDA_NBODY_KERNELSWRAPPER_H

#include <iostream>
#include <cuda.h>

#include "Kernels.cuh"

namespace kernel {

    float setDrawArray(dim3 gridSize, dim3 blockSize, float *ptr, float *x, float *y, float *z, int n);

    float resetArrays(dim3 gridSize, dim3 blockSize, int *mutex, float *x, float *y, float *z, float *mass, int *count,
                      int *start, int *sorted, int *child, int *index, float *maxX, float *minY, float *maxY,
                      float *minZ, float *maxZ  float *top, int n, int m);

    float computeBoundingBox(dim3 gridSize, dim3 blockSize, int *mutex, float *x, float *y, float *z,
                             float *maxX, float *minY, float *maxY, float *minZ, float *maxZ float *top, int n);

    float buildTree(dim3 gridSize, dim3 blockSize, float *x, float *y, float *z, float *mass, int *count, int *start,
                    int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                    float *minZ, float *maxZ int n, int m);

    float centreOfMass(dim3 gridSize, dim3 blockSize, float *x, float *y, float *z, float *mass, int *count, int *start,
                       int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                       float *minZ, float *maxZ int n, int m);

    float sort(dim3 gridSize, dim3 blockSize, int *count, int *start, int *sorted, int *child, int *index, int n);

    float computeForces(dim3 gridSize, dim3 blockSize, float *x, float *y, float *z, float *vx, float *vy, float *vz,
                        float *ax, float *ay, float *az, float *mass, int *sorted, int *child,
                        float *minX, float *maxX, int n, float g);

    float update(dim3 gridSize, dim3 blockSize, float *x, float *y, float *z, float *vx, float *vy, float *vz,
                 float *ax, float *ay, float *az, int n, float dt, float d);

    float copy(dim3 gridSize, dim3 blockSize, float *x, float *y, float *z, float *out, int n);

}

#endif //CUDA_NBODY_KERNELSWRAPPER_H
