//
// Created by Michael Staneker on 22.02.21.
//

/**
 * See [Summary: An Efficient CUDA Implementation of the Tree-Based Barnes Hut n-Body Algorithm](../resources/NBodyCUDA.md)
 */

#ifndef CUDA_NBODY_KERNELS_CUH
#define CUDA_NBODY_KERNELS_CUH

#include <iostream>
#include <stdio.h>
#include <cuda.h>

__global__ void setDrawArrayKernel(float *ptr, float *x, float *y, float *z, int n);
__global__ void resetArraysKernel(int *mutex, float *x, float *y, float *z, float *mass, int *count, int *start,
                                  int *sorted, int *child, int *index, float *maxX, float *minY, float *maxY,
                                  float *minZ, float *maxZ, float *top, int n, int m);

// Kernel 1: computes bounding box around all bodies
__global__ void computeBoundingBoxKernel(int *mutex, float *x, float *y, float *z, float *maxX, float *minY,
                                         float *maxY, float *minZ, float *maxZ, float *top, int n);

// Kernel 2: hierarchically subdivides the root cells
__global__ void buildTreeKernel(float *x, float *y, float *z, float *mass, int *count, int *start,
                                int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                float *minZ, float *maxZ, int n, int m);

// Kernel 3: computes the COM for each cell
__global__ void centreOfMassKernel(float *x, float *y, float *z, float *mass, int *index, int n);

// Kernel 4: sorts the bodies
__global__ void sortKernel(int *count, int *start, int *sorted, int *child, int *index, int n);

// Kernel 5: computes the forces
__global__ void computeForcesKernel(float* x, float *y, float *z, float *vx, float *vy, float *vz,
                                    float *ax, float *ay, float *az, float *mass, int *sorted, int *child,
                                    float *minX, float *maxX, int n, float g);

// Kernel 6: updates the bodies
__global__ void updateKernel(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                             float *ax, float *ay, float *az, int n, float dt, float d);

__global__ void copyKernel(float *x, float *y, float *z, float *out, int n);


#endif //CUDA_NBODY_KERNELS_CUH
