//
// Created by Michael Staneker on 22.02.21.
//

#include "../include/Particle.cuh"
#include "../include/Kernels.cuh"

dim3 gridSize = 512;
dim3 blockSize = 256;

void SetDrawArray(float *ptr, float *x, float *y, int n)
{
    setDrawArrayKernel<<< gridSize, blockSize>>>(ptr, x, y, n);
}


void ResetArrays(int *mutex, float *x, float *y, float *mass, int *count, int *start, int *sorted, int *child,
                 int *index, float *left, float *right, float *bottom, float *top, int n, int m)
{
    resetArraysKernel<<< gridSize, blockSize >>>(mutex, x, y, mass, count, start, sorted, child, index, left, right, bottom, top, n, m);
}


void ComputeBoundingBox(int *mutex, float *x, float *y, float *left, float *right, float *bottom, float *top, int n)
{
    computeBoundingBoxKernel<<< gridSize, blockSize >>>(mutex, x, y, left, right, bottom, top, n);
}


void BuildQuadTree(float *x, float *y, float *mass, int *count, int *start, int *child, int *index,
                   float *left, float *right, float *bottom, float *top, int n, int m)
{
    buildTreeKernel<<< gridSize, blockSize >>>(x, y, mass, count, start, child, index, left, right, bottom, top, n, m);
}


void ComputeCentreOfMass(float *x, float *y, float *mass, int *index, int n)
{
    centreOfMassKernel<<<gridSize, blockSize>>>(x, y, mass, index, n);
}


void SortParticles(int *count, int *start, int *sorted, int *child, int *index, int n)
{
    sortKernel<<< gridSize, blockSize >>>(count, start, sorted, child, index, n);
}


void CalculateForces(float* x, float *y, float *vx, float *vy, float *ax, float *ay, float *mass, int *sorted,
                     int *child, float *left, float *right, int n, float g)
{
    computeForcesKernel<<< gridSize, blockSize >>>(x, y, vx, vy, ax, ay, mass, sorted, child, left, right, n, g);
}


void IntegrateParticles(float *x, float *y, float *vx, float *vy, float *ax, float *ay, int n, float dt, float d)
{
    updateKernel<<<gridSize, blockSize >>>(x, y, vx, vy, ax, ay, n, dt, d);
}


void FillOutputArray(float *x, float *y, float *out, int n)
{
    copyKernel<<<gridSize, blockSize >>>(x, y, out, n);
}