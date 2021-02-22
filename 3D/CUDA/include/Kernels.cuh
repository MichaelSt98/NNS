//
// Created by Michael Staneker on 22.02.21.
//

#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

__global__ void setDrawArrayKernel(float *ptr, float *x, float *y, int n);
__global__ void resetArraysKernel(int *mutex, float *x, float *y, float *mass, int *count, int *start, int *sorted, int *child, int *index, float *left, float *right, float *bottom, float *top, int n, int m);

// Kernel 1: computes bounding box around all bodies
__global__ void computeBoundingBoxKernel(int *mutex, float *x, float *y, float *left, float *right, float *bottom, float *top, int n);

// Kernel 2: hierachically subdivides the root cells
__global__ void buildTreeKernel(float *x, float *y, float *mass, int *count, int *start, int *child, int *index, float *left, float *right, float *bottom, float *top, int n, int m);

// Kernel 3: computes the COM for each cell
__global__ void centreOfMassKernel(float *x, float *y, float *mass, int *index, int n);

// Kernel 4: sorts the bodies
__global__ void sortKernel(int *count, int *start, int *sorted, int *child, int *index, int n);

// Kernel 5: computes the forces
__global__ void computeForcesKernel(float* x, float *y, float *vx, float *vy, float *ax, float *ay, float *mass, int *sorted, int *child, float *left, float *right, int n, float g);

// Kernel 6: updates the bodies
__global__ void updateKernel(float *x, float *y, float *vx, float *vy, float *ax, float *ay, int n, float dt, float d);

__global__ void copyKernel(float* x, float* y, float* out, int n);


#endif //CUDA_KERNELS_CUH
