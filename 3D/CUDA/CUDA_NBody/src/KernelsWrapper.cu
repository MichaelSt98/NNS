//
// Created by Michael Staneker on 23.02.21.
//

#include "../include/KernelsWrapper.cuh"

float kernel::setDrawArray(float *ptr, float *x, float *y, float *z, int n) {

    float elapsedTime;
    cudaEvent_t start, stop; // used for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    std::cout << "setDrawArrayKernel<<< " /*<< gridSize << ", " << blockSize*/ << " >>>" << std::endl;
    setDrawArrayKernel<<< gridSize, blockSize>>>(ptr, x, y, z, n);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsedTime;

}

float kernel::resetArrays(int *mutex, float *x, float *y, float *z, float *mass, int *count,
                          int *start, int *sorted, int *child, int *index, float *maxX, float *minY, float *maxY,
                          float *minZ, float *maxZ, float *top, int n, int m) {

    float elapsedTime;
    cudaEvent_t start, stop; // used for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    std::cout << "resetArraysKernel<<< " /*<< gridSize << ", " << blockSize*/ << " >>>" << std::endl;
    resetArraysKernel<<< gridSize, blockSize >>>(mutex, x, y, z, mass, count, start, sorted, child, index,
            maxX, minY, maxY, minZ, maxZ, n, m);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsedTime;

}

float kernel::computeBoundingBox(int *mutex, float *x, float *y, float *z,
                                 float *maxX, float *minY, float *maxY, float *minZ, float *maxZ, float *top, int n) {

    float elapsedTime;
    cudaEvent_t start, stop; // used for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    std::cout << "computeBoundingBoxKernel<<< " /*<< gridSize << ", " << blockSize*/ << " >>>" << std::endl;
    computeBoundingBoxKernel<<< gridSize, blockSize >>>(mutex, x, y, z, maxX, minY, maxY, minZ, maxZ, n);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsedTime;

}

float kernel::buildTree(float *x, float *y, float *z, float *mass, int *count, int *start,
                        int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                        float *minZ, float *maxZ, int n, int m) {

    float elapsedTime;
    cudaEvent_t start, stop; // used for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    std::cout << "buildTreeKernel<<< " /*<< gridSize << ", " << blockSize */ << " >>>" << std::endl;
    buildTreeKernel<<< gridSize, blockSize >>>(x, y, z, mass, count, start, child, index,
            minX, maxX, minY, maxY, minZ, maxZ, n, m);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsedTime;

}

float kernel::centreOfMass(float *x, float *y, float *z, float *mass, int *index, int n) {

    float elapsedTime;
    cudaEvent_t start, stop; // used for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    std::cout << "centreOfMassKernel<<< " /*<< gridSize << ", " << blockSize */ << " >>>" << std::endl;
    centreOfMassKernel<<< gridSize, blockSize >>>(x, y, z, mass, index, n);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsedTime;

}

float kernel::sort(int *count, int *start, int *sorted, int *child, int *index, int n) {

    float elapsedTime;
    cudaEvent_t start, stop; // used for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    std::cout << "sortKernel<<< " /*<< gridSize << ", " << blockSize*/ << " >>>" << std::endl;
    sortKernel<<< gridSize, blockSize>>>(count, start, sorted, child, index, n);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsedTime;

}

float kernel::computeForces(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                            float *ax, float *ay, float *az, float *mass, int *sorted, int *child,
                            float *minX, float *maxX, int n, float g) {

    float elapsedTime;
    cudaEvent_t start, stop; // used for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    std::cout << "computeForcesKernel<<< " /*<< gridSize << ", " << blockSize*/ << " >>>" << std::endl;
    computeForcesKernel<<<gridSize, blockSize>>>(x, y, z, vx, vy, vz, ax, ay, az,
            mass, sorted, child, minX, maxX, n, g);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsedTime;

}

float kernel::update(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                    float *ax, float *ay, float *az, int n, float dt, float d) {

    float elapsedTime;
    cudaEvent_t start, stop; // used for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    std::cout << "updateKernel<<< " /*<< gridSize << ", " << blockSize*/ << " >>>" << std::endl;
    updateKernel<<< gridSize, blockSize >>>(x, y, z, vx, vy, vz, ax, ay, az, n, dt, d);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsedTime;

}

float kernel::copy(float *x, float *y, float *z, float *out, int n) {

    float elapsedTime;
    cudaEvent_t start, stop; // used for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    std::cout << "copyKernel<<< " /*<< gridSize << ", " << blockSize*/ << " >>>" << std::endl;
    copyKernel<<< gridSize, blockSize >>>(x, y, z, out, n);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsedTime;

}
