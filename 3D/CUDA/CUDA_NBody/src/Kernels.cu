//
// Created by Michael Staneker on 22.02.21.
//

#include "../include/Kernels.cuh"
#include <stdio.h>

__device__ const int blockSize = 256;
__device__ const int warp = 32;
__device__ const int stackSize = 64;
__device__ const float eps_squared = 0.025;
__device__ const float theta = 0.5;

void kernel::setDrawArray(dim3 gridSize, dim3 blockSize, float *ptr, float *x, float *y, float *z, int n) {

    setDrawArrayKernel<<< gridSize, blockSize>>>(ptr, x, y, z, n);

}

void kernel::resetArrays(dim3 gridSize, dim3 blockSize, int *mutex, float *x, float *y, float *z, float *mass, int *count,
                 int *start, int *sorted, int *child, int *index, float *maxX, float *minY, float *maxY,
                 float *minZ, float *maxZ  float *top, int n, int m) {

    resetArraysKernel<<< gridSize, blockSize >>>(mutex, x, y, z, mass, count, start, sorted, child, index,
                                                 maxX, minY, maxY, minZ, maxZ  top, n, m);

}

void kernel::computeBoundingBox(dim3 gridSize, dim3 blockSize, int *mutex, float *x, float *y, float *z,
                        float *maxX, float *minY, float *maxY, float *minZ, float *maxZ float *top, int n) {

    computeBoundingBox<<< gridSize, blockSize >>>(mutex, x, y, z, maxX, minY, maxY, minZ, maxZ top, n);

}

void kernel::buildTree(dim3 gridSize, dim3 blockSize, float *x, float *y, float *z, float *mass, int *count, int *start,
               int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
               float *minZ, float *maxZ int n, int m) {

    buildTreeKernel<<< gridSize, blockSize >>>(x, y, z, mass, count, start, child, index,
                                               minX, maxX, minY, maxY, minZ, maxZ n, m);

}

void kernel::centreOfMass(dim3 gridSize, dim3 blockSize, float *x, float *y, float *z, float *mass, int *count, int *start,
                  int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                  float *minZ, float *maxZ int n, int m) {

    centreOfMassKernel<<< gridSize, blockSize >>>(x, y, z, mass, count, start, child, index,
                                                  minX, maxX, minY, maxY, minZ, maxZ n, m);

}

void kernel::sort(dim3 gridSize, dim3 blockSize, int *count, int *start, int *sorted, int *child, int *index, int n) {

    sortKernel<<< gridSize, blockSize>>>(count, start, sorted, child, index, n);

}

void kernel::computeForces(dim3 gridSize, dim3 blockSize, float *x, float *y, float *z, float *vx, float *vy, float *vz,
                   float *ax, float *ay, float *az, float *mass, int *sorted, int *child,
                   float *minX, float *maxX, int n, float g) {

    computeForcesKernel<<<gridSize, blockSize>>>(x, y, z, vx, vy, vz, ax, ay, az,
                                                 mass, sorted, child, minX, maxX, n, g);

}

void kernel::update(dim3 gridSize, dim3 blockSize, float *x, float *y, float *z, float *vx, float *vy, float *vz,
            float *ax, float *ay, float *az, int n, float dt, float d) {

    updateKernel<<< gridSize, blockSize >>>(x, y, z, vx, vy, vz, ax, ay, az, n, dt, d);

}

void kernel::copy(dim3 gridSize, dim3 blockSize, float *x, float *y, float *z, float *out, int n) {
    copyKernel<<< gridSize, blockSize >>>(x, y, z, out, n);
}


__global__ void setDrawArrayKernel(float *ptr, float *x, float *y, float *z, int n)
{
    int index = threadIdx.x + blockDim.x*blockIdx.x;

    if(index < n){
        ptr[2*index+0] = x[index];
        ptr[2*index+1] = y[index];
        ptr[2*index+2] = z[index];
    }
}


__global__ void resetArraysKernel(int *mutex, float *x, float *y, float *z, float *mass, int *count, int *start,
                                  int *sorted, int *child, int *index, float *minX, float *maxX,
                                  float *minY, float *maxY, float *minZ, float *maxZ, int n, int m)
{
    int bodyIndex = threadIdx.x + blockDim.x*blockIdx.x;
    int stride = blockDim.x*gridDim.x;
    int offset = 0;

    // reset quadtree arrays
    while(bodyIndex + offset < m){
    #pragma unroll 8
        for (int i=0; i<8; i++) {
            child[(bodyIndex + offset)*8 + i] = -1;
        }
        if (bodyIndex + offset < n) {
            count[bodyIndex + offset] = 1;
        }
        else {
            x[bodyIndex + offset] = 0;
            y[bodyIndex + offset] = 0;
            z[bodyIndex + offset] = 0;
            mass[bodyIndex + offset] = 0;
            count[bodyIndex + offset] = 0;
        }
        start[bodyIndex + offset] = -1;
        sorted[bodyIndex + offset] = 0;
        offset += stride;
    }

    if(bodyIndex == 0){
        *mutex = 0;
        *index = n;
        *minX = 0;
        *maxX = 0;
        *minY = 0;
        *maxY = 0;
        *minZ = 0;
        *maxZ = 0;
    }
}

// Kernel 1: computes bounding box around all bodies
__global__ void computeBoundingBoxKernel(int *mutex, float *x, float *y, float *z, float *minX, float *maxX,
                                         float *minY, float *maxY, float *minZ, float *maxZ, int n)
{
    // assigning bodies (array indices) to thread(s)
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    // initialize local min/max
    float x_min = x[index];
    float x_max = x[index];
    float y_min = y[index];
    float y_max = y[index];
    float z_min = z[index];
    float z_max = z[index];

    // initialize block min/max buffer
    __shared__ float x_min_buffer[blockSize];
    __shared__ float x_max_buffer[blockSize];
    __shared__ float y_min_buffer[blockSize];
    __shared__ float y_max_buffer[blockSize];
    __shared__ float z_min_buffer[blockSize];
    __shared__ float z_max_buffer[blockSize];

    int offset = stride;

    // find (local) min/max
    while (index + offset < n) {

        x_min = fminf(x_min, x[index + offset]);
        x_max = fmaxf(x_max, x[index + offset]);
        y_min = fminf(y_min, y[index + offset]);
        y_max = fmaxf(y_max, y[index + offset]);
        z_min = fminf(z_min, z[index + offset]);
        z_max = fmaxf(z_max, z[index + offset]);

        offset += stride;
    }

    // save value in corresponding buffer
    x_min_buffer[threadIdx.x] = x_min;
    x_max_buffer[threadIdx.x] = x_max;
    y_min_buffer[threadIdx.x] = y_min;
    y_max_buffer[threadIdx.x] = y_max;
    z_min_buffer[threadIdx.x] = z_min;
    z_max_buffer[threadIdx.x] = z_max;

    // synchronize threads / wait for unfinished threads
    __syncthreads();

    int i = blockDim.x/2; // assuming blockDim.x is a power of 2!

    // reduction within block
    while (i != 0) {
        if (threadIdx.x < i) {
            x_min_buffer[threadIdx.x] = fminf(x_min_buffer[threadIdx.x], x_min_buffer[threadIdx.x + i]);
            x_max_buffer[threadIdx.x] = fmaxf(x_max_buffer[threadIdx.x], x_max_buffer[threadIdx.x + i]);
            y_min_buffer[threadIdx.x] = fminf(y_min_buffer[threadIdx.x], y_min_buffer[threadIdx.x + i]);
            y_max_buffer[threadIdx.x] = fmaxf(y_max_buffer[threadIdx.x], y_max_buffer[threadIdx.x + i]);
            z_min_buffer[threadIdx.x] = fminf(z_min_buffer[threadIdx.x], z_min_buffer[threadIdx.x + i]);
            z_max_buffer[threadIdx.x] = fmaxf(z_max_buffer[threadIdx.x], z_max_buffer[threadIdx.x + i]);
        }
        __syncthreads();
        i /= 2;
    }

    // global reduction
    if (threadIdx.x == 0) {
        while (atomicCAS(mutex, 0 ,1) != 0) {
            // lock
        }
        *minX = fminf(*minX, x_min_buffer[0]);
        *maxX = fminf(*maxX, x_max_buffer[0]);
        *minY = fminf(*minY, y_min_buffer[0]);
        *maxY = fminf(*maxY, y_max_buffer[0]);
        *minZ = fminf(*minZ, z_min_buffer[0]);
        *maxZ = fminf(*maxZ, z_max_buffer[0]);

        atomicExch(mutex, 0); // unlock
    }
}

// Kernel 2: hierachically subdivides the root cells
__global__ void buildTreeKernel(float *x, float *y, float *z, float *mass, int *count, int *start,
                                int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                float *minZ, float *maxZ int n, int m)
{
    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset;
    bool newBody = true;

    // build quadtree
    float min_x;
    float max_x;
    float min_y;
    float max_y;
    float min_z;
    float max_z;

    int childPath;
    int temp;
    offset = 0;

    while ((bodyIndex + offset) < n) {

        if (newBody) {
            newBody = false;

            min_x = *minX;
            max_x = *maxX;
            min_y = *minY;
            max_y = *maxY;
            min_z = *minZ;
            max_z = *maxZ;

            temp = 0;
            childPath = 0;

            // x direction
            if (x[bodyIndex + offset] < 0.5 * (min_x + max_x)) {
                childPath += 1;
                max_x = 0.5 * (min_x + max_x);
            }
            else {
                min_x = 0.5 * (min_x + max_x);
            }
            // y direction
            if (y[bodyIndex + offset] < 0.5 * (min_y + max_y)) {
                childPath += 1;
                max_y = 0.5 * (min_y + max_y);
            }
            else {
                min_y = 0.5 * (min_y + max_y);
            }
            // z direction
            if (z[bodyIndex + offset] < 0.5 * (min_z + max_z)) {
                childPath += 1;
                max_z = 0.5 * (min_z + max_z);
            }
            else {
                min_z = 0.5 * (min_z + max_z);
            }
        }

        int childIndex = child[temp*8 + childPath];

        // traverse tree until hitting leaf node
        while (childIndex >= n) {

            temp = childIndex;
            childPath = 0;

            // x direction
            if (x[bodyIndex + offset] < 0.5 * (min_x + max_x)) {
                childPath += 1;
                max_x = 0.5 * (min_x + max_x);
            }
            else {
                min_x = 0.5 * (min_x + max_x);
            }
            // y direction
            if (y[bodyIndex + offset] < 0.5 * (min_y + max_y)) {
                childPath += 1;
                max_y = 0.5 * (min_y + max_y);
            }
            else {
                min_y = 0.5 * (min_y + max_y);
            }

            // z direction
            if (z[bodyIndex + offset] < 0.5 * (min_z + max_z)) {
                childPath += 1;
                max_z = 0.5 * (min_z + max_z);
            }
            else {
                min_z = 0.5 * (min_z + max_z);
            }

            atomicAdd(&x[temp], mass[bodyIndex + offset] * x[bodyIndex + offset]);
            atomicAdd(&y[temp], mass[bodyIndex + offset] * y[bodyIndex + offset]);
            atomicAdd(&z[temp], mass[bodyIndex + offset] * z[bodyIndex + offset]);

            atomicAdd(&mass[temp], mass[bodyIndex + offset]);

            atomicAdd(&count[temp], 1);

            childIndex = child[8*temp + childPath];
        }

        if (childIndex != -2) {

            int locked = temp*8 + childPath;

            if (atomicCAS(&child[locked], childIndex, -2) == childIndex) {

                if (childIndex == -1) {
                    child[locked] = bodyIndex + offset;
                }

                else {

                    int patch = 8*n; //4*n
                    while(childIndex >= 0 && childIndex < n){

                        int cell = atomicAdd(index, 1);
                        patch = min(patch, cell);

                        if (patch != cell) {
                            child[8*temp + childPath] = cell;
                        }

                        // insert old particle
                        childPath = 0;
                        if(x[childIndex] < 0.5 * (x_min+x_max)) {
                            childPath += 1;
                        }

                        if (y[childIndex] < 0.5 * (y_min+y_max)) {
                            childPath += 2;
                        }

                        if (z[childIndex] < 0.5 * (z_min+z_max)) {
                            childPath += 4;
                        }

                        x[cell] += mass[childIndex] * x[childIndex];
                        y[cell] += mass[childIndex] * y[childIndex];
                        z[cell] += mass[childIndex] * z[childIndex];

                        mass[cell] += mass[childIndex];
                        count[cell] += count[childIndex];

                        child[8*cell + childPath] = childIndex;

                        start[cell] = -1;

                        // insert new particle
                        temp = cell;
                        childPath = 0;

                        if (x[bodyIndex + offset] < 0.5 * (min_x+max_x)) {
                            childPath += 1;
                            max_x = 0.5 * (min_x+max_x);
                        }
                        else {
                            min_x = 0.5 * (min_x+max_x);
                        }
                        if (y[bodyIndex + offset] < 0.5 * (min_y+max_y)) {
                            childPath += 2;
                            max_y = 0.5 * (min_y + max_y);
                        }
                        else {
                            min_y = 0.5 * (min_y + max_y);
                        }
                        if (z[bodyIndex + offset] < 0.5 * (min_z+max_z)) {
                            childPath += 4;
                            max_z = 0.5 * (min_z + max_z);
                        }
                        else {
                            min_z =  0.5 * (min_z + max_z);
                        }

                        x[cell] += mass[bodyIndex + offset] * x[bodyIndex + offset];
                        y[cell] += mass[bodyIndex + offset] * y[bodyIndex + offset];
                        z[cell] += mass[bodyIndex + offset] * z[bodyIndex + offset];
                        mass[cell] += mass[bodyIndex + offset];
                        count[cell] += count[bodyIndex + offset];
                        childIndex = child[8*temp + childPath];
                    }

                    child[8*temp + childPath] = bodyIndex + offset;

                    __threadfence();  // written to global memory arrays (child, x, y, mass) thus need to fence
                    child[locked] = patch;
                }

                // __threadfence();

                offset += stride;
                newBody = true;
            }
        }

        __syncthreads(); // needed?
    }
}


// Kernel 3: computes the COM for each cell
__global__ void centreOfMassKernel(float *x, float *y, float *z, float *mass, int *index, int n)
{
    int bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x*gridDim.x;
    int offset = 0;

    bodyIndex += n;
    while (bodyIndex + offset < *index) {

        x[bodyIndex + offset] /= mass[bodyIndex + offset];
        y[bodyIndex + offset] /= mass[bodyIndex + offset];
        z[bodyIndex + offset] /= mass[bodyIndex + offset];

        offset += stride;
    }
}


// Kernel 4: sorts the bodies
__global__ void sortKernel(int *count, int *start, int *sorted, int *child, int *index, int n)
{
    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    int s = 0;
    if (threadIdx.x == 0) {
        
        for(int i=0;i<8;i++){
            
            int node = child[i];

            if (node >= n) {  // not a leaf node
                start[node] = s;
                s += count[node];
            }
            else if (node >= 0) {  // leaf node
                sorted[s] = node;
                s++;
            }
        }
    }

    int cell = n + bodyIndex;
    int ind = *index;
    
    while ((cell + offset) < ind) {
        
        s = start[cell + offset];

        if (s >= 0) {

            for(int i=0;i<8;i++) {
                
                int node = child[8*(cell+offset) + i];

                if (node >= n) {  // not a leaf node
                    start[node] = s;
                    s += count[node];
                }
                else if (node >= 0) {  // leaf node
                    sorted[s] = node;
                    s++;
                }
            }
            offset += stride;
        }
    }
}


// Kernel 5: computes the forces
__global__ void computeForcesKernel(float* x, float *y, float *z, float *vx, float *vy, float *vz, 
                                    float *ax, float *ay, float *az, float *mass,
                                    int *sorted, int *child, float *minX, float *maxX, int n, float g)
{
    int bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x*gridDim.x;
    int offset = 0;

    __shared__ float depth[stackSize*blockSize/warp];
    __shared__ int stack[stackSize*blockSize/warp];  // stack controlled by one thread per warp

    float radius = 0.5*(*minX - (*maxX));

    // need this in case some of the first eight entries of child are -1 (otherwise jj = 7)
    int jj = -1;
    for (int i=0;i<8;i++) {
        if (child[i] != -1) {
            jj++;
        }
    }

    int counter = threadIdx.x % warp;
    int stackStartIndex = stackSize*(threadIdx.x / warp);
    
    while (bodyIndex + offset < n) {
        
        int sortedIndex = sorted[bodyIndex + offset];

        float pos_x = x[sortedIndex];
        float pos_y = y[sortedIndex];
        float pos_z = z[sortedIndex];
        
        float acc_x = 0.0;
        float acc_y = 0.0;
        float acc_z = 0.0;

        // initialize stack
        int top = jj + stackStartIndex;
        
        if (counter == 0) {
            
            int temp = 0;
            
            for (int i=0;i<4;i++) {
                if (child[i] != -1) {
                    stack[stackStartIndex + temp] = child[i];
                    depth[stackStartIndex + temp] = radius*radius/theta;
                    temp++;
                }
            }
        }

        __syncthreads();

        // while stack is not empty
        while (top >= stackStartIndex) {
            
            int node = stack[top];
            float dp = 0.25*depth[top]; // float dp = depth[top];
            
            for (int i=0; i<8; i++) {
                
                int ch = child[8*node + i];

                //__threadfence();

                if (ch >= 0) {
                    
                    float dx = x[ch] - pos_x;
                    float dy = y[ch] - pos_y;
                    float dz = z[ch] - pos_z;
                    
                    float r = dx*dx + dy*dy + dz*dz + eps_squared;

                    if (ch < n /*is leaf node*/ || __all(dp <= r)/*meets criterion*/) {
                        r = rsqrt(r);
                        float f = mass[ch] * r * r * r;

                        acc_x += f*dx;
                        acc_y += f*dy;
                        acc_z += f*dz;
                    }
                    else{
                        if(counter == 0){
                            stack[top] = ch;
                            depth[top] = dp;
                            // depth[top] = 0.25*dp;
                        }
                        top++;
                        //__threadfence();
                    }
                }
            }

            top--;
        }

        ax[sortedIndex] = acc_x;
        ay[sortedIndex] = acc_y;
        az[sortedIndex] = acc_z;

        offset += stride;

        __syncthreads();
    }
}


// Kernel 6: updates the bodies
__global__ void updateKernel(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                             float *ax, float *ay, float *az, int n, float dt, float d) {

    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    while (bodyIndex + offset < n) {

        vx[bodyIndex + offset] += dt * ax[bodyIndex + offset];
        vy[bodyIndex + offset] += dt * ay[bodyIndex + offset];
        vz[bodyIndex + offset] += dt * az[bodyIndex + offset];

        x[bodyIndex + offset] += d * dt * vx[bodyIndex + offset];
        y[bodyIndex + offset] += d * dt * vy[bodyIndex + offset];
        z[bodyIndex + offset] += d * dt * vz[bodyIndex + offset];

        offset += stride;
    }
}



__global__ void copyKernel(float *x, float *y, float *z, float *out, int n) {
    int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    while (bodyIndex + offset < n) {

        out[2 * (bodyIndex + offset) + 0] = x[bodyIndex + offset];
        out[2 * (bodyIndex + offset) + 1] = y[bodyIndex + offset];
        out[2 * (bodyIndex + offset) + 2] = z[bodyIndex + offset];

        offset += stride;
    }
}