//
// Created by Michael Staneker on 23.02.21.
//

#include "../include/InitDistribution.cuh"

void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) getchar();
    }
}

InitDistribution::InitDistribution(const SimulationParameters p) {

    parameters = p;
    step = 0;
    numParticles = NUM_BODIES;
    numNodes = 10 * numParticles + 12000; //2 * numParticles + 12000;

    // allocate host data
    h_min_x = new float;
    h_max_x = new float;
    h_min_y = new float;
    h_max_y = new float;
    h_min_y = new float;
    h_max_y = new float;

    h_mass = new float[numNodes];

    h_x = new float[numNodes];
    h_y = new float[numNodes];
    h_z = new float[numNodes];

    h_vx = new float[numNodes];
    h_vy = new float[numNodes];
    h_vz = new float[numNodes];

    h_ax = new float[numNodes];
    h_ay = new float[numNodes];
    h_az = new float[numNodes];

    h_child = new int[8*numNodes];
    
    h_start = new int[numNodes];
    h_sorted = new int[numNodes];
    h_count = new int[numNodes];
    h_output = new float[2*numNodes];

    // allocate device data
    gpuErrorcheck(cudaMalloc((void**)&d_min_x, sizeof(float)));
    gpuErrorcheck(cudaMalloc((void**)&d_max_x, sizeof(float)));
    gpuErrorcheck(cudaMalloc((void**)&d_min_y, sizeof(float)));
    gpuErrorcheck(cudaMalloc((void**)&d_max_y, sizeof(float)));
    gpuErrorcheck(cudaMalloc((void**)&d_min_z, sizeof(float)));
    gpuErrorcheck(cudaMalloc((void**)&d_max_z, sizeof(float)));

    gpuErrorcheck(cudaMemset(d_min_x, 0, sizeof(float)));
    gpuErrorcheck(cudaMemset(d_max_x, 0, sizeof(float)));
    gpuErrorcheck(cudaMemset(d_min_y, 0, sizeof(float)));
    gpuErrorcheck(cudaMemset(d_max_y, 0, sizeof(float)));
    gpuErrorcheck(cudaMemset(d_min_z, 0, sizeof(float)));
    gpuErrorcheck(cudaMemset(d_max_z, 0, sizeof(float)));

    gpuErrorcheck(cudaMalloc((void**)&d_mass, numNodes*sizeof(float)));

    gpuErrorcheck(cudaMalloc((void**)&d_x, numNodes*sizeof(float)));
    gpuErrorcheck(cudaMalloc((void**)&d_y, numNodes*sizeof(float)));
    gpuErrorcheck(cudaMalloc((void**)&d_z, numNodes*sizeof(float)));

    gpuErrorcheck(cudaMalloc((void**)&d_vx, numNodes*sizeof(float)));
    gpuErrorcheck(cudaMalloc((void**)&d_vy, numNodes*sizeof(float)));
    gpuErrorcheck(cudaMalloc((void**)&d_vz, numNodes*sizeof(float)));

    gpuErrorcheck(cudaMalloc((void**)&d_ax, numNodes*sizeof(float)));
    gpuErrorcheck(cudaMalloc((void**)&d_ay, numNodes*sizeof(float)));
    gpuErrorcheck(cudaMalloc((void**)&d_az, numNodes*sizeof(float)));

    gpuErrorcheck(cudaMalloc((void**)&d_index, sizeof(int)));
    gpuErrorcheck(cudaMalloc((void**)&d_child, 8*numNodes*sizeof(int)));
    gpuErrorcheck(cudaMalloc((void**)&d_start, numNodes*sizeof(int)));
    gpuErrorcheck(cudaMalloc((void**)&d_sorted, numNodes*sizeof(int)));
    gpuErrorcheck(cudaMalloc((void**)&d_count, numNodes*sizeof(int)));
    gpuErrorcheck(cudaMalloc((void**)&d_mutex, sizeof(int)));

    gpuErrorcheck(cudaMemset(d_start, -1, numNodes*sizeof(int)));
    gpuErrorcheck(cudaMemset(d_sorted, 0, numNodes*sizeof(int)));

    int memSize = sizeof(float) * 2 * numParticles;

    gpuErrorcheck(cudaMalloc((void**)&d_output, 2*numNodes*sizeof(float)));

    plummerModel(h_mass, h_x, h_y, h_z, h_vx, h_vy, h_vz, h_ax, h_ay, h_az, numParticles);

    // copy data to GPU device
    cudaMemcpy(d_mass, h_mass, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, h_vx, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, h_vy, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, h_vz, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ax, h_ax, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ay, h_ay, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_az, h_az, 2*numParticles*sizeof(float), cudaMemcpyHostToDevice);

}

InitDistribution::~InitDistribution() {
    delete h_min_x;
    delete h_max_x;
    delete h_min_y;
    delete h_max_y;
    delete h_min_z;
    delete h_max_z;

    delete [] h_mass;

    delete [] h_x;
    delete [] h_y;
    delete [] h_z;

    delete [] h_vx;
    delete [] h_vy;
    delete [] h_vz;

    delete [] h_ax;
    delete [] h_ay;
    delete [] h_az;

    delete [] h_child;
    delete [] h_start;
    delete [] h_sorted;
    delete [] h_count;
    delete [] h_output;

    gpuErrorcheck(cudaFree(d_min_x));
    gpuErrorcheck(cudaFree(d_max_x));
    gpuErrorcheck(cudaFree(d_min_y));
    gpuErrorcheck(cudaFree(d_max_y));
    gpuErrorcheck(cudaFree(d_min_z));
    gpuErrorcheck(cudaFree(d_max_z));

    gpuErrorcheck(cudaFree(d_mass));

    gpuErrorcheck(cudaFree(d_x));
    gpuErrorcheck(cudaFree(d_y));
    gpuErrorcheck(cudaFree(d_z));

    gpuErrorcheck(cudaFree(d_vx));
    gpuErrorcheck(cudaFree(d_vy));
    gpuErrorcheck(cudaFree(d_vz));

    gpuErrorcheck(cudaFree(d_ax));
    gpuErrorcheck(cudaFree(d_ay));
    gpuErrorcheck(cudaFree(d_az));

    gpuErrorcheck(cudaFree(d_index));
    gpuErrorcheck(cudaFree(d_child));
    gpuErrorcheck(cudaFree(d_start));
    gpuErrorcheck(cudaFree(d_sorted));
    gpuErrorcheck(cudaFree(d_count));

    gpuErrorcheck(cudaFree(d_mutex));

    gpuErrorcheck(cudaFree(d_output));

    cudaDeviceSynchronize();
}

void InitDistribution::update()
{
    float elapsedTime;
    cudaEventCreate(&start_global);
    cudaEventCreate(&stop_global);
    cudaEventRecord(start_global, 0);

    float elapsedTimeKernel;

    elapsedTimeKernel = kernel::resetArrays(d_mutex, d_x, d_y, d_z, d_mass, d_count, d_start, d_sorted, d_child, d_index,
                        d_min_x, d_max_x, d_min_y, d_max_y, d_min_z, d_max_z, numParticles, numNodes);

    std::cout << "\tElapsed time: " << elapsedTimeKernel << std::endl;

    elapsedTimeKernel = kernel::computeBoundingBox(d_mutex, d_x, d_y, d_z, d_min_x, d_max_x, d_min_y, d_max_y,
                               d_min_z, d_max_z, numParticles);

    std::cout << "\tElapsed time: " << elapsedTimeKernel << std::endl;

    elapsedTimeKernel = kernel::buildTree(d_x, d_y, d_z, d_mass, d_count, d_start, d_child, d_index, d_min_x, d_max_x, d_min_y, d_max_y,
                      d_min_z, d_max_z, numParticles, numNodes);

    std::cout << "\tElapsed time: " << elapsedTimeKernel << std::endl;

    elapsedTimeKernel = kernel::centreOfMass(d_x, d_y, d_z, d_mass, d_index, numParticles);

    std::cout << "\tElapsed time: " << elapsedTimeKernel << std::endl;

    elapsedTimeKernel = kernel::sort(d_count, d_start, d_sorted, d_child, d_index, numParticles);

    std::cout << "\tElapsed time: " << elapsedTimeKernel << std::endl;

    elapsedTimeKernel = kernel::computeForces(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_ax, d_ay, d_az, d_mass, d_sorted, d_child,
                          d_min_x, d_max_x, numParticles, parameters.gravity);

    std::cout << "\tElapsed time: " << elapsedTimeKernel << std::endl;

    elapsedTimeKernel = kernel::update(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_ax, d_ay, d_az, numParticles,
                   parameters.timestep, parameters.dampening);

    std::cout << "\tElapsed time: " << elapsedTimeKernel << std::endl;

    //FillOutputArray(d_x, d_y, d_output, numNodes);

    cudaEventRecord(stop_global, 0);
    cudaEventSynchronize(stop_global);
    cudaEventElapsedTime(&elapsedTime, start_global, stop_global);
    cudaEventDestroy(start_global);
    cudaEventDestroy(stop_global);

    std::cout << "Elapsed time for step " << step << " : " << elapsedTime << std::endl;

    step++;
}


void InitDistribution::plummerModel(float *mass, float *x, float* y, float *z,
                                    float *x_vel, float *y_vel, float *z_vel,
                                    float *x_acc, float *y_acc, float *z_acc, int n)
{
    float a = 1.0;
    float pi = 3.14159265;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0, 1.0);
    std::uniform_real_distribution<float> distribution2(0, 0.1);
    std::uniform_real_distribution<float> distribution_phi(0.0, 2 * pi);
    std::uniform_real_distribution<float> distribution_theta(-1.0, 1.0);

    // loop through all particles
    for (int i = 0; i < n; i++){
        float phi = distribution_phi(generator);
        float theta = acos(distribution_theta(generator));
        float r = a / sqrt(pow(distribution(generator), -0.666666) - 1);

        // set mass and position of particle
        mass[i] = 1.0;
        x[i] = r*cos(phi);
        y[i] = r*sin(phi);
        z[i] = 0.0;

        // set velocity of particle
        float s = 0.0;
        float t = 0.1;
        while(t > s*s*pow(1.0 - s*s, 3.5)){
            s = distribution(generator);
            t = distribution2(generator);
        }
        float v = 100*s*sqrt(2)*pow(1.0 + r*r, -0.25);
        phi = distribution_phi(generator);
        theta = acos(distribution_theta(generator));
        x_vel[i] = v*cos(phi);
        y_vel[i] = v*sin(phi);
        z_vel[i] = 0.0;

        // set acceleration to zero
        x_acc[i] = 0.0;
        y_acc[i] = 0.0;
        z_acc[i] = 0.0;
    }
}



