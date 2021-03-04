#include "../include/Body.h"
#include "../include/Constants.h"
#include "../include/Renderer.h"
#include "../include/Logger.h"
#include "../include/Timer.h"
#include "../include/KernelsWrapper.cuh"
#include "../include/BarnesHut.cuh"
#include "../include/cxxopts.h"

#include <fenv.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <random>

structlog LOGCFG = {};

int main(int argc, char** argv)
{

    cxxopts::Options options("MyProgram", "One line description of MyProgram");

    bool render = false;

    options.add_options()
            ("r,render", "render simulation", cxxopts::value<bool>(render));

    auto result = options.parse(argc, argv);

    //render = result["render"].as<bool>();

    /** Initialization */
    SimulationParameters parameters;

    parameters.iterations = 500;
    parameters.numberOfParticles = 512*256*4;
    parameters.timestep = 0.001;
    parameters.gravity = 1.0;
    parameters.dampening = 1.0;
    parameters.gridSize = 1024;
    parameters.blockSize = 256;
    parameters.warp = 32;
    parameters.stackSize = 64;



    LOGCFG.headers = true;
    LOGCFG.level = DEBUG; //INFO;

    char *image = new char[WIDTH*HEIGHT*3];
    double *hdImage = new double[WIDTH*HEIGHT*3];

    Body *suns = new Body [1];
    Body *bodies = new Body[parameters.numberOfParticles];

    BarnesHut *particles = new BarnesHut(parameters);

    Renderer renderer { parameters.numberOfParticles, WIDTH, HEIGHT, RENDER_SCALE, MAX_VEL_COLOR, MIN_VEL_COLOR,
                        PARTICLE_BRIGHTNESS, PARTICLE_SHARPNESS, DOT_SIZE,
                        2*particles->getSystemSize(), RENDER_INTERVAL };

    /** Simulation */
    for(int i = 0 ; i < parameters.iterations ; i++){

        particles->update(i);

        /**
         * Output
         * * optimize (not yet optimized for code structure)
         * */
        if (render) {
            for (int i_body = 0; i_body < parameters.numberOfParticles; i_body++) {

                Body *current;
                current = &bodies[i_body];
                current->position.x = particles->h_x[i_body];
                current->position.y = particles->h_y[i_body];
                current->position.z = particles->h_z[i_body];
                current->velocity.x = particles->h_vx[i_body];
                current->velocity.y = particles->h_vy[i_body];
                current->velocity.z = particles->h_vz[i_body];
            }
            if (i % RENDER_INTERVAL == 0) {
                renderer.createFrame(image, hdImage, bodies, i);
            }
        }
    }

    /** Postprocessing */
    float total_time_resetArrays = 0.f;
    float total_time_computeBoundingBox = 0.f;
    float total_time_buildTree = 0.f;
    float total_time_centreOfMass = 0.f;
    float total_time_sort = 0.f;
    float total_time_computeForces = 0.f;
    float total_time_update = 0.f;
    float total_time_copyDeviceToHost = 0.f;
    float total_time_all = 0.f;

    for (int i = 0; i < parameters.iterations; i++) {

        total_time_resetArrays += particles->time_resetArrays[i];
        total_time_computeBoundingBox += particles->time_computeBoundingBox[i];
        total_time_buildTree += particles->time_buildTree[i];
        total_time_centreOfMass += particles->time_centreOfMass[i];
        total_time_sort += particles->time_sort[i];
        total_time_computeForces += particles->time_computeForces[i];
        total_time_update += particles->time_update[i];
        total_time_copyDeviceToHost += particles->time_copyDeviceToHost[i];
        total_time_all += particles->time_all[i];

    }

    Logger(INFO) << "Time to reset arrays: " << total_time_resetArrays << "ms";
    Logger(INFO) << "\tper step: " << total_time_resetArrays/parameters.iterations << "ms";

    Logger(INFO) << "Time to compute bounding boxes: " << total_time_computeBoundingBox << "ms";
    Logger(INFO) << "\tper step: " << total_time_computeBoundingBox/parameters.iterations << "ms";

    Logger(INFO) << "Time to build tree: " << total_time_buildTree << "ms";
    Logger(INFO) << "\tper step: " << total_time_buildTree/parameters.iterations << "ms";

    Logger(INFO) << "Time to compute COM: " << total_time_centreOfMass << "ms";
    Logger(INFO) << "\tper step: " << total_time_centreOfMass/parameters.iterations << "ms";

    Logger(INFO) << "Time to sort: " << total_time_sort << "ms";
    Logger(INFO) << "\tper step: " << total_time_sort/parameters.iterations << "ms";

    Logger(INFO) << "Time to compute forces: " << total_time_computeForces << "ms";
    Logger(INFO) << "\tper step: " << total_time_computeForces/parameters.iterations << "ms";

    Logger(INFO) << "Time to update bodies: " << total_time_update << "ms";
    Logger(INFO) << "\tper step: " << total_time_update/parameters.iterations << "ms";

    Logger(INFO) << "Time to copy from device to host: " << total_time_copyDeviceToHost << "ms";
    Logger(INFO) << "\tper step: " << total_time_copyDeviceToHost/parameters.iterations << "ms";

    Logger(INFO) << "----------------------------------------------";
    Logger(INFO) << "TOTAL TIME: " << total_time_all << "ms";
    Logger(INFO) << "\tper step: " << total_time_all/parameters.iterations << "ms";
    Logger(INFO) << "TOTAL TIME (without copying): " << total_time_all-total_time_copyDeviceToHost << "ms";
    Logger(INFO) << "\tper step: " << (total_time_all-total_time_copyDeviceToHost)/parameters.iterations << "ms";
    Logger(INFO) << "----------------------------------------------";

    /** Cleaning */
    delete[] image;

    return 0;
}
