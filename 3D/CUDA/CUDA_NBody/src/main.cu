#include "../include/Body.h"
#include "../include/Constants.h"
#include "../include/Renderer.h"
#include "../include/Vector3D.h"
#include "../include/Utils.h"
#include "../include/Logger.h"
#include "../include/Timer.h"
#include "../include/KernelsWrapper.cuh"
#include "../include/BarnesHut.cuh"

#include <fenv.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <random>

Renderer renderer { NUM_SUNS, NUM_BODIES, WIDTH, HEIGHT, RENDER_SCALE, MAX_VEL_COLOR, MIN_VEL_COLOR,
                    PARTICLE_BRIGHTNESS, PARTICLE_SHARPNESS, DOT_SIZE,
                    SYSTEM_SIZE, RENDER_INTERVAL};

structlog LOGCFG = {};

int main()
{
    /** Initialization */
    SimulationParameters parameters;

    parameters.iterations = 500;
    parameters.timestep = 0.001;
    parameters.gravity = 1.0;
    parameters.dampening = 1.0;

    LOGCFG.headers = true;
    LOGCFG.level = DEBUG; //INFO;

    Logger(INFO) << SYSTEM_THICKNESS << "AU thick disk";

    char *image = new char[WIDTH*HEIGHT*3];
    double *hdImage = new double[WIDTH*HEIGHT*3];

    Body *suns = new Body [NUM_SUNS];
    Body *bodies = new Body[NUM_BODIES];

    BarnesHut *particles = new BarnesHut(parameters);

    /** Simulation */
    for(int i = 0 ; i < parameters.iterations ; i++){

        particles->update(i);

        /**
         * Output
         * * optimize (not yet optimized for code structure)
         * */
        for (int i_body = 0; i_body < NUM_BODIES; i_body++) {

            Body *current;
            current = &bodies[i_body];
            current->position.x =  particles->h_x[i_body];
            current->position.y =  particles->h_y[i_body];
            current->position.z =  particles->h_z[i_body];
            current->velocity.x =  particles->h_vx[i_body];
            current->velocity.y =  particles->h_vy[i_body];
            current->velocity.z =  particles->h_vz[i_body];
        }
        //renderer.createFrame(image, hdImage, suns, bodies, i);
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
    Logger(INFO) << "\nper step: " << total_time_resetArrays/parameters.iterations << "ms";

    Logger(INFO) << "Time to compute bounding boxes: " << total_time_computeBoundingBox << "ms";
    Logger(INFO) << "\nper step: " << total_time_computeBoundingBox/parameters.iterations << "ms";

    Logger(INFO) << "Time to build tree: " << total_time_buildTree << "ms";
    Logger(INFO) << "\nper step: " << total_time_buildTree/parameters.iterations << "ms";

    Logger(INFO) << "Time to compute COM: " << total_time_centreOfMass << "ms";
    Logger(INFO) << "\nper step: " << total_time_centreOfMass/parameters.iterations << "ms";

    Logger(INFO) << "Time to sort: " << total_time_sort << "ms";
    Logger(INFO) << "\nper step: " << total_time_sort/parameters.iterations << "ms";

    Logger(INFO) << "Time to compute forces: " << total_time_computeForces << "ms";
    Logger(INFO) << "\nper step: " << total_time_computeForces/parameters.iterations << "ms";

    Logger(INFO) << "Time to update bodies: " << total_time_update << "ms";
    Logger(INFO) << "\nper step: " << total_time_update/parameters.iterations << "ms";

    Logger(INFO) << "Time to copy from device to host: " << total_time_copyDeviceToHost << "ms";
    Logger(INFO) << "\nper step: " << total_time_copyDeviceToHost/parameters.iterations << "ms";

    Logger(INFO) << "----------------------------------------------";
    Logger(INFO) << "TOTAL TIME: " << total_time_all << "ms";
    Logger(INFO) << "\nper step: " << total_time_all/parameters.iterations << "ms";
    Logger(INFO) << "----------------------------------------------";

    /** Cleaning */
    delete[] image;

    return 0;
}
