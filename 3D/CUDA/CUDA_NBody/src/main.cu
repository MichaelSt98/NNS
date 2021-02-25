//
// Created by Michael Staneker on 25.01.21.
//

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

    for(int i = 0 ; i < parameters.iterations ; i++){

        particles->update();

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
        renderer.createFrame(image, hdImage, suns, bodies, i);
    }

    delete[] image;

    return 0;
}
