//
// Created by Michael Staneker on 25.01.21.
//

#include "../include/Body.h"
#include "../include/Constants.h"
#include "../include/Interaction.h"
#include "../include/Octant.h"
#include "../include/Renderer.h"
#include "../include/Tree.h"
#include "../include/Vector3D.h"
#include "../include/Utils.h"
#include "../include/Logger.h"
#include "../include/Timer.h"
#include "../include/InitializeDistribution.h"
#include "../include/KernelsWrapper.cuh"
#include "../include/InitDistribution.cuh"

#include <fenv.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <random>

Renderer renderer { NUM_SUNS, NUM_BODIES, WIDTH, HEIGHT, RENDER_SCALE, MAX_VEL_COLOR, MIN_VEL_COLOR,
                    PARTICLE_BRIGHTNESS, PARTICLE_SHARPNESS, DOT_SIZE,
                    SYSTEM_SIZE, RENDER_INTERVAL};

Interaction interactionHandler { false };

structlog LOGCFG = {};

/*
void runSimulation(Body* s, Body* b, char* image, double* hdImage);

void runSimulation(Body* s, Body* b, char* image, double* hdImage)
{
    double stepDurations [STEP_COUNT];
    Timer stepTimer;

    renderer.createFrame(image, hdImage, s, b, 0);
    for (int step=1; step<STEP_COUNT; step++)
    {
        stepTimer.reset();

        Logger(INFO) << "Timestep: " << step;

        interactionHandler.interactBodies(s, b);

        double elapsedTime = stepTimer.elapsed();
        stepDurations[step] = elapsedTime;

        if (step%renderer.getRenderInterval()==0)
        {
            renderer.createFrame(image, hdImage, s, b, step);
        }

        Logger(INFO) << "-------------- finished timestep: " << step << " in " << elapsedTime << " s";
    }

    double totalElapsedTime = 0.0;
    for(auto& num : stepDurations)
        totalElapsedTime += num;

    Logger(INFO) << "Total elapsed time:     " << totalElapsedTime;
    Logger(INFO) << "Averaged time per step: " << totalElapsedTime/STEP_COUNT;
}
 */


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

    InitDistribution *particles = new InitDistribution(parameters);
    //particles->reset();

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
