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

#include <fenv.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <random>
#include <omp.h>

Renderer renderer { NUM_BODIES, WIDTH, HEIGHT, RENDER_SCALE, MAX_VEL_COLOR, MIN_VEL_COLOR,
                    PARTICLE_BRIGHTNESS, PARTICLE_SHARPNESS, DOT_SIZE,
                    SYSTEM_SIZE, RENDER_INTERVAL};

Interaction interactionHandler { false };

structlog LOGCFG = {};

void runSimulation(Body* b, char* image, double* hdImage);

void runSimulation(Body* b, char* image, double* hdImage)
{
    Timer stepTimer;

    renderer.createFrame(image, hdImage, b, 1);
    for (int step=1; step<STEP_COUNT; step++)
    {
        stepTimer.reset();

        Logger(INFO) << "Timestep: " << step;

        interactionHandler.interactBodies(b);

        if (step%renderer.getRenderInterval()==0)
        {
            renderer.createFrame(image, hdImage, b, step + 1);
        }
        Logger(INFO) << "-------------- finished timestep: " << step << " in " << stepTimer.elapsed() << " s";
    }
}


int main()
{
    LOGCFG.headers = true;
    LOGCFG.level = INFO;

    Logger(INFO) << SYSTEM_THICKNESS << "AU thick disk";

    char *image = new char[WIDTH*HEIGHT*3];
    double *hdImage = new double[WIDTH*HEIGHT*3];

    Body *bodies = new Body[NUM_BODIES];

    InitializeDistribution::starParticleDisk(bodies);

    runSimulation(bodies, image, hdImage);

    Logger(INFO) << "FINISHED!";

    delete[] bodies;
    delete[] image;

    return 0;
}
