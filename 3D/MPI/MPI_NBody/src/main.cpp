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
//#include <boost/mpi.hpp>

boost::mpi::environment env;
boost::mpi::communicator world;

Renderer renderer { NUM_SUNS, NUM_BODIES, WIDTH, HEIGHT, RENDER_SCALE, MAX_VEL_COLOR, MIN_VEL_COLOR,
                    PARTICLE_BRIGHTNESS, PARTICLE_SHARPNESS, DOT_SIZE,
                    SYSTEM_SIZE, RENDER_INTERVAL};

Interaction interactionHandler { false };

structlog LOGCFG = {};


void runSimulation(Body* s, Body* b, char* image, double* hdImage);

void runSimulation(Body* s, Body* b, char* image, double* hdImage)
{
    double stepDurations [STEP_COUNT];
    double elapsedTime = 0.0;
    Timer stepTimer;

    if (world.rank() == 0) {
        renderer.createFrame(image, hdImage, s, b, 0);
    }

    for (int step=1; step<STEP_COUNT; step++)
    {
        stepTimer.reset();

        if (world.rank() == 0) {
            Logger(INFO) << "Timestep: " << step;
        }

        interactionHandler.interactBodies(s, b);

        boost::mpi::all_gather(world, &b[world.rank() * interactionHandler.partSize],
                               interactionHandler.partSize, &b[0]);

        if (world.rank() == 0) {
            elapsedTime = stepTimer.elapsed();
            stepDurations[step] = elapsedTime;
        }

        if (world.rank() == 0) {
            if (step % renderer.getRenderInterval() == 0) {
                renderer.createFrame(image, hdImage, s, b, step);
            }
        }

        if (world.rank() == 0) {
            Logger(INFO) << "-------------- finished timestep: " << step << " in " << elapsedTime << " s";
        }
    }

    if (world.rank() == 0) {
        double totalElapsedTime = 0.0;
        for (auto &num : stepDurations)
            totalElapsedTime += num;

        Logger(INFO) << "Total elapsed time:     " << totalElapsedTime;
        Logger(INFO) << "Averaged time per step: " << totalElapsedTime / STEP_COUNT;
    }
}


int main(int argc, char** argv)
{

    //std::cout << "rank: " << world.rank() << " | size: " << world.size() << std::endl;

    LOGCFG.headers = true;
    LOGCFG.level = DEBUG; //INFO;

    Logger(INFO) << SYSTEM_THICKNESS << "AU thick disk";

    char *image = new char[WIDTH*HEIGHT*3];
    double *hdImage = new double[WIDTH*HEIGHT*3];

    Body *suns = new Body [NUM_SUNS];
    Body *bodies = new Body[NUM_BODIES];

    // if (world.rank() == 0) ???
    if (!BINARY) {
        InitializeDistribution::starParticleDisk(suns, bodies);
    } else {
        InitializeDistribution::binaryParticleDisk(suns, bodies);
        //InitializeDistribution::binary(suns, bodies);
    }

    runSimulation(suns, bodies, image, hdImage);

    if (world.rank() == 0) {
        Logger(INFO) << "FINISHED!";
    }

    delete[] suns;
    delete[] bodies;
    delete[] image;

    MPI_Finalize();

    return 0;
}
