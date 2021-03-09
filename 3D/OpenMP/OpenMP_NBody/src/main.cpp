//
// Created by Michael Staneker on 25.01.21.
//

#include "../include/cxxopts.hpp"
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
#include "../include/ConfigParser.h"

#include <fenv.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <random>
#include <omp.h>

Renderer *renderer; /*{ NUM_SUNS, NUM_BODIES, WIDTH, HEIGHT, RENDER_SCALE, MAX_VEL_COLOR, MIN_VEL_COLOR,
                    PARTICLE_BRIGHTNESS, PARTICLE_SHARPNESS, DOT_SIZE,
                    SYSTEM_SIZE, RENDER_INTERVAL};*/

Interaction interactionHandler { false };

// re-declare extern global variables, i.e. they are initializen in this source file
int NUM_BODIES, NUM_SUNS, TIME_STEP, STEP_COUNT, WIDTH, HEIGHT;
double SYSTEM_SIZE, SYSTEM_THICKNESS, INNER_BOUND, SOFTENING,
             SOLAR_MASS, EXTRA_MASS, MAX_DISTANCE, BINARY_SEPARATION;
bool BINARY;

// extern variable from Logger has to be initialized here
structlog LOGCFG = {};

bool HASHED_MODE = false;

// function declarations
cxxopts::Options init(const std::string &configFile);
void runSimulation(Body* s, Body* b, char* image, double* hdImage);

// function definitions
cxxopts::Options init(const std::string &configFile) {
    // read in config file
    ConfigParser confP{ConfigParser(configFile)};

    // initialize logger
    LOGCFG.headers = true;
    LOGCFG.level = DEBUG;

    // integers
    NUM_BODIES = confP.getVal<int>("numBodies");
    NUM_SUNS = confP.getVal<int>("numSuns");
    TIME_STEP = confP.getVal<int>("timeStep");
    STEP_COUNT = confP.getVal<int>("stepCount");
    WIDTH = confP.getVal<int>("width");
    HEIGHT = confP.getVal<int>("height");

    // doubles
    SYSTEM_SIZE = confP.getVal<double>("systemSize");
    SYSTEM_THICKNESS = confP.getVal<double>("systemThickness");
    INNER_BOUND = confP.getVal<double>("innerBound");
    SOFTENING = confP.getVal<double>("softening");
    SOLAR_MASS = confP.getVal<double>("solarMass");
    EXTRA_MASS = confP.getVal<double>("extraMass");
    MAX_DISTANCE = confP.getVal<double>("maxDistance");
    BINARY_SEPARATION = confP.getVal<double>("binarySeparation");

    // bools
    BINARY = confP.getVal<bool>("binary");

    // initialize renderer
    renderer = new Renderer(
            confP.getVal<int>("numSuns"),
            confP.getVal<int>("numBodies"),
            WIDTH, HEIGHT,
            confP.getVal<double>("renderScale"),
            confP.getVal<double>("maxVelColor"),
            confP.getVal<double>("minVelColor"),
            confP.getVal<double>("particleBrightness"),
            confP.getVal<double>("particleSharpness"),
            confP.getVal<int>("dotSize"),
            confP.getVal<double>("systemSize"),
            confP.getVal<int>("renderInterval"));

    // initialize command line options parser
    cxxopts::Options opts_("runner", "NBody simulation parallelized by OpenMP");
    opts_.add_options()("H,hashed-tree", "Utilize a hashed tree.")("h,help", "Show this help");

    return opts_;
}

void runSimulation(Body* s, Body* b, char* image, double* hdImage)
{
    double stepDurations [STEP_COUNT];
    Timer stepTimer;

    renderer->createFrame(image, hdImage, s, b, 0);
    for (int step=1; step<STEP_COUNT; step++)
    {
        stepTimer.reset();

        Logger(INFO) << "Timestep: " << step;

        interactionHandler.interactBodies(s, b);

        double elapsedTime = stepTimer.elapsed();
        stepDurations[step] = elapsedTime;

        if (step%renderer->getRenderInterval()==0)
        {
            renderer->createFrame(image, hdImage, s, b, step);
        }

        Logger(INFO) << "-------------- finished timestep: " << step << " in " << elapsedTime << " s";
    }

    double totalElapsedTime = 0.0;
    for(auto& num : stepDurations)
        totalElapsedTime += num;

    Logger(INFO) << "Total elapsed time:     " << totalElapsedTime;
    Logger(INFO) << "Averaged time per step: " << totalElapsedTime/STEP_COUNT;
}

// main
int main(int argc, char **argv)
{
    Logger(INFO) << SYSTEM_THICKNESS << "AU thick disk";

    // initialization
    auto options = init("config.info");
    auto opts = options.parse(argc, argv);

    char *image = new char[WIDTH*HEIGHT*3];
    double *hdImage = new double[WIDTH*HEIGHT*3];

    Body *suns = new Body [NUM_SUNS];
    Body *bodies = new Body[NUM_BODIES];

    if (!BINARY) {
        InitializeDistribution::starParticleDisk(suns, bodies);
    }
    else {
        InitializeDistribution::binaryParticleDisk(suns, bodies);
        //InitializeDistribution::binary(suns, bodies);
    }

    if (opts.count("help")){
        Logger(INFO) << options.help();
        return 0;
    }
    if (opts.count("hashed-tree")){
        Logger(INFO) << "HASHED-TREE MODE";
        HASHED_MODE = true;
    }
    runSimulation(suns, bodies, image, hdImage);

    Logger(INFO) << "FINISHED!";

    delete[] suns;
    delete[] bodies;
    delete[] image;

    delete renderer;

    return 0;
}
