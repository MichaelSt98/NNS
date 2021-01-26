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

#include <fenv.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <random>
#include <omp.h>

Renderer renderer;
Interaction interactionHandler { false };

void initializeBodies(Body* bods);
void runSimulation(Body* b, char* image, double* hdImage);
void interactBodies(Body* b);
void singleInteraction(Body* a, Body* b);
double magnitude(const Vector3D& v);
void updateBodies(Body* b);

void initializeBodies(Body* bods)
{
    using std::uniform_real_distribution;
    uniform_real_distribution<double> randAngle (0.0, 200.0*PI);
    uniform_real_distribution<double> randRadius (INNER_BOUND, SYSTEM_SIZE);
    uniform_real_distribution<double> randHeight (0.0, SYSTEM_THICKNESS);
    std::default_random_engine gen (0);
    double angle;
    double radius;
    double velocity;
    Body *current;

    //STARS
    velocity = 0.67*sqrt((G*SOLAR_MASS)/(4*BINARY_SEPARATION*TO_METERS));
    //STAR 1
    current = &bods[0];
    current->position.x = 0.0;///-BINARY_SEPARATION;
    current->position.y = 0.0;
    current->position.z = 0.0;
    current->velocity.x = 0.0;
    current->velocity.y = 0.0;//velocity;
    current->velocity.z = 0.0;
    current->mass = SOLAR_MASS;
    //STAR 2
    /*
    current = bods + 1;
    current->position.x = BINARY_SEPARATION;
    current->position.y = 0.0;
    current->position.z = 0.0;
    current->velocity.x = 0.0;
    current->velocity.y = -velocity;
    current->velocity.z = 0.0;
    current->= SOLAR_MASS;
    */

    ///STARTS AT NUMBER OF STARS///
    double totalExtraMass = 0.0;
    for (int index=1; index<NUM_BODIES; index++)
    {
        angle = randAngle(gen);
        radius = sqrt(SYSTEM_SIZE)*sqrt(randRadius(gen));
        velocity = pow(((G*(SOLAR_MASS+((radius-INNER_BOUND)/SYSTEM_SIZE)*EXTRA_MASS*SOLAR_MASS))
                        / (radius*TO_METERS)), 0.5);
        //velocity = pow(((G*(SOLAR_MASS+((radius-INNER_BOUND)/SYSTEM_SIZE)*EXTRA_MASS*SOLAR_MASS))
          //              / (radius*TO_METERS)), 0.5);
        current = &bods[index];
        current->position.x =  radius*cos(angle);
        current->position.y =  radius*sin(angle);
        current->position.z =  randHeight(gen)-SYSTEM_THICKNESS/2;
        current->velocity.x =  velocity*sin(angle);
        current->velocity.y = -velocity*cos(angle);
        current->velocity.z =  0.0;
        current->mass = (EXTRA_MASS*SOLAR_MASS)/NUM_BODIES;
        totalExtraMass += (EXTRA_MASS*SOLAR_MASS)/NUM_BODIES;
    }
    std::cout << "\nTotal Disk Mass: " << totalExtraMass;
    std::cout << "\nEach Particle weight: " << (EXTRA_MASS*SOLAR_MASS)/NUM_BODIES
              << "\n______________________________\n";
}

void runSimulation(Body* b, char* image, double* hdImage)
{
    renderer.createFrame(image, hdImage, b, 1);
    for (int step=1; step<STEP_COUNT; step++)
    {
        std::cout << "\nBeginning timestep: " << step;
        interactBodies(b);

        if (step%RENDER_INTERVAL==0)
        {
            renderer.createFrame(image, hdImage, b, step + 1);
        }
        if (DEBUG_INFO) {std::cout << "\n-------Done------- timestep: "
                                   << step << "\n" << std::flush;}
    }
}

void interactBodies(Body* bods)
{
    // Sun interacts individually
    if (DEBUG_INFO) {std::cout << "\nCalculating Force from star..." << std::flush;}
    Body *sun = &bods[0];
    #pragma omp parallel for
    for (int bIndex=1; bIndex<NUM_BODIES; bIndex++)
    {
        singleInteraction(sun, &bods[bIndex]);
    }

    if (DEBUG_INFO) {std::cout << "\nBuilding Octree..." << std::flush;}

    // Build tree
    Octant&& proot = Octant(0.0, /// center x
                            0.0, /// center y
                            0.1374, /// center z Does this help?
                            60.0*SYSTEM_SIZE);
    Tree *tree = new Tree(std::move(proot));

    //std::cout << "Hallo" << std::endl;
    //std::cout << "Num bodies: " << NUM_BODIES << std::endl;

    for (int bIndex=1; bIndex<NUM_BODIES; bIndex++)
    {
        //std::cout << "bIndex: " << bIndex << std::endl;
        //tree->print();
        if (tree->getOctant().contains(bods[bIndex].position))
        {
            //std::cout << "within" << std::endl;
            tree->insert(&bods[bIndex]);
        }
    }

    std::cout << "There" << std::endl;

    if (DEBUG_INFO) {std::cout << "\nCalculating particle interactions..." << std::flush;}

    // loop through interactions
    #pragma omp parallel for
    for (int bIndex=1; bIndex<NUM_BODIES; bIndex++)
    {
        if (tree->getOctant().contains(bods[bIndex].position))
        {
            interactionHandler.treeInteraction(tree, &bods[bIndex]);
        }
    }
    // Destroy tree
    delete tree;
    //
    if (DEBUG_INFO) {std::cout << "\nUpdating particle positions..." << std::flush;}
    updateBodies(bods);
}

void singleInteraction(Body* a, Body* b)
{
    Vector3D posDiff;
    posDiff.x = (a->position.x-b->position.x)*TO_METERS;
    posDiff.y = (a->position.y-b->position.y)*TO_METERS;
    posDiff.z = (a->position.z-b->position.z)*TO_METERS;

    double dist = magnitude(posDiff);
    double F = TIME_STEP*(G*a->mass*b->mass) / ((dist*dist + SOFTENING*SOFTENING) * dist);

    a->acceleration.x -= F*posDiff.x/a->mass;
    a->acceleration.y -= F*posDiff.y/a->mass;
    a->acceleration.z -= F*posDiff.z/a->mass;
    b->acceleration.x += F*posDiff.x/b->mass;
    b->acceleration.y += F*posDiff.y/b->mass;
    b->acceleration.z += F*posDiff.z/b->mass;
}

double magnitude(const Vector3D& v)
{
    return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

void updateBodies(Body* bods)
{
    double mAbove = 0.0;
    double mBelow = 0.0;
    #pragma omp for
    for (int bIndex=0; bIndex<NUM_BODIES; bIndex++)
    {
        Body *current = &bods[bIndex];
        if (DEBUG_INFO)
        {
            if (bIndex==0)
            {
                std::cout << "\nStar x acceleration: " << current->acceleration.x
                          << "  Star y acceleration: " << current->acceleration.y;
            } else if (current->position.y > 0.0)
            {
                mAbove += current->mass;
            } else {
                mBelow += current->mass;
            }
        }
        current->velocity.x += current->acceleration.x;
        current->velocity.y += current->acceleration.y;
        current->velocity.z += current->acceleration.z;
        current->acceleration.x = 0.0;
        current->acceleration.y = 0.0;
        current->acceleration.z = 0.0;
        current->position.x += TIME_STEP*current->velocity.x/TO_METERS;
        current->position.y += TIME_STEP*current->velocity.y/TO_METERS;
        current->position.z += TIME_STEP*current->velocity.z/TO_METERS;
    }
    if (DEBUG_INFO)
    {
        std::cout << "\nMass below: " << mBelow << " Mass Above: "
                  << mAbove << " \nRatio: " << mBelow/mAbove;
    }
}


int main()
{
    std::cout << SYSTEM_THICKNESS << "AU thick disk\n";;
    char *image = new char[WIDTH*HEIGHT*3];
    double *hdImage = new double[WIDTH*HEIGHT*3];
    Body *bodies = new Body[NUM_BODIES];

    initializeBodies(bodies);
    runSimulation(bodies, image, hdImage);
    std::cout << "\nwe made it\n";
    delete[] bodies;
    delete[] image;

    return 0;
}
