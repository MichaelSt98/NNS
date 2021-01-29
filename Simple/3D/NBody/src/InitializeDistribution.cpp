//
// Created by Michael Staneker on 27.01.21.
//

#include "../include/InitializeDistribution.h"

InitializeDistribution::InitializeDistribution() {
    LOGCFG.headers = true;
    LOGCFG.level = DEBUG;
}

void InitializeDistribution::starParticleDisk(Body* suns, Body* bods)
{
    using std::uniform_real_distribution;
    uniform_real_distribution<double> randAngle (0.0, 200.0*PI);
    uniform_real_distribution<double> randRadius (INNER_BOUND, SYSTEM_SIZE);
    uniform_real_distribution<double> randHeight (0.0, SYSTEM_THICKNESS);
    std::default_random_engine gen (0);
    double angle;
    double radius;
    double velocity;
    Body *currentSun;

    currentSun = &suns[0];
    currentSun->position.x = 0.0;
    currentSun->position.y = 0.0;
    currentSun->position.z = 0.0;
    currentSun->velocity.x = 0.0;
    currentSun->velocity.y = 0.0;
    currentSun->velocity.z = 0.0;
    currentSun->mass = SOLAR_MASS;

    Body *current;
    double totalExtraMass = 0.0;
    for (int index=0; index<NUM_BODIES; index++)
    {
        angle = randAngle(gen);
        radius = sqrt(SYSTEM_SIZE)*sqrt(randRadius(gen));
        velocity = pow(((G*(SOLAR_MASS+((radius-INNER_BOUND)/SYSTEM_SIZE)*EXTRA_MASS*SOLAR_MASS))
                        / (radius*TO_METERS)), 0.5);

        current = &bods[index];
        current->position.x =  radius*cos(angle);
        //std::cout << "x = " << radius*cos(angle) << std::endl;
        current->position.y =  radius*sin(angle);
        current->position.z =  randHeight(gen)-SYSTEM_THICKNESS/2;
        current->velocity.x =  velocity*sin(angle);
        current->velocity.y = -velocity*cos(angle);
        current->velocity.z =  0.0;
        current->mass = (EXTRA_MASS*SOLAR_MASS)/NUM_BODIES;
        totalExtraMass += (EXTRA_MASS*SOLAR_MASS)/NUM_BODIES;
    }

    Logger(DEBUG) << "Star mass:       " << SOLAR_MASS;
    Logger(DEBUG) << "Particle weight: " << (EXTRA_MASS*SOLAR_MASS)/NUM_BODIES;
    Logger(DEBUG) << "Total disk mass: " << totalExtraMass;
    Logger(DEBUG) << "______________________________";
}

void InitializeDistribution::binaryParticleDisk(Body* suns, Body* bods)
{
    using std::uniform_real_distribution;
    uniform_real_distribution<double> randAngle (0.0, 200.0*PI);
    uniform_real_distribution<double> randRadius (INNER_BOUND, SYSTEM_SIZE);
    uniform_real_distribution<double> randHeight (0.0, SYSTEM_THICKNESS);
    std::default_random_engine gen (0);
    double angle;
    double radius;
    double velocity;
    Body *currentSun;

    //STARS
    velocity = 0.67*sqrt((G*SOLAR_MASS)/(4*BINARY_SEPARATION*TO_METERS));
    //STAR 1
    currentSun = &suns[0];
    currentSun->position.x = BINARY_SEPARATION;
    currentSun->position.y = 0.0;
    currentSun->position.z = 0.0;
    currentSun->velocity.x = 0.0;
    currentSun->velocity.y = velocity;
    currentSun->velocity.z = 0.0;
    currentSun->mass = SOLAR_MASS;
    //STAR 2
    currentSun = suns + 1;
    currentSun->position.x = BINARY_SEPARATION;
    currentSun->position.y = 0.0;
    currentSun->position.z = 0.0;
    currentSun->velocity.x = 0.0;
    currentSun->velocity.y = -velocity;
    currentSun->velocity.z = 0.0;
    currentSun->mass = SOLAR_MASS;

    Body *current;

    ///STARTS AT NUMBER OF STARS///
    double totalExtraMass = 0.0;
    for (int index=1; index<NUM_BODIES; index++)
    {
        angle = randAngle(gen);
        radius = sqrt(SYSTEM_SIZE)*sqrt(randRadius(gen));
        velocity = pow(((G*(SOLAR_MASS+((radius-INNER_BOUND)/SYSTEM_SIZE)*EXTRA_MASS*SOLAR_MASS))
                        / (radius*TO_METERS)), 0.5);
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
