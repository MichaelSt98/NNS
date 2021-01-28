//
// Created by Michael Staneker on 27.01.21.
//

#include "../include/InitializeDistribution.h"

void InitializeDistribution::starParticleDisk(Body* bods)
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

    //STAR
    current = &bods[0];
    current->position.x = 0.0;
    current->position.y = 0.0;
    current->position.z = 0.0;
    current->velocity.x = 0.0;
    current->velocity.y = 0.0;
    current->velocity.z = 0.0;
    current->mass = SOLAR_MASS;

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
    std::cout   << std::endl
                << "Star mass:       " << SOLAR_MASS << std::endl
                << "Particle weight: " << (EXTRA_MASS*SOLAR_MASS)/NUM_BODIES
                << "Total disk mass: " << totalExtraMass << std::endl
                << "______________________________" << std::endl;
}