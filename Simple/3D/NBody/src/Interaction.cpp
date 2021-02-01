//
// Created by Michael Staneker on 25.01.21.
//

#include "../include/Interaction.h"

Interaction::Interaction() : frictionEnabled { true } {
    LOGCFG.headers = true;
    LOGCFG.level = INFO;
}

Interaction::Interaction(bool _frictionEnabled) : frictionEnabled { _frictionEnabled } {
    LOGCFG.headers = true;
    LOGCFG.level = INFO;
}

void Interaction::singleInteraction(Body* body1, Body* body2, bool symmetric, bool debug) {
    Vector3D positionDifference;
    positionDifference.x = (body1->position.x - body2->position.x) * TO_METERS;
    positionDifference.y = (body1->position.y - body2->position.y) * TO_METERS;
    positionDifference.z = (body1->position.z - body2->position.z) * TO_METERS;
    double distance = positionDifference.magnitude();

    if (debug) {
        Logger(DEBUG) << "positionDiffMag: " << distance;
    }

    if (distance == 0) {
        return;
    }

    double force = TIME_STEP * (G * body1->mass * body2->mass) /
                    ((distance*distance + SOFTENING*SOFTENING) * distance);

    if (debug) {
        Logger(DEBUG) << "Force: " << force;
    }

    body1->acceleration.x -= force * positionDifference.x / body1->mass;
    body1->acceleration.y -= force * positionDifference.y / body1->mass;
    body1->acceleration.z -= force * positionDifference.z / body1->mass;

    if (debug) {
        Logger(DEBUG) << "\tAcceleration x: " << body1->acceleration.x;
        Logger(DEBUG) << "\tAcceleration y: " << body1->acceleration.y;
        Logger(DEBUG) << "\tAcceleration z: " << body1->acceleration.z;
    }

    if (symmetric) {
        body2->acceleration.x += force * positionDifference.x / body2->mass;
        body2->acceleration.y += force * positionDifference.y / body2->mass;
        body2->acceleration.z += force * positionDifference.z / body2->mass;
    }

}


void Interaction::treeInteraction(Tree *tree, Body *body) {

    if (tree->isExternal()) {
        Body *treeBody = &tree->centerOfMass;
        singleInteraction(body, treeBody, false);
    }
    else if (tree->getOctant().getLength() /
            Vector3D::magnitude(tree->centerOfMass.position.x - body->position.x,
                                tree->centerOfMass.position.y - body->position.y,
                                tree->centerOfMass.position.z - body->position.z) < MAX_DISTANCE) {

        Body *treeBody = &tree->centerOfMass;
        singleInteraction(body, treeBody, false);
    }
    else {
        if (tree->UNW != NULL) {
            treeInteraction(tree->UNW, body);
        }
        if (tree->UNE != NULL) {
            treeInteraction(tree->UNE, body);
        }
        if (tree->USW != NULL) {
            treeInteraction(tree->USW, body);
        }
        if (tree->USE != NULL) {
            treeInteraction(tree->USE, body);
        }
        if (tree->LNW != NULL) {
            treeInteraction(tree->LNW, body);
        }
        if (tree->LNE != NULL) {
            treeInteraction(tree->LNE, body);
        }
        if (tree->LSW != NULL) {
            treeInteraction(tree->LSW, body);
        }
        if (tree->LSE != NULL) {
            treeInteraction(tree->LSE, body);
        }
    }
}


void Interaction::interactBodies(Body* suns, Body* bods)
{

    Logger(DEBUG) << "Calculating force from star(s) ...";

    for (int sIndex=0; sIndex<NUM_SUNS; sIndex++)
    {
        for (int ssIndex = 0; ssIndex<NUM_SUNS; ssIndex++) {
            if (sIndex != ssIndex) {
                singleInteraction(&suns[sIndex], &suns[ssIndex], false, false);
            }
        }


        //#pragma omp parallel for
        for (int bIndex=0; bIndex<NUM_BODIES; bIndex++)
        {
            singleInteraction(&suns[sIndex], &bods[bIndex], true);
        }
    }

    Logger(DEBUG) << "Building Octree ...";

    Octant&& proot = Octant(0.0, // center x
                            0.0, // center y
                            0.0, // center z
                            60.0*SYSTEM_SIZE);

    Tree *tree = new Tree(std::move(proot));

    for (int bIndex=0; bIndex<NUM_BODIES; bIndex++)
    {
        if (tree->getOctant().contains(bods[bIndex].position))
        {
            tree->insert(&bods[bIndex]);
        }
    }

    Logger(DEBUG) << "Calculating particle-particle interactions ...";

    #pragma omp parallel for
    for (int bIndex=0; bIndex<NUM_BODIES; bIndex++)
    {
        if (tree->getOctant().contains(bods[bIndex].position))
        {
            treeInteraction(tree, &bods[bIndex]);
        }
    }

    delete tree;

    updateBodies(suns, bods);

}


void Interaction::updateBodies(Body* suns, Body* bods)
{
    Logger(DEBUG) << "Updating particle positions ...";

    //#pragma omp for
    for (int sIndex=0; sIndex<NUM_SUNS; sIndex++)
    {
        Body *currentSun = &suns[sIndex];

        currentSun->velocity.x += currentSun->acceleration.x;
        currentSun->velocity.y += currentSun->acceleration.y;
        currentSun->velocity.z += currentSun->acceleration.z;
        currentSun->acceleration.x = 0.0;
        currentSun->acceleration.y = 0.0;
        currentSun->acceleration.z = 0.0;
        currentSun->position.x += TIME_STEP*currentSun->velocity.x/TO_METERS;
        currentSun->position.y += TIME_STEP*currentSun->velocity.y/TO_METERS;
        currentSun->position.z += TIME_STEP*currentSun->velocity.z/TO_METERS;
    }

    //#pragma omp for
    for (int bIndex=0; bIndex<NUM_BODIES; bIndex++) {

        Body *current = &bods[bIndex];

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

}

