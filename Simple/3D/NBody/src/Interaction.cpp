//
// Created by Michael Staneker on 25.01.21.
//

#include "../include/Interaction.h"

Interaction::Interaction() : frictionEnabled { true } { }

Interaction::Interaction(bool _frictionEnabled) : frictionEnabled { _frictionEnabled } { }

void Interaction::singleInteraction(Body* body1, Body* body2, bool symmetric) {
    Vector3D positionDifference;
    positionDifference.x = (body1->position.x - body2->position.x) * TO_METERS;
    positionDifference.y = (body1->position.y - body2->position.y) * TO_METERS;
    positionDifference.z = (body1->position.z - body2->position.z) * TO_METERS;
    double distance = positionDifference.magnitude();

    if (distance == 0) {
        return;
    }

    double force = TIME_STEP * (G * body1->mass * body2->mass) /
                    ((distance*distance + SOFTENING*SOFTENING) * distance);

    body1->acceleration.x -= force * positionDifference.x / body1->mass;
    body1->acceleration.y -= force * positionDifference.y / body1->mass;
    body1->acceleration.z -= force * positionDifference.z / body1->mass;

    if (symmetric) {
        body2->acceleration.x += force * positionDifference.x / body2->mass;
        body2->acceleration.y += force * positionDifference.y / body2->mass;
        body2->acceleration.z += force * positionDifference.z / body2->mass;
    }

}

/**
void Interaction::singleInteractionSymmetric(Body* body1, Body* body2)
{
    Vector3D positionDifference;
    positionDifference.x = (body1->position.x-body2->position.x)*TO_METERS;
    positionDifference.y = (body1->position.y-body2->position.y)*TO_METERS;
    positionDifference.z = (body1->position.z-body2->position.z)*TO_METERS;

    double distance = positionDifference.magnitude();
    double force = TIME_STEP * (G * body1->mass * body2->mass) /
                   ((distance*distance + SOFTENING*SOFTENING) * distance);

    body1->acceleration.x -= force * positionDifference.x / body1->mass;
    body1->acceleration.y -= force * positionDifference.y / body1->mass;
    body1->acceleration.z -= force * positionDifference.z / body1->mass;
    body2->acceleration.x += force * positionDifference.x / body2->mass;
    body2->acceleration.y += force * positionDifference.y / body2->mass;
    body2->acceleration.z += force * positionDifference.z / body2->mass;
}
 **/


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


void Interaction::interactBodies(Body* bods)
{
    // Sun interacts individually
    if (DEBUG_INFO) {std::cout << "\nCalculating Force from star..." << std::flush;}
    Body *sun = &bods[0];
    #pragma omp parallel for
    for (int bIndex=1; bIndex<NUM_BODIES; bIndex++)
    {
        singleInteraction(sun, &bods[bIndex], true);
    }

    if (DEBUG_INFO) {std::cout << "\nBuilding Octree..." << std::flush;}

    // Build tree
    Octant&& proot = Octant(0.0, // center x
                            0.0, // center y
                            0.0, //0.1374,
                            60.0*SYSTEM_SIZE);
    Tree *tree = new Tree(std::move(proot));


    for (int bIndex=1; bIndex<NUM_BODIES; bIndex++)
    {
        if (tree->getOctant().contains(bods[bIndex].position))
        {
            tree->insert(&bods[bIndex]);
        }
    }

    if (DEBUG_INFO) {std::cout << "\nCalculating particle-particle interactions..." << std::flush;}

    // loop through interactions
    #pragma omp parallel for
    for (int bIndex=1; bIndex<NUM_BODIES; bIndex++)
    {
        if (tree->getOctant().contains(bods[bIndex].position))
        {
            treeInteraction(tree, &bods[bIndex]);
        }
    }

    delete tree;

    updateBodies(bods);
}


void Interaction::updateBodies(Body* bods)
{
    if (DEBUG_INFO) {std::cout << "\nUpdating particle positions..." << std::flush;}
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
                std::cout   << std::endl
                            << "  Star x acceleration: " << current->acceleration.x
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

    //if (DEBUG_INFO)
    //{
    //    std::cout << "\nMass below: " << mBelow << " Mass Above: "
    //              << mAbove << " \nRatio: " << mBelow/mAbove;
    //}
}

