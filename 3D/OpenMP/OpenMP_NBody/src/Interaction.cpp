//
// Created by Michael Staneker on 25.01.21.
//

#include "../include/Interaction.h"

Interaction::Interaction() : frictionEnabled { true } {
    LOGCFG.headers = true;
    LOGCFG.level = DEBUG;
}

Interaction::Interaction(bool _frictionEnabled) : frictionEnabled { _frictionEnabled } {
    LOGCFG.headers = true;
    LOGCFG.level = DEBUG;
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


void Interaction::interactBodies(Body* suns, Body* bods, bool hashed) {

    Timer t_starInteraction;
    Timer t_buildingOctree;
    Timer t_particleParticle;
    Timer t_updateBodies;

    Logger(DEBUG) << "Calculating force from star(s) ...";

    t_starInteraction.reset();
    for (int sIndex = 0; sIndex < NUM_SUNS; sIndex++) {
    #pragma omp parallel for
        for (int ssIndex = 0; ssIndex < NUM_SUNS; ssIndex++) {
            if (sIndex != ssIndex) {
                singleInteraction(&suns[sIndex], &suns[ssIndex], false, false);
            }
        }


    #pragma omp parallel for
        for (int bIndex = 0; bIndex < NUM_BODIES; bIndex++) {
            singleInteraction(&suns[sIndex], &bods[bIndex], true);
        }
    }
    Logger(WARN) << "\tTime for star interaction: " << t_starInteraction.elapsed();

    Logger(DEBUG) << "Building Octree ...";

    // calculation x,y and z spans normed to zero
    double xMin, xMax, yMin, yMax, zMin, zMax = 0;

    for (int bIndex = 0; bIndex < NUM_BODIES; bIndex++) {
        const Body *body = &bods[bIndex];
        xMin = body->position.x < xMin ? body->position.x : xMin;
        yMin = body->position.y < yMin ? body->position.y : yMin;
        zMin = body->position.z < zMin ? body->position.z : zMin;
        xMax = body->position.x > xMax ? body->position.x : xMax;
        yMax = body->position.y > yMax ? body->position.y : yMax;
        zMax = body->position.z > xMax ? body->position.z : zMax;
    }
    Logger(INFO) << "simulationBox: [(" << xMin << ", " << yMin << ", " << zMin
                 << "), (" << xMax << ", " << yMax << ", " << zMax << ")]";
    double xSpan = std::fabs(xMax - xMin);
    double ySpan = std::fabs(yMax - yMin);
    double zSpan = std::fabs(zMax - zMin);
    double maxSpan = xSpan > ySpan ? xSpan : ySpan;
    maxSpan = maxSpan < zSpan ? zSpan : maxSpan;

    Octant proot = Octant(0., 0., 0., maxSpan);
    /*Octant proot = Octant(0.0, // center x
                   0.0, // center y
                   0.0, // center z
                   60.0 * SYSTEM_SIZE);*/

    // container for particles
    std::map<uint64_t, Body*> bodies; // unused when !hashed
    std::vector<Body*> bodsVec; // unused when !hashed

    if (hashed) {
        Logger(INFO) << "maxSpan = " << maxSpan << ", origin = (" << xMin << ", " << yMin << ", "
                     << zMin << ")";

        for (int bIndex = 0; bIndex < NUM_BODIES; bIndex++) {
            Body *body = &bods[bIndex];
            body->discretizePosition(maxSpan, Vector3D(xMin, yMin, zMin));
            bodies[body->getKey()] = body; // store bodies in hashed map
        }
        // sanity check
        if (bodies.size() != NUM_BODIES) {
            Logger(WARN) << "Some bodies seem to have been lost during position-discretization.";
            //TODO: Throw exception
        }
        // vectorize map which is ordered by key
        for (auto const &bdy: bodies){
            bodsVec.push_back(bdy.second);
        }
    }

    Tree *tree = new Tree(std::move(proot));

    t_buildingOctree.reset();
    for (int bIndex = 0; bIndex < NUM_BODIES; bIndex++) {
        if (tree->getOctant().contains(bods[bIndex].position)) {
            tree->insert(&bods[bIndex]);
        }
    }

    Logger(WARN) << "\tTime for building octree: " << t_buildingOctree.elapsed();

    Logger(DEBUG) << "Calculating particle-particle interactions ...";

    t_particleParticle.reset();
    if (hashed) {
        std::vector<Body*>::iterator it;
        #pragma omp parallel for
        for (it = bodsVec.begin(); it < bodsVec.end(); ++it) {
            if (tree->getOctant().contains((*it)->position)) {
                treeInteraction(tree, (*it));
            }
        }
        Logger(WARN) << "\tTime for particle inter: " << t_particleParticle.elapsed();
    } else {
        #pragma omp parallel for
        for (int bIndex = 0; bIndex < NUM_BODIES; bIndex++) {
            if (tree->getOctant().contains(bods[bIndex].position)) {
                treeInteraction(tree, &bods[bIndex]);
            }
        }
        Logger(WARN) << "\tTime for particle inter: " << t_particleParticle.elapsed();
    }
    delete tree;

    t_updateBodies.reset();
    updateBodies(suns, bods);
    Logger(WARN) << "\tTime for updating bods: " << t_updateBodies.elapsed();

}

void Interaction::updateBodies(Body* suns, Body* bods)
{
    Logger(DEBUG) << "Updating particle positions ...";

    #pragma omp parallel for
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

    #pragma omp parallel for
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

