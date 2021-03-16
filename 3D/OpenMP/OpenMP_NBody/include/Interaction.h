//
// Created by Michael Staneker on 25.01.21.
//

#ifndef NBODY_INTERACTION_H
#define NBODY_INTERACTION_H

#include <cstdint>
#include <cstdlib>
#include <map>
#include <vector>

#include "Vector3D.h"
#include "Body.h"
#include "Tree.h"
#include "Constants.h"
#include "Logger.h"
#include "Utils.h"
#include "Timer.h"

class Interaction {

private:

    bool frictionEnabled;

    /**
     * Gravitational interaction between two bodies.
     *
     * @param body1 Body instance
     * @param body2 Body instance
     * @param bool, symmetric apply force/acceleration on both bodies
     */
    void singleInteraction(Body* body1, Body* body2, bool symmetric, bool debug=false);

    /**!
     * Gravitational interaction from Octree (recursive).
     *
     * @param tree Tree instance
     * @param body Body instance
     */
    void treeInteraction(Tree *tree, Body *body);

    /**!
     * Update/advance particles/bodies.
     *
     * @param bods Body instances
     */
    void updateBodies(Body* suns, Body* bods);

public:

    /**!
     * Default constructor for Interaction class.
     */
    Interaction();

    /**!
     * Constructor for Interaction class.
     *
     * @param _frictionEnabled
     */
    Interaction(bool _frictionEnabled);

    /**!
     * Handle interactions for all particles/bodies.
     *
     * * Calculate force from star(s)
     * * Build Octree
     * * Calculate (approximate) gravitational forces
     * * update/advance particles
     *
     * @param b Body instances.
     * @param hashed flag for using a hashed tree for parallelization
     */
    void interactBodies(Body* suns, Body* bods, bool hashed=false);

};


#endif //NBODY_INTERACTION_H
