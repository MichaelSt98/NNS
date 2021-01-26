//
// Created by Michael Staneker on 25.01.21.
//

#ifndef NBODY_INTERACTION_H
#define NBODY_INTERACTION_H

#include "Vector3D.h"
#include "Body.h"
#include "Tree.h"
#include "Constants.h"

class Interaction {

public:

    bool frictionEnabled;

    //friend class Tree;

    Interaction();

    Interaction(bool _frictionEnabled);

    void singleInteraction(Body* body1, Body* body2, bool single);

    void treeInteraction(Tree *tree, Body *body);
};


#endif //NBODY_INTERACTION_H
