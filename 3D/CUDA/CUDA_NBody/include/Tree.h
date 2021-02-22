//
// Created by Michael Staneker on 25.01.21.
//

#ifndef NBODY_TREE_H
#define NBODY_TREE_H

#include "Body.h"
#include "Octant.h"
#include <iostream>
#include <utility>

class Tree {

public:
    Body centerOfMass;
    Octant octant;

    Tree *UNW; //0
    Tree *UNE; //1
    Tree *USW; //2
    Tree *USE; //3
    Tree *LNW; //4
    Tree *LNE; //5
    Tree *LSW; //6
    Tree *LSE; //7

    Tree(Octant& o);
    Tree(Octant&& o);

    const Octant& getOctant () const;

    ~Tree();

    bool isExternal();

    int getMaxDepth();

    void getDepth(Body *body, int &depth);

    void insert(Body *body);
};


#endif //NBODY_TREE_H
