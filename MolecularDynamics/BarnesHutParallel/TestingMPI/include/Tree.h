//
// Created by Michael Staneker on 15.03.21.
//

#ifndef BARNESHUTSERIAL_TREE_H
#define BARNESHUTSERIAL_TREE_H

#include "Constants.h"
#include "Particle.h"
#include <cmath>
#include <iostream>
#include <climits> // for ulong_max
#include <mpi.h>

#define KEY_MAX ULONG_MAX

/** three types of tree nodes:
 * * **particle:** leaf nodes, which are nodes without sons in which particle data is stored
 * * **pseudoParticle:** inner tree nodes that don not belong to the common coarse tree (in which pseudoparticles are stored)
 * * **domainList:** tree nodes belonging to the common coarse tree describing the domain decomposition
 */
typedef enum { particle, pseudoParticle, domainList } nodetype;

typedef struct TreeNode {
    Particle p;
} TreeNode;


typedef unsigned long keytype;

const int maxlevel = (sizeof(keytype)*CHAR_BIT - 1)/DIM;

typedef struct {
    int myrank;
    int numprocs;
    keytype *range;
} SubDomainKeyTree;


ParticleList* build_particle_list(TreeNode *t, ParticleList *pLst);

void get_particle_array(TreeNode *root, Particle *p);


//new
void sendParticles(TreeNode *root, SubDomainKeyTree *s);

//new
void buildSendlist(TreeNode *t, SubDomainKeyTree *s, ParticleList *plist, int *plistLength);

#endif