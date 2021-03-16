//
// Created by Michael Staneker on 15.03.21.
//

#ifndef BARNESHUTSERIAL_TREE_H
#define BARNESHUTSERIAL_TREE_H

#include "Domain.h"
#include "Constants.h"
#include "Particle.h"
#include <cmath>
#include <iostream>

typedef struct TreeNode {
    Particle p;
    Box box;
    struct TreeNode *son[POWDIM];
} TreeNode;

bool isLeaf(TreeNode *t);

void insertTree(Particle *p, TreeNode *t);

int sonNumber(Box *box, Box *sonbox, Particle *p);

void compPseudoParticles(TreeNode *t);

void compF_BH(TreeNode *t, TreeNode *root, float diam);

void force_tree(TreeNode *tl, TreeNode *t, float diam);

void compX_BH(TreeNode *t, float delta_t);

void compV_BH(TreeNode *t, float delta_t);

void moveParticles_BH(TreeNode *root);

void setFlags(TreeNode *t);

void moveLeaf(TreeNode *t, TreeNode *root);

void repairTree(TreeNode *t);

void output_particles(TreeNode *root);

void build_particle_list(TreeNode *t, ParticleList *pLst);

void get_particle_array(TreeNode *root, Particle *p);

void freeTree_BH(TreeNode *root);

#endif //BARNESHUTSERIAL_TREE_H
