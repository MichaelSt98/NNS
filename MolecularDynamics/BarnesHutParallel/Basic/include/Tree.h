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
#include <climits> // for ulong_max
#include <mpi.h>

#define KEY_MAX ULONG_MAX

typedef enum { particle, pseudoParticle, domainList } nodetype;

typedef struct TreeNode {
    Particle p;
    Box box;
    struct TreeNode *son[POWDIM];
    nodetype node;
} TreeNode;


typedef unsigned long keytype;

const int maxlevel = (sizeof(keytype)*CHAR_BIT - 1)/DIM;

typedef struct {
    int myrank;
    int numprocs;
    keytype *range;
} SubDomainKeyTree;

int key2proc(keytype k, SubDomainKeyTree *s);

void createDomainList(TreeNode *t, int level, keytype k, SubDomainKeyTree *s);

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

void freeTree_BH(TreeNode *root);

void sendParticles(TreeNode *root, SubDomainKeyTree *s);

void buildSendlist(TreeNode *t, SubDomainKeyTree *s, ParticleList *plist);

void compPseudoParticlespar(TreeNode *root, SubDomainKeyTree *s);

void compLocalPseudoParticlespar(TreeNode *t);

void compDomainListPseudoParticlespar(TreeNode *t);

void symbolicForce(TreeNode *td, TreeNode *t, float diam, ParticleList *plist, SubDomainKeyTree *s);

void compF_BHpar(TreeNode *root, float diam, SubDomainKeyTree *s);

void compTheta(TreeNode *t, TreeNode *root, SubDomainKeyTree *s, ParticleList *plist, float diam);

#endif //BARNESHUTSERIAL_TREE_H
