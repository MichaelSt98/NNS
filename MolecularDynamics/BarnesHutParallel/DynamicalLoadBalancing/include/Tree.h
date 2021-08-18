//
// Created by Michael Staneker on 15.03.21.
//

#ifndef BARNESHUTSERIAL_TREE_H
#define BARNESHUTSERIAL_TREE_H

#include "Domain.h"
#include "Constants.h"
#include "Particle.h"
#include "Logger.h"
#include <cmath>
#include <iostream>
#include <bitset>
#include <algorithm> // for sorting particle keys
#include <climits> // for ulong_max
#include <mpi.h>
#include <map>
#include <fstream>

#define KEY_MAX ULONG_MAX

extern MPI_Datatype mpiParticle;

/** three types of tree nodes:
 * * **particle:** leaf nodes, which are nodes without sons in which particle data is stored
 * * **pseudoParticle:** inner tree nodes that don not belong to the common coarse tree (in which pseudoparticles are stored)
 * * **domainList:** tree nodes belonging to the common coarse tree describing the domain decomposition
 */
typedef enum { particle, pseudoParticle, domainList } nodetype;

struct NodeList {
    nodetype node;
    Particle p;
    NodeList *next;

    NodeList();
};

struct TreeNode {
    Particle p;
    Box box;
    TreeNode *son[POWDIM];
    nodetype node;

    TreeNode();
};

typedef unsigned long keytype;
typedef std::map<keytype, Particle> ParticleMap;

struct KeyList {
    keytype k;
    KeyList *next;

    KeyList();
};

const int maxlevel = (sizeof(keytype)*CHAR_BIT-1)/DIM;

struct SubDomainKeyTree {
    int myrank;
    int numprocs;
    keytype *range;
};

/** TREE TRAVERSING AND CORE FUNTIONS **/

long countParticles(TreeNode *t, long count=0);

long countNodes(TreeNode *t, long count=0);

void getParticleKeys(TreeNode *t, keytype *p, int &pCounter, keytype k=0UL, int level=0);

// TODO: replace usage of createRanges() by newLoadDistribution()
void createRanges(TreeNode *root, int N, SubDomainKeyTree *s);

void newLoadDistribution(TreeNode *root, SubDomainKeyTree *s);

void updateRange(TreeNode *t, long &n, int &p, keytype *range, long *newdist, keytype k=0UL, int level=0);

int key2proc(keytype k, SubDomainKeyTree *s);

void createDomainList(TreeNode *t, int level, keytype k, SubDomainKeyTree *s);

void clearDomainList(TreeNode *t);

bool isLeaf(TreeNode *t);

void insertTree(Particle *p, TreeNode *t);

int sonNumber(Box *box, Box *sonbox, Particle *p);

void compPseudoParticles(TreeNode *t);

void compF_BH(TreeNode *t, TreeNode *root, float diam, SubDomainKeyTree *s, keytype k=0UL, int level=0);

void forceTree(TreeNode *tl, TreeNode *t, float diam);

void compX_BH(TreeNode *t, float delta_t);

void compV_BH(TreeNode *t, float delta_t);

void moveParticles_BH(TreeNode *root);

void setFlags(TreeNode *t);

void moveLeaf(TreeNode *t, TreeNode *root);

void repairTree(TreeNode *t);

void freeTree_BH(TreeNode *root);

void sendParticles(TreeNode *root, SubDomainKeyTree *s);

void buildSendList(TreeNode *t, SubDomainKeyTree *s, ParticleList *plist,
                   int *pIndex, keytype k, int level);

/** OUTPUT FUNCTIONS FOR DEBUGGING AND MONITORING **/

void outputTree(TreeNode *root, bool detailed=false, bool onlyParticles=false);

void outputTree(TreeNode *root, std::string file, bool detailed=false, bool onlyParticles=false);

//void outputParticles(TreeNode *root);

/** FUNCTIONS ACCESSING TREE DATA **/

NodeList* buildTreeList(TreeNode *t, NodeList *nLst);

int getTreeArray(TreeNode *root, Particle *&p, nodetype *&n);

KeyList* buildTreeList(TreeNode *t, KeyList *kLst, keytype k=0UL, int level=0);

int getTreeArray(TreeNode *root, Particle *&p, nodetype *&n, keytype *&k);

ParticleList* buildParticleList(TreeNode *t, ParticleList *pLst);

int getParticleListLength(ParticleList *plist); // replaceable by countParticles()

int getParticleArray(TreeNode *root, Particle *&p);

//void getDomainListNodes(TreeNode *t, ParticleList *pList, int &pCounter);

//int getDomainListArray(TreeNode *root, Particle *&pArray);

//void getLowestDomainListNodes(TreeNode *t, ParticleList *pList, int &pCounter);

//int getLowestDomainListArray(TreeNode *root, Particle *&pArray);

void getLowestDomainListNodes(TreeNode *t, ParticleList *pList, KeyList *kList,
                                  int &pCounter, keytype k=0UL, int level=0);

int getLowestDomainListArray(TreeNode *root, Particle *&pArray, keytype *&kArray);

/** FUNCTIONS FOR UPDATING DOMAIN LIST NODES **/

void zeroLowestDomainListNodes(TreeNode *t);

void zeroDomainListNodes(TreeNode *t);

void updateLowestDomainListNodesMomentsMasses(TreeNode *t, int &pCounter, float * masses, float * moments);

void updateLowestDomainListNodesCom(TreeNode *t);

void updateLowestDomainListNodes(TreeNode *t, int &pCounter, float * masses, float * moments);

bool isLowestDomainListNode(TreeNode *t);

/** FUNCTIONS FOR FORCE CALCULATION **/

void compPseudoParticlesPar(TreeNode *root, SubDomainKeyTree *s);

void compLocalPseudoParticlesPar(TreeNode *t);

void compDomainListPseudoParticlesPar(TreeNode *t);

float smallestDistance(TreeNode *td, TreeNode *t);

void symbolicForce(TreeNode *td, TreeNode *t, float diam, ParticleMap &pmap, SubDomainKeyTree *s,
                   keytype k=0UL, int level=0);

void compF_BHpar(TreeNode *root, float diam, SubDomainKeyTree *s);

void compTheta(TreeNode *t, TreeNode *root, SubDomainKeyTree *s, ParticleMap *pmap, float diam,
               keytype k=0UL, int level=0);

//** MISC **/

//bool compareParticles(Particle p1, Particle p2);

int gatherParticles(TreeNode *root, SubDomainKeyTree *s, Particle *&pArrayAll);

int gatherParticles(TreeNode *root, SubDomainKeyTree *s, Particle *&pArrayAll, int*&processNumber);

#endif //BARNESHUTSERIAL_TREE_H
