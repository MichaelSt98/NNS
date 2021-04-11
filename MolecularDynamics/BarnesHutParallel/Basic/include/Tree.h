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
    //~NodeList();
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
};

const int maxlevel = (sizeof(keytype)*CHAR_BIT-1)/DIM;

struct SubDomainKeyTree {
    int myrank;
    int numprocs;
    keytype *range;
};

// keytype key(TreeNode *t); // DUMMY
//keytype key(TreeNode *t, TreeNode *&keynode, keytype k=0UL, int level=0);

//void deleteNodeList(NodeList * nLst);
/*
 void deleteNodeList(NodeList * nLst) {
    while (nLst->next)
    {
        NodeList* old = nLst;
        nLst = nLst->next;
        delete old;
    }
    if (nLst) {
        delete nLst;
    }
}
 */

/** First step in the path key creation done explicitly [p. 343f]
 * - Starting at the root node 1
 * - Each level needs 3 bits => [0,7] are the labels of the sons
 * - The labels [0,7] are shifted 3 x level times
 *
 * @param t Current node in recursion, should be initialized with root
 * @param p Container to be filled with path keys of all leaves (keytype[N])
 * @param pCounter Global counter by reference
 * @param k default=1UL (root node)
 * @param level default=0
 */
void getParticleKeysSimple(TreeNode *t, keytype *p, int &pCounter, keytype k=1UL, int level=0);

void getParticleKeys(TreeNode *t, keytype *p, int &pCounter, keytype k=0UL, int level=0);

void createRanges(TreeNode *root, int N, SubDomainKeyTree *s);

int key2proc(keytype k, SubDomainKeyTree *s);

void createDomainList(TreeNode *t, int level, keytype k, SubDomainKeyTree *s);

bool isLeaf(TreeNode *t);

void insertTree(Particle *p, TreeNode *t);

int sonNumber(Box *box, Box *sonbox, Particle *p);

void compPseudoParticles(TreeNode *t);

void compF_BH(TreeNode *t, TreeNode *root, float diam, SubDomainKeyTree *s, keytype k=0UL, int level=0);

void force_tree(TreeNode *tl, TreeNode *t, float diam);

void compX_BH(TreeNode *t, float delta_t);

void compV_BH(TreeNode *t, float delta_t);

void moveParticles_BH(TreeNode *root);

void setFlags(TreeNode *t);

void moveLeaf(TreeNode *t, TreeNode *root);

void repairTree(TreeNode *t);

void output_tree(TreeNode *root, bool detailed=false, bool onlyParticles=false);

void output_tree(TreeNode *root, std::string file, bool detailed=false, bool onlyParticles=false);

void output_particles(TreeNode *root);

NodeList* build_tree_list(TreeNode *t, NodeList *nLst);

ParticleList* build_particle_list(TreeNode *t, ParticleList *pLst);

int getParticleListLength(ParticleList *plist);

int get_tree_node_number(TreeNode *root);

int get_tree_array(TreeNode *root, Particle *&p, nodetype *&n);

int get_particle_array(TreeNode *root, Particle *&p);

void freeTree_BH(TreeNode *root);

void sendParticles(TreeNode *root, SubDomainKeyTree *s);

//void buildSendlist(TreeNode *root, TreeNode *t, SubDomainKeyTree *s, ParticleList *plist, int *pIndex);
void buildSendlist(TreeNode *root, TreeNode *t, SubDomainKeyTree *s, ParticleList *plist, int *pIndex, keytype k, int level);
//void buildSendlist(TreeNode *t, SubDomainKeyTree *s, ParticleList *plist);

int get_domain_list_array(TreeNode *root, Particle *&pArray);

int get_lowest_domain_list_array(TreeNode *root, Particle *&pArray);

int get_lowest_domain_list_array(TreeNode *root, Particle *&pArray, keytype *&kArray);

void get_domain_list_nodes(TreeNode *t, ParticleList *pList, int &pCounter);

void get_lowest_domain_list_nodes(TreeNode *t, ParticleList *pList, int &pCounter);

void get_lowest_domain_list_nodes(TreeNode *t, ParticleList *pList, KeyList *kList,
                                  int &pCounter, keytype k=0UL, int level=0);

void zero_lowest_domain_list_nodes(TreeNode *t);

void zero_domain_list_nodes(TreeNode *t);

void update_lowest_domain_list_nodes_moments_masses(TreeNode *t, int &pCounter, float * masses, float * moments);

void update_lowest_domain_list_nodes_com(TreeNode *t);

void update_lowest_domain_list_nodes(TreeNode *t, int &pCounter, float * masses, float * moments);

int get_domain_moments_array(TreeNode *root, float * moments);

bool isLowestDomainListNode(TreeNode *t);

void compPseudoParticlespar(TreeNode *root, SubDomainKeyTree *s);

void compLocalPseudoParticlespar(TreeNode *t);

void compDomainListPseudoParticlespar(TreeNode *t);

float smallestDistance(TreeNode *td, TreeNode *t);

void symbolicForce(TreeNode *td, TreeNode *t, float diam, ParticleList *plist, SubDomainKeyTree *s,
                   int &pCounter, keytype k=0UL, int level=0);

void symbolicForce(TreeNode *td, TreeNode *t, float diam, ParticleMap &pmap, SubDomainKeyTree *s,
                   keytype k=0UL, int level=0);

void compF_BHpar(TreeNode *root, float diam, SubDomainKeyTree *s);

//void compTheta(TreeNode *t, TreeNode *root, SubDomainKeyTree *s, ParticleList *plist, float diam);

void compTheta(TreeNode *t, TreeNode *root, SubDomainKeyTree *s, ParticleList *plist, int *& pCounter, float diam,
               keytype k=0UL, int level=0);

void compTheta(TreeNode *t, TreeNode *root, SubDomainKeyTree *s, ParticleMap *pmap, float diam,
               keytype k=0UL, int level=0);

bool compareParticles(Particle p1, Particle p2);

int gatherParticles(TreeNode *root, SubDomainKeyTree *s, Particle *&pArrayAll);

int gatherParticles(TreeNode *root, SubDomainKeyTree *s, Particle *&pArrayAll, int*&processNumber);

#endif //BARNESHUTSERIAL_TREE_H
