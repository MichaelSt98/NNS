//
// Created by Michael Staneker on 13.04.21.
//

#ifndef OOP_SUBDOMAIN_H
#define OOP_SUBDOMAIN_H

#include "Keytype.h"
#include "Tree.h"

#include <boost/mpi.hpp>
#include <vector>
#include <map>

typedef std::map<KeyType, Particle> ParticleMap;

class SubDomain {
public:
    boost::mpi::communicator comm;

    int rank;
    int numProcesses;
    KeyType *range;
    TreeNode root;

    SubDomain();

    void moveParticles();

    void getParticleKeys(KeyList &keyList, KeyType k=0UL, int level=0);

    int key2proc(KeyType k);

    void createRanges();
    void newLoadDistribution();

    void createDomainList(TreeNode &t, int level, KeyType k);
    void createDomainList();

    void sendParticles();
    void buildSendList(TreeNode &t, ParticleList *pList, KeyType k, int level);

    void symbolicForce(TreeNode &td, TreeNode &t, float diam, ParticleMap &pMap, KeyType k=0UL, int level=0);
    //void symbolicForce(TreeNode &t, float diam, ParticleList pList, KeyType k=0UL, int level=0);
    void compPseudoParticles();
    void compF(TreeNode &t, float diam, KeyType k=0UL, int level=0);
    void compFParallel(float diam);
    void compTheta(TreeNode &t, ParticleMap *pMap, float diam, KeyType k=0UL, int level=0);

    void gatherKeys(KeyList &keyList, IntList &lengths, KeyList &localKeyList);
    void gatherParticles(ParticleList &pList);
    void gatherParticles(ParticleList &pList, IntList &processList);
};


#endif //OOP_SUBDOMAIN_H
