//
// Created by Michael Staneker on 13.04.21.
//

#ifndef OOP_SUBDOMAIN_H
#define OOP_SUBDOMAIN_H

#include "Keytype.h"
#include "Tree.h"

#include <boost/mpi.hpp>
#include <vector>

class SubDomain {
public:
    boost::mpi::communicator comm;

    int rank;
    int numProcesses;
    KeyList range;
    TreeNode root;

    SubDomain();

    void moveParticles();

    //void getParticleKeys(TreeNode &t, KeyList &keyList, KeyType k=0UL, int level=0);
    void getParticleKeys(KeyList &keyList, KeyType k=0UL, int level=0);

    //TODO: implement
    void key2proc();

    void createRanges();

    //TODO: implement
    void createDomainList();

    //TODO: implement
    void sendParticles();
    void buildSendList();

    //TODO: implement
    void symbolicForce(TreeNode &td, TreeNode &t, float diam, ParticleList pList, KeyType k=0UL, int level=0);
    void symbolicForce(TreeNode &t, float diam, ParticleList pList, KeyType k=0UL, int level=0);
    void compPseudoParticles();
    void compF();
    void compTheta();

    //TODO: implement
    void gatherParticles(ParticleList &pList);
    void gatherParticles(ParticleList &pList, IntList &processList);
};


#endif //OOP_SUBDOMAIN_H
