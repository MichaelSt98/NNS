//
// Created by Michael Staneker on 13.04.21.
//

#ifndef OOP_SUBDOMAIN_H
#define OOP_SUBDOMAIN_H

#include "Keytype.h"
#include "Tree.h"

#include <boost/mpi.hpp>
#include <vector>

typedef std::vector<KeyType> KeyList;

class SubDomain {
public:
    boost::mpi::communicator comm;

    int rank;
    int numProcesses;
    KeyList range;
    TreeNode root;

    SubDomain();

    void getParticleKeys(TreeNode &t, KeyList &keyList, KeyType k=0UL, int level=0);
    void getParticleKeys(KeyList &keyList, KeyType k=0UL, int level=0);

    void createRanges();
};


#endif //OOP_SUBDOMAIN_H
