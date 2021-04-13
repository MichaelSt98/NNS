//
// Created by Michael Staneker on 13.04.21.
//

#include "../include/SubDomain.h"

SubDomain::SubDomain() {
    rank = comm.rank();
    numProcesses = comm.size();
}

void SubDomain::getParticleKeys(TreeNode &t, KeyList &keyList, KeyType k, int level) {
    for (int i=0; i<8; i++) {
        if (t.son[i] != NULL) {
            if (t.son[i]->isLeaf()) {
                keyList.push_back((k | KeyType((keyInteger) i << (3 * (k.maxLevel - level - 1)))));
            } else {
                getParticleKeys(*t.son[i], keyList, (k | KeyType((keyInteger) i << (3 * (k.maxLevel - level - 1)))),
                                level + 1);
            }
        }
    }
}

void SubDomain::getParticleKeys(KeyList &keyList, KeyType k, int level) {
    for (int i=0; i<8; i++) {
        if (root.son[i] != NULL) {
            if (root.son[i]->isLeaf()) {
                keyList.push_back((k | KeyType((keyInteger) i << (3 * (k.maxLevel - level - 1)))));
            } else {
                getParticleKeys(*root.son[i], keyList, (k | KeyType((keyInteger) i << (3 * (k.maxLevel - level - 1)))),
                                level + 1);
            }
        }
    }
}

void SubDomain::createRanges() {

}