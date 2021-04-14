//
// Created by Michael Staneker on 13.04.21.
//

#include "../include/SubDomain.h"

SubDomain::SubDomain() {
    rank = comm.rank();
    numProcesses = comm.size();
}

void SubDomain::moveParticles() {
    root.resetParticleFlags();
    root.moveLeaf(root);
    root.repairTree();
}

void SubDomain::getParticleKeys(KeyList &keyList, KeyType k, int level) {
    root.getParticleKeys(keyList, k, level);
}

void SubDomain::createRanges() {

}

void SubDomain::gatherParticles(ParticleList &pList) {

    ParticleList myProcessParticles;
    root.getParticleList(myProcessParticles);

    Particle *pArrayLocal = &myProcessParticles[0];

    int localLength = (int)myProcessParticles.size();
    IntList receiveLengths;

    //boost::mpi::gather(comm, &localLength, 1, receiveLengths, 0);
    boost::mpi::all_gather(comm, &localLength, 1, receiveLengths);

    int totalReceiveLength = 0;
    for (auto it = std::begin(receiveLengths); it != std::end(receiveLengths); ++it) {
        //std::cout << "receiveLengths: " << *it << std::endl;
        totalReceiveLength += *it;
    }

    Particle *pArray;

    if (rank == 0) {
        pArray = new Particle[totalReceiveLength];
    }

    boost::mpi::gatherv(comm, myProcessParticles, pArray, receiveLengths, 0);

    if (rank == 0) {
        pList.assign(pArray, pArray + totalReceiveLength);
        delete [] pArray;
    }
}

void SubDomain::gatherParticles(ParticleList &pList, IntList &processList) {

    ParticleList myProcessParticles;
    root.getParticleList(myProcessParticles);

    Particle *pArrayLocal = &myProcessParticles[0];

    int localLength = (int)myProcessParticles.size();
    IntList receiveLengths;

    //boost::mpi::gather(comm, &localLength, 1, receiveLengths, 0);
    boost::mpi::all_gather(comm, &localLength, 1, receiveLengths);

    int totalReceiveLength = 0;
    for (auto it = std::begin(receiveLengths); it != std::end(receiveLengths); ++it) {
        //std::cout << "receiveLengths: " << *it << std::endl;
        totalReceiveLength += *it;
    }

    Particle *pArray;

    if (rank == 0) {
        pArray = new Particle[totalReceiveLength];
    }

    boost::mpi::gatherv(comm, myProcessParticles, pArray, receiveLengths, 0);

    if (rank == 0) {
        pList.assign(pArray, pArray + totalReceiveLength);
        IntList helper;
        for (int proc=0; proc<numProcesses; proc++) {
            helper.assign(receiveLengths[proc], proc);
            processList.insert(processList.end(), helper.begin(), helper.end());
        }
        delete [] pArray;
    }
}