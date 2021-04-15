//
// Created by Michael Staneker on 13.04.21.
//

#include "../include/SubDomain.h"

SubDomain::SubDomain() {
    rank = comm.rank();
    numProcesses = comm.size();
    range = new KeyType[numProcesses + 1];
}

void SubDomain::moveParticles() {
    root.resetParticleFlags();
    root.moveLeaf(root);
    root.repairTree();
}

void SubDomain::getParticleKeys(KeyList &keyList, KeyType k, int level) {
    root.getParticleKeys(keyList, k, level);
}

int SubDomain::key2proc(KeyType k) {
    for (int proc=0; proc<numProcesses; proc++) {
        if (k >= range[proc] && k < range[proc+1]) {
            return proc;
        }
    }
    Logger(ERROR) << "key2proc(k=" << k << ") = -1!";
    return -1; //error
}

void SubDomain::createRanges() {

    KeyList kList;
    getParticleKeys(kList);

    KeyList globalKeyList;
    IntList particlesOnEachProcess;
    gatherKeys(globalKeyList, particlesOnEachProcess, kList);

    std::sort(globalKeyList.begin(), globalKeyList.end());

    int N = globalKeyList.size();
    const int ppr = (N % numProcesses != 0) ? N/numProcesses+1 : N/numProcesses;

    if (rank == 0) {
        range[0] = 0UL;
        for (int i = 1; i < numProcesses; i++) {
            range[i] = globalKeyList[i * ppr];
        }
        range[numProcesses] = KEY_MAX;
    }

    boost::mpi::broadcast(comm, range, numProcesses+1, 0);
}

void SubDomain::createDomainList(TreeNode &t, int level, KeyType k) {
    t.node = TreeNode::domainList;
    int proc1 = key2proc(k);
    int proc2 = key2proc(k | ~(~0L << DIM*(k.maxLevel-level)));
    if (proc1 != proc2) {
        for (int i=0; i<POWDIM; i++) {
            t.son[i] = new TreeNode;
            createDomainList(*t.son[i], level + 1, KeyType(k | ((keyInteger)i << (DIM*(k.maxLevel-level-1)))));
        }
    }
}

void SubDomain::createDomainList() {
    createDomainList(root, 0, 0UL);
}

void SubDomain::sendParticles() {
    ParticleList *particleLists;
    particleLists = new ParticleList[numProcesses];

    buildSendList(root, particleLists, 0UL, 0);

    root.repairTree();

    Particle ** pArray = new Particle*[numProcesses];

    int *pLengthSend;
    pLengthSend = new int[numProcesses];
    pLengthSend[rank] = -1;

    int *pLengthReceive;
    pLengthReceive = new int[numProcesses];
    pLengthReceive[rank] = -1;

    for (int proc=0; proc<numProcesses; proc++) {
        if (proc != rank) {
            pLengthSend[proc] = (int)particleLists[proc].size();
            pArray[proc] = new Particle[pLengthSend[proc]];
            pArray[proc] = &particleLists[proc][0];
        }
    }

    std::vector<boost::mpi::request> reqMessageLengths;
    std::vector<boost::mpi::status> statMessageLengths;

    for (int proc=0; proc<numProcesses; proc++) {
        if (proc != rank) {
            reqMessageLengths.push_back(comm.isend(proc, 17, &pLengthSend[proc], 1));
            statMessageLengths.push_back(comm.recv(proc, 17, &pLengthReceive[proc], 1));
        }
    }

    boost::mpi::wait_all(reqMessageLengths.begin(), reqMessageLengths.end());

    int totalReceiveLength = 0;
    for (int proc=0; proc<numProcesses; proc++) {
        //Logger(INFO) << "Receive length[" << proc << "]: " << pLengthReceive[proc];
        if (proc != rank) {
            totalReceiveLength += pLengthReceive[proc];
        }
    }

    pArray[rank] = new Particle[totalReceiveLength];

    std::vector<boost::mpi::request> reqParticles;
    std::vector<boost::mpi::status> statParticles;

    int offset = 0;
    for (int proc=0; proc<numProcesses; proc++) {
        if (proc != rank) {
            reqParticles.push_back(comm.isend(proc, 17, pArray[proc], pLengthSend[proc]));
            statParticles.push_back(comm.recv(proc, 17, pArray[rank] + offset, pLengthReceive[proc]));
            offset += pLengthReceive[proc];
        }
    }

    boost::mpi::wait_all(reqParticles.begin(), reqParticles.end());

    for (int i=0; i<totalReceiveLength; i++) {
        //Logger(INFO) << "Inserting particle pArray[" << i << "].x : " << pArray[rank][i].x;
        root.insert(pArray[rank][i]);
    }

    delete [] pLengthReceive;
    delete [] pLengthSend;
    for (int proc=0; proc<numProcesses; proc++) {
        delete [] pArray[proc];
    }
    delete [] pArray;

}

void SubDomain::buildSendList(TreeNode &t, ParticleList *pList, KeyType k, int level) {
    int proc;
    if (t.isLeaf() && ((proc = key2proc(k)) != rank) && !t.isDomainList()) {
        pList[proc].push_back(t.p);
        t.p.toDelete = true;
    }
    for (int i=0; i<POWDIM; i++) {
        if (t.son[i] != NULL) {
            buildSendList(*t.son[i], pList, KeyType(k | ((keyInteger) i << (DIM * (k.maxLevel - level - 1)))),
                          level + 1);
        }
    }
}

void SubDomain::symbolicForce(TreeNode &td, TreeNode &t, float diam, ParticleMap &pMap, KeyType k, int level) {
    if (key2proc(k) == rank || t.isDomainList()) {
        if (!t.isDomainList()) {
            bool insert = true;
            for (ParticleMap::iterator pit = pMap.begin(); pit != pMap.end(); pit++) {
                if (pit->second.x == t.p.x) {
                    insert = false;
                }
            }
            if (insert) {
                pMap[k] = t.p;
            }
        }
    }
    float r = td.smallestDistance(t.p);

    if (diam > theta*r) {
        for (int i=0; i<POWDIM; i++) {
            if (t.son[i] != NULL) {
                symbolicForce(td, *t.son[i], 0.5 * diam, pMap,
                              KeyType(k | ((keyInteger) i << (DIM * (k.maxLevel - level - 1)))),
                              level + 1);
            }
        }
    }
}

void SubDomain::compPseudoParticles() {

    root.resetDomainList();

    root.compLocalPseudoParticles();

    ParticleList lowestDomainList;
    root.getLowestDomainList(lowestDomainList);

    int dLength = (int)lowestDomainList.size();

    pFloat moments[3*dLength];
    pFloat globalMoments[3*dLength];
    pFloat masses[dLength];
    pFloat globalMasses[dLength];

    for (int i=0; i<dLength; i++) {
        moments[3*i] = lowestDomainList[i].x[0] * lowestDomainList[i].m;
        moments[3*i+1] = lowestDomainList[i].x[1] * lowestDomainList[i].m;
        moments[3*i+2] = lowestDomainList[i].x[2] * lowestDomainList[i].m;
        masses[i] = lowestDomainList[i].m;
    }

    //void all_reduce(const communicator & comm, const T * in_values, int n, T * out_values, Op op);
    boost::mpi::all_reduce(comm, moments, 3*dLength, globalMoments, std::plus<pFloat>());
    boost::mpi::all_reduce(comm, masses, dLength, globalMasses, std::plus<pFloat>());

    int dCounter=0;
    root.updateLowestDomainList(dCounter, globalMasses, globalMoments);

    root.compDomainListPseudoParticles();
}

void SubDomain::compF(float diam) {

}

void SubDomain::compTheta(TreeNode &t, ParticleMap *pMap, float diam, KeyType k, int level) {
    for (int i=0; i<POWDIM; i++) {
        if (t.son[i] != NULL) {
            compTheta(*t.son[i], pMap, diam, KeyType(k | ((keyInteger) i << (DIM * (k.maxLevel - level - 1)))),
                      level + 1);
        }
    }
    int proc;
    if (t.isDomainList() && (proc = key2proc(k) != rank)) {
        symbolicForce(t, root, diam, pMap[proc], 0UL, 0);
    }
}

void SubDomain::gatherKeys(KeyList &keyList, IntList &lengths, KeyList &localKeyList) {

    KeyType *kArrayLocal = &localKeyList[0];

    int localLength = (int)localKeyList.size();

    //boost::mpi::gather(comm, &localLength, 1, receiveLengths, 0);
    boost::mpi::all_gather(comm, &localLength, 1, lengths);

    int totalReceiveLength = 0;
    for (auto it = std::begin(lengths); it != std::end(lengths); ++it) {
        //std::cout << "receiveLengths: " << *it << std::endl;
        totalReceiveLength += *it;
    }

    KeyType *kArray;

    if (rank == 0) {
        kArray = new KeyType[totalReceiveLength];
    }

    boost::mpi::gatherv(comm, localKeyList, kArray, lengths, 0);

    if (rank == 0) {
        keyList.assign(kArray, kArray + totalReceiveLength);
        delete [] kArray;
    }
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