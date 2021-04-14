
#include "../include/Logger.h"
//#include "../include/Keytype.h"
#include "../include/Particle.h"
#include "../include/Tree.h"
#include "../include/Domain.h"
#include "../include/SubDomain.h"

#include <iostream>
#include <climits>
#include <boost/mpi.hpp>

#define KEY_MAX ULONG_MAX

structlog LOGCFG = {};

int main(int argc, char** argv) {

    LOGCFG.headers = true;
    LOGCFG.level = DEBUG;

    boost::mpi::environment env{argc, argv};
    boost::mpi::communicator comm;

    SubDomain subDomainHandler;
    //std::cout << "subDomainHandler.rank = " << subDomainHandler.rank << std::endl;

    //std::cout << "rank = " << comm.rank() << std::endl;

    Vector3<float> lowerVec {-1, -1, -1};
    Vector3<float> upperVec {1, 1, 1};
    Domain domain { lowerVec, upperVec };

    ParticleList particleList;

    for (int i=1; i<10+(10*comm.rank()); i++) {
        Vector3<float> x {i/100.f, i/50.f, i/150.f};
        particleList.push_back(Particle(x, comm.rank()));
    }

    Vector3<float> root {0, 0, 0};
    Particle rootParticle { root };
    //TreeNode *treeNode = new TreeNode(rootParticle, domain);
    TreeNode treeNode(rootParticle, domain);
    treeNode.node = TreeNode::domainList;

    std::cout << treeNode << std::endl << std::endl;

    int counter = 0;
    for (auto it = std::begin(particleList); it != std::end(particleList); ++it) {
        treeNode.insert(*it);
        counter++;
    }

    ParticleList pList;
    treeNode.getTreeList(pList);



    std::cout << "len(pList) = " << pList.size() << std::endl;

    //treeNode.printTreeSummary(true, TreeNode::particle);

    subDomainHandler.root = treeNode;

    ParticleList allParticles;
    IntList procList;
    subDomainHandler.gatherParticles(allParticles, procList);

    if (comm.rank() == 0) {
        int pCounter = 0;
        for (auto it = std::begin(allParticles); it != std::end(allParticles); ++it) {
            //std::cout << "particle[" << pCounter << "].m = " << it->m << std::endl;
            pCounter++;
        }
        std::cout << "pCounter = " << pCounter << std::endl;
        for (int i=0; i<allParticles.size(); i++) {
            std::cout << "particle[" << i << "].m = " << allParticles[i].m << " from proc: " << procList[i] << std::endl;
        }
    }

    /*KeyList kList;
    subDomainHandler.getParticleKeys(kList);

    for (auto it = std::begin(kList); it != std::end(kList); ++it) {
        std::cout << "key = " << *it << std::endl;
    }*/

    //extensionType test = 2 << 30;
    //Logger(INFO) << "2 << 2 = " << test;



    //KeyType key(KEY_MAX-7);

    /*KeyType key(5UL);
    KeyType shiftedKey = key << 3;
    KeyType orKey = key | shiftedKey;
    KeyType andKey = key & shiftedKey;
    KeyType plusKey = key + shiftedKey;

    std::cout << "key.key: " << key.key << std::endl;
    std::cout << "key.maxLevel: " << key.maxLevel << std::endl;
    std::cout << key << std::endl;
    std::cout << shiftedKey << std::endl;
    std::cout << orKey << std::endl;
    std::cout << andKey << std::endl;
    std::cout << plusKey << std::endl;*/

    /*std::cout << "max level: " << key.getMaxLevel() << std::endl;
    std::cout << "keyStandard: " << key.keyStandard << std::endl;
    std::cout << "keyExtension: " << std::endl;
    for (const auto& val: key.keyExtension) {
        std::cout << val << std::endl;
    }

    std::cout << key << std::endl;*/

    return 0;
}

