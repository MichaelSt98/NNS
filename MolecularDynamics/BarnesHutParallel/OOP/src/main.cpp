
#include "../include/Logger.h"
#include "../include/Keytype.h"
#include "../include/Particle.h"
#include "../include/Tree.h"
#include "../include/Domain.h"

#include <iostream>
#include <climits>

#define KEY_MAX ULONG_MAX

structlog LOGCFG = {};

int main() {

    LOGCFG.headers = true;
    LOGCFG.level = DEBUG;

    Particle p;
    //std::cout << p << std::endl;

    //nodeType node(nodeType::pseudoParticle) ; //= nodeType::pseudoParticle;
    //std::cout << "node = " << node.type << std::endl;

    Vector3<float> lowerVec {-5, -5, -5};
    Vector3<float> upperVec {5, 5, 5};
    Domain domain { lowerVec, upperVec };

    /*
    Vector3<float> vec {1, 2, 3};

    bool inside = domain.withinDomain(vec);

    std::cout << "Inside: " << (inside ? "true" : "false")  << std::endl;*/

    ParticleList particleList;

    for (int i=1; i<100; i++) {
        Vector3<float> x {i/100.f, i/50.f, i/150.f};
        particleList.push_back(Particle(x));
    }

    Vector3<float> root {0, 0, 0};
    Particle rootParticle { root };
    TreeNode *treeNode = new TreeNode(rootParticle, domain);

    std::cout << *treeNode << std::endl << std::endl;

    /*for (const auto& particle: particleList) {
        //std::cout << particle;
        treeNode->insert(particle);
    }*/

    int counter = 0;
    for (auto it = std::begin(particleList); it != std::end(particleList); ++it) {
        std::cout << "Counter = " << counter << std::endl;
        treeNode->insert(*it);
        std::cout << *treeNode << std::endl;
        counter++;
    }


    //extensionType test = 2 << 30;
    //Logger(INFO) << "2 << 2 = " << test;


    //KeyType key(KEY_MAX-7);
    //std::cout << "max level: " << key.getMaxLevel() << std::endl;
    //std::cout << key << std::endl;

    return 0;
}

