//
// Created by Michael Staneker on 12.04.21.
//

#ifndef OOP_TREE_H
#define OOP_TREE_H

#include "Particle.h"
#include "Domain.h"
#include <vector>
#include <string>

#define DIM 3
#define POWDIM 8

typedef std::vector<Particle> ParticleList;

class TreeNode {

public:
    enum nodeType
    {
        particle, pseudoParticle, domainList
    };

    std::string getNodeType() const;

    Particle p;
    Domain box;
    TreeNode *son[POWDIM];
    nodeType node;

    TreeNode();
    TreeNode(Particle &p, Domain &box, nodeType node_=particle);

    friend std::ostream &operator << (std::ostream &os, const TreeNode &t);

    bool isLeaf();
    bool isDomainList();
    bool isLowestDomainList();

    int getSonBox(Particle &particle);
    int getSonBox(Particle &particle, Domain &sonBox);

    void insert(Particle &p);

    void getParticleList(ParticleList &ParticleList);

};

inline std::ostream &operator<<(std::ostream &os, const TreeNode &t)
{
    os << "TreeNode------------------------------------" << std::endl;
    os << "Particle: " << std::endl;
    os << t.p << std::endl;
    os << "Box: " << t.box << std::endl;
    os << "NodeType: " << t.getNodeType() << std::endl;
    os << "Son: [";
    for (int i=0; i<POWDIM; i++) {
        os << ((t.son[i] == NULL) ? "-" : std::to_string(i).c_str());
        if (i<POWDIM-1) {
            os << "|";
        }
    }
    os << "]" << std::endl;
    os << "--------------------------------------------";

    return os;
}

const char* getNodeType(int nodeIndex);


#endif //OOP_TREE_H
