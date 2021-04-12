//
// Created by Michael Staneker on 12.04.21.
//

#include "../include/Tree.h"

const char* getNodeType(int nodeIndex) {
    switch (nodeIndex) {
        case 0:  return "particle      ";
        case 1:  return "pseudoParticle";
        case 2:  return "domainList    ";
        default: return "not valid     ";
    }
}

TreeNode::TreeNode() {
    son[0] = NULL;
    son[1] = NULL;
    son[2] = NULL;
    son[3] = NULL;
    son[4] = NULL;
    son[5] = NULL;
    son[6] = NULL;
    son[7] = NULL;
    node = particle;
}

TreeNode::TreeNode(Particle &p, Domain &box, nodeType node_) : p { p }, box{ box } {
    son[0] = NULL;
    son[1] = NULL;
    son[2] = NULL;
    son[3] = NULL;
    son[4] = NULL;
    son[5] = NULL;
    son[6] = NULL;
    son[7] = NULL;
    node = particle;
    node = node_;
}

std::string TreeNode::getNodeType() const {
    std::string nodeTypeStr = "";
    switch (node) {
        case 0:  nodeTypeStr +=  "particle      "; break;
        case 1:  nodeTypeStr +=  "pseudoParticle"; break;
        case 2:  nodeTypeStr +=  "domainList    "; break;
        default: nodeTypeStr +=  "not valid     "; break;
    }
    return nodeTypeStr;
}

bool TreeNode::isLeaf() {
    for (int i = 0; i < POWDIM; i++) {
        if (son[i] != NULL) {
            return false;
        }
    }
    return true;
}

bool TreeNode::isDomainList() {
    if (node == domainList) {
        return true;
    }
    return false;
}

bool TreeNode::isLowestDomainList() {
    if (node == domainList) {
        if (this->isLeaf()) {
            return true;
        }
        else {
            for (int i=0; i<POWDIM; i++) {
                if (son[i] && son[i]->node == domainList) {
                    return false;
                }
            }
            return true;
        }
    }
    return false;
}

int TreeNode::getSonBox(Particle &particle) {
    int son = 0;
    Vector3<dFloat> center;
    box.getCenter(center);
    for (int d=DIM-1; d>= 0; d--) {
        if (particle.x[d] < center[d]) {
            son = 2 * son;
        }
        else {
            son = 2 * son + 1;
        }
    }
    return son;
}

int TreeNode::getSonBox(Particle &particle, Domain &sonBox) {
    int son = 0;
    Vector3<dFloat> center;
    box.getCenter(center);
    for (int d=DIM-1; d>= 0; d--) {
        if (particle.x[d] < center[d]) {
            son = 2 * son;
            sonBox.lower[d] = box.lower[d];
            sonBox.upper[d] = center[d];
        }
        else {
            son = 2 * son + 1;
            sonBox.lower[d] = center[d];
            sonBox.upper[d] = box.upper[d];
        }
    }
    return son;
}

void TreeNode::insert(Particle &p) {

    Domain sonBox;
    int nSon = getSonBox(p, sonBox);
    //std::cout << "sonBox: " << sonBox << std::endl;

    if (this->isDomainList() && son[nSon] != NULL) {
        son[nSon]->box = sonBox;
        son[nSon]->insert(p);
    }
    else {
        if (son[nSon] == NULL) {
            if (this->isLeaf() && !this->isDomainList()) {
                Particle p2 = p;
                node = pseudoParticle;
                son[nSon] = new TreeNode(p, sonBox);
                this->insert(p2);
            }
            else {
                son[nSon] = new TreeNode(p, sonBox);
            }
        }
        else {
            son[nSon]->box = sonBox;
            son[nSon]->insert(p);
        }
    }
}

void TreeNode::getParticleList(ParticleList &ParticleList) {
    if (this->isLeaf() && !this->isDomainList()) {
        ParticleList.push_back(p);
    }
    else {
        for (int i=0; i<POWDIM; i++) {
            if (son[i] != NULL) {
                son[i]->getParticleList(ParticleList);
            }
        }
    }
}

