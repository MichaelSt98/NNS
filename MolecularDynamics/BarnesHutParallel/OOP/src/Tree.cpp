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

void TreeNode::resetSons() {
    son[0] = NULL;
    son[1] = NULL;
    son[2] = NULL;
    son[3] = NULL;
    son[4] = NULL;
    son[5] = NULL;
    son[6] = NULL;
    son[7] = NULL;
}

TreeNode::TreeNode() {
    resetSons();
    node = particle;
}

TreeNode::TreeNode(Particle &p, Domain &box, nodeType node_) : p { p }, box{ box } {
    resetSons();
    node = node_;
}

TreeNode::~TreeNode() {
    //*this = NULL;
}

Node::Node() {

}

Node::Node(Particle p_, TreeNode::nodeType n_) {
    p = p_;
    n = n_;
}

void TreeNode::printTreeSummary(bool detailed, int type) {
    NodeList nList;
    getTreeList(nList);

    int counterParticle = 0;
    int counterPseudoParticle = 0;
    int counterDomainList = 0;

    int counter=0;
    for (auto it = std::begin(nList); it != std::end(nList); ++it) {
        if (detailed) {

            if (type == -1) {
                std::cout << "Node " << std::setw(5) << std::setfill('0') << counter << " " << *it << std::endl;
            }
            else {
                if (type == it->n) {
                    std::cout << "Node " << " " << *it << std::endl;
                }
            }
        }
        if (it->n == particle) {
            counterParticle++;
        }
        else if (it->n == pseudoParticle) {
            counterPseudoParticle++;
        }
        else if (it->n == domainList) {
            counterDomainList++;
        }
        counter++;
    }

    std::cout << "----------------------------------------------------------------";
    std::cout << "counterParticle:       " << counterParticle << std::endl;
    std::cout << "counterPseudoParticle: " << counterPseudoParticle << std::endl;
    std::cout << "counterDomainList:     " << counterDomainList << std::endl;
    std::cout << "Nodes:                 " << counter << std::endl; //nList.size();
    std::cout << "----------------------------------------------------------------";
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
        if (isLeaf()) {
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

void TreeNode::insert(Particle &p2insert) {

    Domain sonBox;
    int nSon = getSonBox(p2insert, sonBox);

    if (isDomainList() && son[nSon] != NULL) {
        son[nSon]->box = sonBox;
        son[nSon]->insert(p2insert);
    }
    else {
        if (son[nSon] == NULL) {
            if (isLeaf() && !isDomainList()) {
                Particle p2 = p;
                node = pseudoParticle;
                son[nSon] = new TreeNode(p2insert, sonBox);
                insert(p2);
            }
            else {
                son[nSon] = new TreeNode(p2insert, sonBox);
            }
        }
        else {
            son[nSon]->box = sonBox;
            son[nSon]->insert(p2insert);
        }
    }
}

void TreeNode::getTreeList(ParticleList &particleList) {
    for (int i=0; i<POWDIM; i++) {
        if (son[i] != NULL) {
            son[i]->getTreeList(particleList);
        }
    }
    particleList.push_back(p);
}

void TreeNode::getTreeList(NodeList &nodeList) {
    for (int i=0; i<POWDIM; i++) {
        if (son[i] != NULL) {
            son[i]->getTreeList(nodeList);
        }
    }
    nodeList.push_back(Node(p, node));
}

void TreeNode::getParticleList(ParticleList &particleList) {
    if (isLeaf() && !isDomainList()) {
        particleList.push_back(p);
    }
    else {
        for (int i=0; i<POWDIM; i++) {
            if (son[i] != NULL) {
                son[i]->getParticleList(particleList);
            }
        }
    }
}

void TreeNode::force(TreeNode &tl, tFloat diam) {
    tFloat r = 0;
    r = sqrt((p.x - tl.p.x) * (p.x -tl.p.x));
    if ((tl.isLeaf() || (diam < theta * r)) && !tl.isDomainList()) {
        if (r == 0) {
            Logger(WARN) << "Zero radius has been encoutered.";
        }
        else {
            tl.p.force(p);
        }
    }
    else {
        for (int i=0; i<POWDIM; i++) {
            force(*son[i], 0.5*diam);
        }
    }
}

void TreeNode::compX(tFloat deltaT) {
    for (int i=0; i<POWDIM; i++) {
        son[i]->compX(deltaT);
    }
    if (isLeaf() && ! isDomainList()) {
        p.updateX(deltaT);
    }
}

void TreeNode::compV(tFloat deltaT) {
    for (int i=0; i<POWDIM; i++) {
        son[i]->compV(deltaT);
    }
    if (isLeaf() && ! isDomainList()) {
        p.updateV(deltaT);
    }
}

void TreeNode::resetParticleFlags() {
    for (int i=0; i<POWDIM; i++) {
        if (son[i] != NULL) {
            son[i]->resetParticleFlags();
        }
        p.moved = false;
        p.toDelete = false;
    }
}

void TreeNode::moveLeaf(TreeNode &root) {
    for (int i=0; i<POWDIM; i++) {
        if (son[i] != NULL) {
            son[i]->moveLeaf(root);
        }
    }
    if (isLeaf() && !isDomainList() && !p.moved) {
        p.moved = true;
        if (!box.withinDomain(p.x)) {
            if (!root.box.withinDomain(p.x)) {
                Logger(INFO) << "Particle left system: " << p;
            }
            else {
                root.insert(p);
            }
            p.toDelete = true;
        }
    }
}

void TreeNode::repairTree() {
    for (int i=0; i<POWDIM; i++) {
        if (son[i] != NULL) {
            son[i]->repairTree();
        }
    }

    if (!isLeaf()) {
        int numberOfSons = 0;
        int d;
        for (int i=0; i<POWDIM; i++) {
            if (son[i] != NULL && !son[i]->isDomainList()) {
                if (son[i]->p.toDelete) {
                    delete son[i];
                    son[i] = NULL;
                }
                else {
                    numberOfSons++;
                    d = i;
                }
            }
        }
        if (!isDomainList()) {
            if (numberOfSons == 0) {
                p.toDelete = true;
            }
            else if (numberOfSons == 1) {
                if (!son[d]->isDomainList() && son[d]->isLeaf()) {
                    p = son[d]->p;
                    node = son[d]->node;
                    delete son[d];
                    son[d] = NULL;
                }
            }
        }
    }
}

