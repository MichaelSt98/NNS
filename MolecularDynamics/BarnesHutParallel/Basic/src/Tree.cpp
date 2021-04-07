//
// Created by Michael Staneker on 15.03.21.
//

#include "../include/Tree.h"

/*void FUNCTION(TreeNode *t) {
    if (t != NULL) {
        for (int i=0; i<POWDIM; i++)
            FUNCTION(t->son[i]);
        Perform the operations of the function FUNCTION on *t ;
    }
}*/

NodeList::NodeList() {
    node = particle;
    //p = p();
    next = NULL;
}

//NodeList::~NodeList() {
//    delete next;
//}

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


const char* get_node_type(int nodetype)
{
    switch (nodetype)
    {
        case 0: return "particle      ";
        case 1: return "pseudoParticle";
        case 2: return "domainList    ";
    }
}


//TODO: create key (!?)
// simple approach:
// numbering all possible nodes of each new generation consecutively (see page 343 and 344) = three bits are added to the key in each level of the tree
// key of a father note of a given node can be determined by deleting the last three bits
// the keys are global, unique and of integer type
// decomposing the tree = decomposing the keys (set a range for each process)
// more sophisticated (and better approach):
// transform key as in the following:
// - remove the leading bit (original root)
// - shift remaining bits to the left until the maximal bit length of the key type is reached
// (- denote the resulting bit word as *domain key*)
// now the keys are vertically sorted, keys of nodes within one level are unique but keys for nodes of different levels can happen to be the same
// --> less data needs to be exchanged
// [range_i, range_i+1) defines a minimal upper part of the tree, that has to be present in all processes as a copy to ensure the consistency of the global tree
// DUMMY
keytype key(TreeNode *t){
    return KEY_MAX;
}

// t for tree traversal, keynode for comparison if we are at the right key
/*keytype key(TreeNode *t, TreeNode *&keynode, keytype k, int level) {
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++){
            if (isLeaf(t->son[i])){
                if (&(t->son[i]) == &keynode) {
                    Logger(INFO) << "Key of son = " << k + i << DIM*(maxlevel-level-1);
                    return k + i << DIM*(maxlevel-level-1);
                }
            } else {
                keytype keyCandidate = key(t->son[i], keynode,
                                           k + i << DIM*(maxlevel-level-1), level+1);
                if (keyCandidate != KEY_MAX) {
                    Logger(DEBUG) << "Key found.";
                    return keyCandidate;
                }
            }
        }
    }
    return KEY_MAX;
}*/

void getParticleKeysSimple(TreeNode *t, keytype *p, int &pCounter, keytype k, int level){
    if (t != NULL){
        for (int i = 0; i < POWDIM; i++) {
            if (isLeaf(t->son[i])){
                p[pCounter] = k + (static_cast<keytype>(i) << level*DIM); // inserting key
                //Logger(DEBUG) << "Inserted particle '" << p[pCounter] << "'@" << pCounter;
                ++pCounter; // counting inserted particles
            } else {
                getParticleKeys(t->son[i], p, pCounter,
                                k + (static_cast<keytype>(i) << level*DIM), level+1); // go deeper
            }
        }
    }
}

void getParticleKeys(TreeNode *t, keytype *p, int &pCounter, keytype k, int level){
    if (t != NULL){
        for (int i = 0; i < POWDIM; i++) {
            if (isLeaf(t->son[i])){
                //Logger(DEBUG) << "Inserting son #" << i;
                p[pCounter] = (keytype)(k | ((keytype)i << (DIM*(maxlevel-level-1)))); // inserting key
                //Logger(DEBUG) << "Inserted particle '" << std::bitset<64>(p[pCounter]) << "'@" << pCounter;
	       	    pCounter++;
            } else {
                getParticleKeys(t->son[i], p, pCounter,
                                (keytype)(k | ((keytype)i << (DIM*(maxlevel-level-1)))), level+1); // go deeper
            }
        }
    }
}

void createRanges(TreeNode *root, int N, SubDomainKeyTree *s) {

    s->range = new keytype[s->numprocs+1];
    keytype *pKeys = new keytype[N+1]; //TODO: N+1 instead of N due to valgrind but why???
    //Logger(ERROR) << "N = " << N;
    for (int i=0; i<N; i++) {
        pKeys[i] = 0UL;
    }

    int pIndex{ 0 };
    getParticleKeys(root, pKeys, pIndex);
    // sort keys in ascending order
    std::sort(pKeys, pKeys+N);

    s->range[0] = 0UL; // range_0 = 0

    const int ppr = (N % s->numprocs != 0) ? N/s->numprocs+1 : N/s->numprocs; // particles per range

    for (int i=1; i<s->numprocs; i++){
        s->range[i] = pKeys[i*ppr];
        Logger(DEBUG) << "Computed range[" << i << "] = " << std::bitset<64>(s->range[i]);
        Logger(DEBUG) << s->range[i];
    }
    s->range[s->numprocs] = KEY_MAX;

    /*for (int i=0; i<N; i++){
        Logger(WARN) << "pKeys[" << i << "] = " << std::bitset<64>(pKeys[i])
                     << ", proc = " << key2proc(pKeys[i], s);
    }*/

    delete[] pKeys;
}

int key2proc(keytype k, SubDomainKeyTree *s) {
    for (int i=0; i<s->numprocs; i++) { //1
        if (k >= s->range[i] && k < s->range[i+1]) {
            //std::cout << "key2proc: " << i << std::endl;
            return i;
        }
    }
    //Logger(ERROR) << "key2proc(k= " << k << "): -1!";
    return -1; // error
}

//version described in MolecularDynamics
/*int key2proc(keytype k, SubDomainKeyTree *s) {
    for (int i=1; i<=s->numprocs; i++) { //1
        if (k >= s->range[i]) {
            //std::cout << "key2proc: " << i << std::endl;
            return i-1;
        }
    }
    return -1; // error
}*/

// initial call: createDomainList(root, 0, 0, s)
void createDomainList(TreeNode *t, int level, keytype k, SubDomainKeyTree *s) {
    t->node = domainList;
    int p1 = key2proc(k, s);
    int p2 = key2proc(k | ~(~0L << DIM*(maxlevel-level)),s);
    if (p1 != p2) {
        for (int i = 0; i < POWDIM; i++) {
            t->son[i] = (TreeNode *) calloc(1, sizeof(TreeNode));
            createDomainList(t->son[i], level + 1,  (keytype)(k | ((keytype)i << (DIM*(maxlevel-level-1)))), s);
        }
    }
}

bool isLeaf(TreeNode *t) {
    if (t != NULL) {
        for (int i=0; i<POWDIM; i++) {
            if (t->son[i] != NULL) {
                return false;
            }
        }
        return true;
    }
    return false;
}

//TODO: do not insert particle data within domainList nodes, instead:
// insert recursively into its son nodes (may cause particles lying deeper in the tree)
// AND: insert certain pseudoparticles from other processes into its local tree to be able to compute the force on its particles.
// Therefore, extend the routine in such a way that a domainList node can be inserted into a given tree.
// Then, the associated decomposition of the domain changes. To this end, pseudoparticles can be simply re-flagged as domainList there.
// However, if a real particle is encountered, create a domainList node and insert the particle as a son node.
// reflagging ?!
void insertTree(Particle *p, TreeNode *t) {
    // determine the son b of t in which particle p is located
    // compute the boundary data of the subdomain of the son node and store it in t->son[b].box;
    Box sonbox;
    int b = sonNumber(&t->box, &sonbox, p);

    if (t->node == domainList && t->son[b] != NULL) {
        t->son[b]->box = t->box; //sonbox; //t->box; //sonbox; //TODO: sonbox or t->box ?
        insertTree(p, t->son[b]);
    }
    else {
        if (t->son[b] == NULL) {
            if (isLeaf(t) && t->node != domainList) {
                Particle p2 = t->p;
                t->node = pseudoParticle;
                t->son[b] = (TreeNode *) calloc(1, sizeof(TreeNode));
                t->son[b]->p = *p;
                t->son[b]->box = sonbox;
                insertTree(&p2, t);
            } else {
                t->son[b] = (TreeNode *) calloc(1, sizeof(TreeNode));
                t->son[b]->p = *p;
                t->son[b]->box = sonbox;
            }
        } else {
            t->son[b]->box = sonbox;
            insertTree(p, t->son[b]);
        }
    }
}

int sonNumber(Box *box, Box *sonbox, Particle *p) {
    int b = 0;
    for (int d = DIM - 1; d >= 0; d--) {
        if (p->x[d] < 0.5 * (box->upper[d] + box->lower[d])) {
            b = 2 * b;
            sonbox->lower[d] = box->lower[d];
            sonbox->upper[d] = .5 * (box->upper[d] + box->lower[d]);
        } else {
            b = 2 * b + 1;
            sonbox->lower[d] = .5 * (box->upper[d] + box->lower[d]);
            sonbox->upper[d] = box->upper[d];
        }
    }
    return b;
}

void compPseudoParticles(TreeNode *t) {
    if (t != NULL) {
        for (int i=0; i<POWDIM; i++) {
            compPseudoParticles(t->son[i]);
        }
        if (!isLeaf(t)) {
            if (t->node != domainList) {
                t->node = pseudoParticle; //TODO: new (correct?)
            }
            t->p.m = 0;
            for (int d = 0; d < DIM; d++) {
                t->p.x[d] = 0;
            }
            for (int j = 0; j < POWDIM; j++) {
                if (t->son[j] != NULL) {
                    t->p.m += t->son[j]->p.m;
                    for (int d = 0; d < DIM; d++) {
                        t->p.x[d] += t->son[j]->p.m * t->son[j]->p.x[d];
                    }
                }
            }
            for (int d = 0; d < DIM; d++) {
                t->p.x[d] = t->p.x[d] / t->p.m;
            }
        }
    }
}

// adapted to parallel implementation
void compF_BH(TreeNode *t, TreeNode *root, float diam, SubDomainKeyTree *s, keytype k, int level) {
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++) {
            compF_BH(t->son[i], root, diam, s, (keytype)(k | ((keytype)i << (DIM*(maxlevel-level-1)))), level+1);
        }
        // start of the operation on *t
        // TODO: check if domainList check is needed
        if (isLeaf(t) && key2proc(k, s) == s->myrank && t->node != domainList) {
            for (int d = 0; d < DIM; d++) {
                t->p.F[d] = 0;
            }
            //Logger(DEBUG) << "compF_BH(): Calculation force";
            force_tree(t, root, diam);
        }
    }
}

void force_tree(TreeNode *tl, TreeNode *t, float diam) {
    //if (t == tl){
    //    Logger(INFO) << "Catching same particle should work.";
    //}
    //if ((t != tl) && (t != NULL)) {
    if (t != NULL) {
        float r = 0;
        for (int d=0; d<DIM; d++) {
            //r += sqrt(abs(t->p.x[d] - tl->p.x[d]));
            r += (t->p.x[d] - tl->p.x[d])*(t->p.x[d] - tl->p.x[d]);
        }
        r = sqrt(r);
        if (((isLeaf(tl)) || (diam < theta * r)) && tl->node != domainList) {
            // check if radius is zero, which means force should not be computed
            if (r == 0){
                Logger(WARN) << "Zero radius has been encoutered.";
            } else {
                //Logger(DEBUG) << "force_tree(): calling force";
                force(&tl->p, &t->p);
            }
        } else {
            for (int i = 0; i < POWDIM; i++) {
                force_tree(t, t->son[i], .5 * diam);
            }
        }
    }
}

void compX_BH(TreeNode *t, float delta_t) {
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++)
            compX_BH(t->son[i], delta_t);
        if (isLeaf(t) && t->node != domainList) {
            updateX(&t->p, delta_t);
        }
    }
}

void compV_BH(TreeNode *t, float delta_t) {
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++)
            compV_BH(t->son[i], delta_t);
        if (isLeaf(t)) {
            updateV(&t->p, delta_t);
        }
    }
}

void moveParticles_BH(TreeNode *root) {
    setFlags(root);
    moveLeaf(root, root);
    repairTree(root);
}

void setFlags(TreeNode *t) {
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++) {
            setFlags(t->son[i]);
        }
        t->p.moved = false;
        t->p.todelete = false;
    }
}

void moveLeaf(TreeNode *t, TreeNode *root) {
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++) {
            moveLeaf(t->son[i], root);
        }
        if ((isLeaf(t)) && t->node != domainList && (!t->p.moved)) {
            t->p.moved = true;
            if (!particleWithinBox(t->p, t->box)) {
                insertTree(&t->p, root);
                t->p.todelete = true;
            }
        }
    }
}

//parallel change: do not delete if domainList node
void repairTree(TreeNode *t) {
    if (t != NULL) {
        for (int i=0; i<POWDIM; i++) {
            repairTree(t->son[i]); // recursive call according to algorithm 8.1
        }
        if (!isLeaf(t)) { // && t->node != domainList) {
            int numberofsons = 0;
            int d;
            for (int i = 0; i < POWDIM; i++) {
                if (t->son[i] != NULL && t->son[i]->node != domainList) {
                    if (t->son[i]->p.todelete) {
                        //Logger(ERROR) << "delete (0) t-son x = " << t->son[i]->p.x[0];
                        free(t->son[i]);
                        t->son[i] = NULL;
                    }
                    else {
                        numberofsons++;
                        d = i;
                    }
                }
            }
            if (t->node != domainList) {
                if (numberofsons == 0) {
                    // *t is an *empty* leaf node and can be deleted
                    t->p.todelete = true;
                    //Logger(DEBUG) << "Empty leaf node";
                } else if (numberofsons == 1) {
                    // *t adopts the role of its only son node and
                    // the son node is deleted directly
                    //Logger(ERROR) << "parent x = " << t->p.x[0] << ", node = " << t->node;
                    if (t->son[d]->node != domainList && isLeaf(t->son[d])) {
                        // son is an only son and particle
                        t->p = t->son[d]->p;
                        //t->p.todelete = false;
                        t->node = t->son[d]->node;
                        //free(&t->son[d]->p);
                        //Logger(ERROR) << "delete (1) t-son x = " << t->son[d]->p.x[0]
                          //                  << ", node = " << get_node_type(t->son[d]->node);
                        //t->son[d]->p.todelete = true;
                        free(t->son[d]); // TODO: Check if this makes any sense at all ?!
                        //free(&t->son[d]->p);
                        t->son[d] = NULL; // TODO: May be redundant
                    }
                }
            }
        }
    }
}

void output_tree(TreeNode *t, bool detailed, bool onlyParticles) {

    int counterParticle = 0;
    int counterPseudoParticle = 0;
    int counterDomainList = 0;

    int nNodes = get_tree_node_number(t);
    Particle * pArray;
    nodetype * nArray;
    pArray = new Particle[nNodes];
    nArray = new nodetype[nNodes];
    get_tree_array(t, pArray, nArray);

    Logger(INFO) << "-------------------------------------------------------------------------";

    for (int i=nNodes-1; i>=0; i--) {
        if (nArray[i] == 0) {
            counterParticle++;
        }
        else if (nArray[i] == 1) {
            counterPseudoParticle++;
        }
        else {
            counterDomainList++;
        }
        if (detailed) {
            if (onlyParticles) {
                if (nArray[i] == particle) {
                    Logger(INFO) << "\tnodetype: " << get_node_type(nArray[i]) << " ["
                                 << (pArray[i].todelete ? "true " : "false")
                                 << "]  x = (" << pArray[i].x[0] << ", "
                                 << pArray[i].x[1] << ", "
                                 << pArray[i].x[2] << ")";
                }
            }
            else {
                Logger(INFO) << "\tnodetype: " << get_node_type(nArray[i]) << " ["
                             << (pArray[i].todelete ? "true " : "false")
                             << "]  x = (" << pArray[i].x[0] << ", "
                             << pArray[i].x[1] << ", "
                             << pArray[i].x[2] << ")";
            }
        }
    }

    Logger(INFO) << "-------------------------------------------------------------------------";
    Logger(INFO) << "NUMBER OF NODES:            " << nNodes;
    Logger(INFO) << "amount of particles:        " << counterParticle;
    Logger(INFO) << "amount of pseudoParticles:  " << counterPseudoParticle;
    Logger(INFO) << "amount of domainList nodes: " << counterDomainList;
    Logger(INFO) << "-------------------------------------------------------------------------";

    delete [] nArray;
    delete [] pArray;
}

void output_particles(TreeNode *t) {
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++) {
            output_particles(t->son[i]);

        }
        if (isLeaf(t) && t->node != domainList) {
            Logger(DEBUG) << "\tparticle x = (" << t->p.x[0] << ", " << t->p.x[1] << ", " << t->p.x[2] << ")";
        }
    }
}

NodeList* build_tree_list(TreeNode *t, NodeList *nLst) {
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++) {
            nLst = build_tree_list(t->son[i], nLst);
        }
        nLst->p = t->p;
        nLst->node = t->node;
        nLst->next = new NodeList;
        return nLst->next;
    }
    return nLst;
}

ParticleList* build_particle_list(TreeNode *t, ParticleList *pLst){
    if (t != NULL) {
        if (isLeaf(t) && t->node != domainList) {
            pLst->p = t->p;
            pLst->next = new ParticleList;
            return pLst->next;
        } else {
            for (int i = 0; i < POWDIM; i++) {
                pLst = build_particle_list(t->son[i], pLst);
            }
        }
    }
    return pLst;
}

void get_domain_list_nodes(TreeNode *t, ParticleList *pList, int &pCounter) {
    if (t != NULL){
        ParticleList * current;
        if (t->node == domainList) { // && isLowestDomainListNode(t)) {
            current = pList;
            for (int j=0; j<pCounter; j++) {
                current = current->next;
            }
            current->p = t->p;
            current->next = new ParticleList;
            pCounter++;
        }
        for (int i = 0; i < POWDIM; i++) {
            get_domain_list_nodes(t->son[i], pList, pCounter);
        }
    }
}

void get_lowest_domain_list_nodes(TreeNode *t, ParticleList *pList, int &pCounter) {
    //Logger(ERROR) << "get_lowest_domain_list_nodes";
    if (t != NULL){
        ParticleList * current;
        if (t->node == domainList && isLowestDomainListNode(t)) {
            current = pList;
            for (int j=0; j<pCounter; j++) {
                current = current->next;
            }
            current->p = t->p;
            current->next = new ParticleList;
            //Logger(DEBUG) << "Adding lowest domain list node";
            pCounter++;
        }
        for (int i = 0; i < POWDIM; i++) {
            get_lowest_domain_list_nodes(t->son[i], pList, pCounter);
        }
    }
}

void get_lowest_domain_list_nodes(TreeNode *t, ParticleList *pList, KeyList *kList, int &pCounter, keytype k, int level) {
    //Logger(ERROR) << "get_lowest_domain_list_nodes";
    if (t != NULL){
        ParticleList * current;
        KeyList * kCurrent;
        if (t->node == domainList && isLowestDomainListNode(t)) {
            current = pList;
            kCurrent = kList;
            for (int j=0; j<pCounter; j++) {
                current = current->next;
                kCurrent = kCurrent->next;
            }
            current->p = t->p;
            current->next = new ParticleList;
            //Logger(INFO) << "k = " << std::bitset<64>(k);
            kCurrent->k = k; //(unsigned long)(k << DIM*(maxlevel-level-1)); //k+i ?!
            //Logger(INFO) << "->k = " << std::bitset<64>(kCurrent->k);
            //Logger(INFO) << "DIM*(maxlevel-level-1) = " << DIM*(maxlevel-level-1);
            kCurrent->next = new KeyList;
            //Logger(DEBUG) << "Adding lowest domain list node";
            pCounter++;
        }
        for (int i = 0; i < POWDIM; i++) {
            get_lowest_domain_list_nodes(t->son[i], pList, kList, pCounter,
                                         (keytype)(k | ((keytype)i << (DIM*(maxlevel-level-1)))), level+1);
        }
    }
}

void zero_lowest_domain_list_nodes(TreeNode *t) {
    if (t != NULL){
        if (t->node == domainList && isLowestDomainListNode(t)) {

            t->p.x[0] = 0;
            t->p.x[1] = 0;
            t->p.x[2] = 0;
            t->p.m = 0;

        }
        for (int i = 0; i < POWDIM; i++) {
            zero_lowest_domain_list_nodes(t->son[i]);
        }
    }
}

void update_lowest_domain_list_nodes_moments_masses(TreeNode *t, int &pCounter, float * masses, float * moments) {
    if (t != NULL){
        if (t->node == domainList && isLowestDomainListNode(t)) {

            t->p.x[0] += moments[pCounter * 3];
            t->p.x[1] += moments[pCounter * 3 + 1];
            t->p.x[2] += moments[pCounter * 3 + 2];

            t->p.m += masses[pCounter];

            pCounter++;
        }
        for (int i = 0; i < POWDIM; i++) {
            update_lowest_domain_list_nodes_moments_masses(t->son[i], pCounter, masses, moments);
        }
    }
}

void update_lowest_domain_list_nodes_com(TreeNode *t) {
    if (t != NULL){
        if (t->node == domainList && isLowestDomainListNode(t)) {

            if (t->p.m > 0) {
                t->p.x[0] = t->p.x[0] / t->p.m;
                t->p.x[1] = t->p.x[1] / t->p.m;
                t->p.x[2] = t->p.x[2] / t->p.m;
            }
        }
        for (int i = 0; i < POWDIM; i++) {
            update_lowest_domain_list_nodes_com(t->son[i]);
        }
    }
}



void update_lowest_domain_list_nodes(TreeNode *t, int &pCounter, float * masses, float * moments) {
    zero_lowest_domain_list_nodes(t);
    update_lowest_domain_list_nodes_moments_masses(t, pCounter, masses, moments);
    update_lowest_domain_list_nodes_com(t);
}

int get_tree_node_number(TreeNode *root) {
    NodeList * nLst;
    nLst = new NodeList;
    int nIndex = 0;
    build_tree_list(root, nLst);
    while (nLst->next) {
        nIndex++;
        NodeList* old = nLst;
        nLst = nLst->next;
        delete old;
    }
    //delete nLst; //deleteNodeList(nLst); //delete nLst;
    return nIndex;
}

int get_tree_array(TreeNode *root, Particle *&p, nodetype *&n) {
    NodeList * nLst;
    nLst = new NodeList;
    int nIndex = 0;

    build_tree_list(root, nLst);
    if (nLst->next) {
        while (nLst->next) {
            p[nIndex] = nLst->p;
            n[nIndex] = nLst->node;
            NodeList * old = nLst;
            nLst = nLst->next;
            delete old;
            ++nIndex;
        }
    }
    //delete nLst; //deleteNodeList(nLst); //delete nLst;
    return nIndex;
}

int get_particle_array(TreeNode *root, Particle *&p){
    ParticleList * pLst;
    pLst = new ParticleList;

    build_particle_list(root, pLst);

    ParticleList * current;
    current = pLst;

    int pLength = 0;
    while(current->next) {
        pLength++;
        current = current->next;
    }

    p = new Particle[pLength];

    int pIndex { 0 };
    while(pLst->next){
        p[pIndex] = pLst->p;
        ParticleList * old = pLst;
        //Logger(DEBUG) << "Adding to *p: x = (" << p[pIndex].x[0] << ", " << p[pIndex].x[1] << ", " << p[pIndex].x[2] << ")";
        pLst = pLst->next;
        delete old;
        ++pIndex;
    }
    return pIndex;
}

int get_domain_list_array(TreeNode *root, Particle *&pArray) {
    int pCounter = 0;
    ParticleList *pList;
    pList = new ParticleList;
    get_domain_list_nodes(root, pList, pCounter);
    pArray = new Particle[pCounter];
    for (int i=0; i<pCounter; i++) {
        pArray[i] = pList->p;
        ParticleList* old = pList;
        pList = pList->next;
        delete old;
    }
    //delete pList; //deleteParticleList(pList);
    return pCounter;
}

int get_lowest_domain_list_array(TreeNode *root, Particle *&pArray) {
    int pCounter = 0;
    ParticleList *pList;
    pList = new ParticleList;

    get_lowest_domain_list_nodes(root, pList, pCounter);
    Logger(INFO) << "pCounter from lowest domain list array: " << pCounter;
    pArray = new Particle[pCounter];

    for (int i=0; i<pCounter; i++) {
        pArray[i] = pList->p;
        ParticleList* old = pList;
        pList = pList->next;
        delete old;
    }
    //delete pList; //deleteParticleList(pList);
    return pCounter;
}

int get_lowest_domain_list_array(TreeNode *root, Particle *&pArray, keytype *&kArray) {
    int pCounter = 0;
    ParticleList *pList;
    pList = new ParticleList;
    KeyList *kList;
    kList = new KeyList;

    get_lowest_domain_list_nodes(root, pList, kList, pCounter);
    Logger(INFO) << "pCounter from lowest domain list array: " << pCounter;
    pArray = new Particle[pCounter];
    kArray = new keytype[pCounter];

    for (int i=0; i<pCounter; i++) {
        pArray[i] = pList->p;
        kArray[i] = kList->k;
        ParticleList* old = pList;
        KeyList* kOld = kList;
        pList = pList->next;
        kList = kList->next;
        delete old;
        delete kOld;
    }
    //delete pList; //deleteParticleList(pList);
    return pCounter;
}

/*int get_domain_moments_array(TreeNode *root, float * moments) {
    Particle *pArray;
    int pCounter = get_domain_list_array(root, pArray);
    for (int i=0; i<pCounter; i++) {
        Logger(INFO) << "m[" << i << "]  = " << pArray[i].m << "   x = " << pArray[i].x[0];
        //Logger(INFO) << "Bin ich druff oder C++?";
    }
    return pCounter;
}*/

void freeTree_BH(TreeNode *root) {
    if (root != NULL) {
        for (int i=0; i<POWDIM; i++) {
            if (root->son[i] != NULL) {
                freeTree_BH(root->son[i]);
                if (root->son[i] != NULL) {
                    free(root->son[i]);
                }
            }
        }
    }
}

/*

void sendParticles(TreeNode *root, SubDomainKeyTree *s) {
    //allocate memory for s->numprocs particle lists in plist;
    //initialize ParticleList plist[to] for all processes to;
    buildSendlist(root, s, plist);
    repairTree(root); // here, domainList nodes may not be deleted
    for (int i=1; i<s->numprocs; i++) {
        int to = (s->myrank+i)%s->numprocs;
        int from = (s->myrank+s->numprocs-i)%s->numprocs;
        //send particle data from plist[to] to process to;
        //receive particle data from process from;
        //insert all received particles p into the tree using insertTree(&p, root);
    }
    delete plist;
}

void buildSendlist(TreeNode *t, SubDomainKeyTree *s, ParticleList *plist) {
    called recursively as in Algorithm 8.1;
    // start of the operation on *t
    int proc;
    if ((*t is a leaf node) && ((proc = key2proc(key(*t), s)) != s->myrank)) {
        // the key of *t can be computed step by step in the recursion
        insert t->p into list plist[proc];
        mark t->p as to be deleted;
    }
    // end of the operation on *t }
}

 */

int getParticleListLength(ParticleList *plist) {
    ParticleList * current = plist;
    int count = 0;
    while (current->next) {
        count++;
        current = current->next;
    }
    //count++;
    return count;
}

//TODO: implement sendParticles (Sending Particles to Their Owners and Inserting Them in the Local Tree)
// determine right amount of memory which has to be allocated for the `buffer`,
// by e.g. communicating the length of the message as prior message or by using other MPI commands
void sendParticles(TreeNode *root, SubDomainKeyTree *s) {
    //allocate memory for s->numprocs particle lists in plist;
    //initialize ParticleList plist[to] for all processes to;
    ParticleList * plist;
    plist = new ParticleList[s->numprocs];

    int pIndex[s->numprocs];
    for (int proc = 0; proc < s->numprocs; proc++) {
        pIndex[proc] = 0;
    }

    //Logger(INFO) << "Range    = " << std::bitset<64>(s->range[1]);
    buildSendlist(root, root, s, plist, pIndex, 0UL, 0); //TODO: something to be changed?

    //for (int proc=0; proc<s->numprocs; proc++) {
    //    Logger(ERROR) << "pIndex: " << pIndex[proc];
    //}

    for (int proc = 0; proc < s->numprocs; proc++) {
        ParticleList *current = &plist[proc]; // needed not to 'consume' plist
        for (int i=0; i<pIndex[proc]; i++){
            //Logger(DEBUG) << "x2send = (" << current->p.x[0] << ", " << current->p.x[1] << ", " << current->p.x[2] << ")";
            current = current->next;
        }
    }

    //Logger(WARN) << "sendParticles(): BEFORE repairTree()";
    //output_tree(root);

    repairTree(root); // here, domainList nodes may not be deleted //TODO: something to be changed?

    //Logger(WARN) << "sendParticles(): AFTER repairTree()";
    //output_tree(root);

    Particle ** pArray = new Particle*[s->numprocs];

    int *plistLengthSend;
    plistLengthSend = new int[s->numprocs];
    plistLengthSend[s->myrank] = -1;

    int *plistLengthReceive;
    plistLengthReceive = new int[s->numprocs];
    plistLengthReceive[s->myrank] = -1; // nothing to receive from yourself

    for (int proc=0; proc < s->numprocs; proc++) {
        if (proc != s->myrank) {
            plistLengthSend[proc] = getParticleListLength(&plist[proc]);

            Logger(INFO) << "plistLengthSend[" << proc << "] = " << plistLengthSend[proc];

            pArray[proc] = new Particle[plistLengthSend[proc]];
            ParticleList * current = &plist[proc];
            for (int i = 0; i < plistLengthSend[proc]; i++) {
                pArray[proc][i] = current->p;
                current = current->next;
            }
        }
    }

    int reqCounter = 0;
    MPI_Request reqMessageLengths[s->numprocs-1];
    MPI_Status statMessageLengths[s->numprocs-1];

    //send plistLengthSend and receive plistLengthReceive
    for (int proc=0; proc < s->numprocs; proc++) {
        if (proc != s->myrank) {
            MPI_Isend(&plistLengthSend[proc], 1, MPI_INT, proc, 17, MPI_COMM_WORLD, &reqMessageLengths[reqCounter]);
            MPI_Recv(&plistLengthReceive[proc], 1, MPI_INT, proc, 17, MPI_COMM_WORLD, &statMessageLengths[reqCounter]);
            reqCounter++;
        }
    }
    MPI_Waitall(s->numprocs-1, reqMessageLengths, statMessageLengths);

    //sum over to get total amount of particles to receive
    int receiveLength = 0;
    for (int proc=0; proc < s->numprocs; proc++) {
        if (proc != s->myrank) {
            receiveLength += plistLengthReceive[proc];
        }
    }

    //Logger(INFO) << "receiveLength: " << receiveLength;


    Logger(INFO) << "receiveLength = " << receiveLength;

    // allocate missing (sub)array for process rank
    pArray[s->myrank] = new Particle[receiveLength];

    //for (int proc=0; proc<s->numprocs; proc++) {
    //    for (int i = 0; i < plistLengthSend[proc]; i++) {
    //        Logger(INFO) << "Sending particle pArray[" << proc << "][" << i << "] : " << pArray[proc][i].x[0];
    //    }
    //}

    MPI_Request reqParticles[s->numprocs-1];
    MPI_Status statParticles[s->numprocs-1];

    //send and receive particles
    reqCounter = 0;
    int receiveOffset = 0;
    for (int proc=0; proc < s->numprocs; proc++) {
        if (proc != s->myrank) {
            MPI_Isend(pArray[proc], plistLengthSend[proc], mpiParticle, proc, 17, MPI_COMM_WORLD, &reqParticles[reqCounter]);
            MPI_Recv(pArray[s->myrank] + receiveOffset, plistLengthReceive[proc], mpiParticle, proc, 17,
                         MPI_COMM_WORLD, &statParticles[reqCounter]);
            receiveOffset += plistLengthReceive[proc];
            reqCounter++;
        }
    }
    MPI_Waitall(s->numprocs-1, reqParticles, statParticles);

    for (int i=0; i<receiveLength; i++) {
        //Logger(INFO) << "Inserting particle pArray[" << i << "] : " << pArray[s->myrank][i].x[0];
        insertTree(&pArray[s->myrank][i], root);
    }

    delete [] plist; //deleteParticleList(plist); //delete [] plist; //delete plist;
    delete [] plistLengthSend;
    delete [] plistLengthReceive;
    for (int proc=0; proc < s->numprocs; proc++) {
        delete [] pArray[proc];
    }
    delete [] pArray;
}

//TODO: implement buildSendlist (Sending Particles to Their Owners and Inserting Them in the Local Tree)
void buildSendlist(TreeNode *root, TreeNode *t, SubDomainKeyTree *s, ParticleList *plist, int *pIndex, keytype k, int level) {
    ParticleList * current;
    if (t != NULL) {
        // start of the operation on *t
        int proc;
        if ((isLeaf(t)) && ((proc = key2proc(k, s)) != s->myrank) && t->node != domainList) {
            current = &plist[proc];
            //Logger(DEBUG) << "key2send = " << std::bitset<64>(k)
              //      << "; x = (" << t->p.x[0] << ", " << t->p.x[1] << ", " << t->p.x[2] << ")";
            //Logger(INFO) << "proc = " << proc;
            // the key of *t can be computed step by step in the recursion //TODO: compute key of *t
            //insert t->p into list plist[proc];
            for (int i=0; i<pIndex[proc]; i++) {
                current = current->next;
            }
            //Logger(INFO) << "Adding particle to send: " << t->p.x[0];
            current->p = t->p;
            current->next = new ParticleList; //TODO: similar problem as with get_particle_array() ?!
            //mark t->p as to be deleted;
            //Logger(ERROR) << "buildSendList to be deleted!";
            t->p.todelete = true;
            pIndex[proc]++;
        }
        for (int i=0; i<POWDIM; i++) {
            buildSendlist(root, t->son[i], s, plist, pIndex, (keytype)(k | ((keytype)i << (DIM*(maxlevel-level-1)))), level+1);
        }
        // end of the operation on *t }
    }
}

//TODO: implement compPseudoParticlespar (Parallel Computation of the Values of the Pseudoparticles)
void compPseudoParticlespar(TreeNode *root, SubDomainKeyTree *s) {

    compLocalPseudoParticlespar(root);

    Particle * pArray;
    keytype  * kArray;
    //int pLength = get_domain_list_array(root, pArray);
    int pLength = get_lowest_domain_list_array(root, pArray, kArray);

    //Logger(INFO) << "pLength = " << pLength;
    //for (int i=0; i<pLength; i++) {
        //Logger(ERROR) << "pArray k[" << i << "]  = " << std::bitset<64>(kArray[i]) << "   x = " << pArray[i].x[0];
    //}

    float moments[pLength*3];
    float global_moments[3*pLength];
    float masses[pLength];
    float global_masses[pLength];

    for (int i=0; i<pLength; i++) {
        moments[3*i] = pArray[i].x[0] * pArray[i].m;
        moments[3*i+1] = pArray[i].x[1] * pArray[i].m;
        moments[3*i+2] = pArray[i].x[2] * pArray[i].m;
        masses[i] = pArray[i].m;
    }


    /*Particle * pArrayDomainList;
    int pLengthDomainList = get_domain_list_array(root, pArrayDomainList);

    Logger(INFO) << "pLengthDomainList = " << pLengthDomainList;
    for (int i=0; i<pLengthDomainList; i++) {
        Logger(ERROR) << "pArrayDomainList m[" << i << "]  = " << pArrayDomainList[i].m << "   x = "
                            << pArrayDomainList[i].x[0];
    }*/

    //if (s->myrank == ) {
    //output_tree(root, false);
    //}

    //MPI_Allreduce(..., {mass, moments} of the lowest domainList nodes, MPI_SUM, ...);
    /*MPI_Allreduce(
            void* send_data,
            void* recv_data,
            int count,
            MPI_Datatype datatype,
            MPI_Op op,
            MPI_Comm communicator)*/

    MPI_Allreduce(&moments, &global_moments, 3*pLength, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&masses, &global_masses, pLength, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    /*for (int i=0; i<pLength; i++) {
        Logger(INFO) << "       masses[" << i << "] = " << masses[i];
        Logger(INFO) << "global masses[" << i << "] = " << global_masses[i];
    }*/

    int dCounter = 0;

    /*Particle * dArray1;
    int dLength1 = get_domain_list_array(root, dArray1);

    for (int i = 0; i < dLength1; i++) {
        Logger(INFO) << "Domain dArray1[" << i << "].x = " << dArray1[i].x[0];
    }*/

    update_lowest_domain_list_nodes(root, dCounter, global_masses, global_moments);

    Logger(ERROR) << "dCounter = " << dCounter;

    compDomainListPseudoParticlespar(root);
}

/*bool isLowestDomainListNode(TreeNode *t){
    if (t != NULL){
        if (t->node == domainList && !isLeaf(t)) {
            for (int i=0; i<POWDIM; i++){
                if (t->son[i] && t->son[i]->node == domainList) {
                    return false;
                }
            }
            return true;
        }
    }
    return false;
}*/

bool isLowestDomainListNode(TreeNode *t){
    if (t != NULL){
        if (t->node == domainList) {
            if (isLeaf(t)) {
                return true;
            } else {
                for (int i = 0; i < POWDIM; i++) {
                    if (t->son[i] && t->son[i]->node == domainList) {
                            return false;
                    }
                }
                return true;
            }
        }
    }
    return false;
}

//TODO: implement compLocalPseudoParticlespar (Parallel Computation of the Values of the Pseudoparticles)
void compLocalPseudoParticlespar(TreeNode *t) {
    //called recursively as in Algorithm 8.1;
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++) {
            compLocalPseudoParticlespar(t->son[i]);
        }
        // start of the operation on *t
        if (!isLeaf(t) && (t->node != domainList || isLowestDomainListNode(t))) {
        //if (!isLeaf(t) && t->node != domainList || isLowestDomainListNode(t)) {
            // operations analogous to Algorithm 8.5 (see below)
            if (t->node != domainList) { //TODO: or !isLowestDomainListNode(t)
                t->node = pseudoParticle;
            }
            t->p.m = 0;
            for (int d = 0; d < DIM; d++) {
                t->p.x[d] = 0;
            }
            for (int j = 0; j < POWDIM; j++) {
                if (t->son[j] != NULL) {
                    t->p.m += t->son[j]->p.m;
                    for (int d = 0; d < DIM; d++) {
                        t->p.x[d] += t->son[j]->p.m * t->son[j]->p.x[d];
                    }
                }
            }
            for (int d = 0; d < DIM; d++) {
                t->p.x[d] = t->p.x[d] / t->p.m;
            }
        }
    }
    // end of the operation on *t
}

//TODO: implement compDomainListPsudoParticlespar (Parallel Computation of the Values of the Pseudoparticles)
void compDomainListPseudoParticlespar(TreeNode *t) {
    //called recursively as in Algorithm 8.1 for the coarse domainList-tree;
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++) {
            compDomainListPseudoParticlespar(t->son[i]);
        }
        // start of the operation on *t
        if (t->node == domainList && !isLowestDomainListNode(t)) {
            // operations analogous to Algorithm 8.5 (see below)
            t->p.m = 0;
            for (int d = 0; d < DIM; d++) {
                t->p.x[d] = 0;
            }
            for (int j = 0; j < POWDIM; j++) {
                if (t->son[j] != NULL) {
                    t->p.m += t->son[j]->p.m;
                    for (int d = 0; d < DIM; d++) {
                        t->p.x[d] += t->son[j]->p.m * t->son[j]->p.x[d];
                    }
                }
            }
            for (int d = 0; d < DIM; d++) {
                if (t->p.m != 0) {
                    t->p.x[d] = t->p.x[d] / t->p.m;
                }
            }
        }
        // end of the operation on *t
    }
}

/*
 * Algorithm 8.5:
 void compPseudoParticles(TreeNode *t) {
    if (t != NULL) {
        for (int i=0; i<POWDIM; i++) {
            compPseudoParticles(t->son[i]);
        }
        if (!isLeaf(t)) {
            t->p.m = 0;
            for (int d = 0; d < DIM; d++) {
                t->p.x[d] = 0;
            }
            for (int j = 0; j < POWDIM; j++) {
                if (t->son[j] != NULL) {
                    t->p.m += t->son[j]->p.m;
                    for (int d = 0; d < DIM; d++) {
                        t->p.x[d] += t->son[j]->p.m * t->son[j]->p.x[d];
                    }
                }
            }
            for (int d = 0; d < DIM; d++) {
                t->p.x[d] = t->p.x[d] / t->p.m;
            }
        }
    }
}
 */

/*
 * The computation of the distance of a cell to a particle can be implemented
with appropriate case distinctions. One has to test whether the particle lies
left, right, or inside the cell along each coordinate direction.
 */
float smallestDistance(TreeNode *td, TreeNode *t) {
    //smallest distance from t->p.x to cell td->box;
    float dx;
    if (t->p.x[0] < td->box.lower[0]) {
        // "left" regarding 0-axis
        dx = td->box.lower[0] - t->p.x[0];
    }
    else if (t->p.x[0] > td->box.upper[0]) {
        // "right" regarding 0-axis
        dx = t->p.x[0] - td->box.upper[0];
    }
    else {
        // "inbetween" regarding 0-axis
        dx = 0.f;
    }

    float dy;
    if (t->p.x[1] < td->box.lower[1]) {
        // "left" regarding 1-axis
        dy = td->box.lower[1] - t->p.x[1];
    }
    else if (t->p.x[1] > td->box.upper[1]) {
        // "right" regarding 1-axis
        dy = t->p.x[1] - td->box.upper[1];
    }
    else {
        // "inbetween" regarding 1-axis
        dy = 0.f;
    }

    float dz;
    if (t->p.x[2] < td->box.lower[2]) {
        // "left" regarding 2-axis
        dz = td->box.lower[2] - t->p.x[2];
    }
    else if (t->p.x[2] > td->box.upper[2]) {
        // "right" regarding 2-axis
        dz = t->p.x[2] - td->box.upper[2];
    }
    else {
        // "inbetween" regarding 2-axis
        dz = 0.f;
    }

    if (dx+dy+dz == 0.f){
        //Logger(ERROR) << "Smallest distance is zero!";
    }
    return sqrt(dx*dx + dy*dy + dz*dz);
}

//TODO: implement symbolicForce (Determining Subtrees that are Needed in the Parallel Force Computation)
void symbolicForce(TreeNode *td, TreeNode *t, float diam, ParticleList *plist, SubDomainKeyTree *s) {
    if ((t != NULL) && (key2proc(key(t), s) == s->myrank)) {
        // the key of *t can be computed step by step in the recursion insert t->p into list plist;
        float r = smallestDistance(td, t); //IMPLEMENT: smallest distance from t->p.x to cell td->box;
        //Logger(DEBUG) << "diam = " << diam << "  theta * r = " << theta * r;
        if (diam >= theta * r) {
            for (int i = 0; i < POWDIM; i++) {
                symbolicForce(td, t->son[i], .5 * diam, plist, s);
            }
        }
    }
}
/**
 *
 * @param td : Domain list node which do NOT belong to my process
 * @param t : current tree node, called initially with root
 * @param diam : box side lenght of current tree node t
 * @param plist
 * @param s
 * @param pCounter
 * @param k : called initially with 0UL
 * @param level : called initially with 0
 */
void symbolicForce(TreeNode *td, TreeNode *t, float diam, ParticleList *plist, SubDomainKeyTree *s,
                   int &pCounter, keytype k, int level) {

    Logger(INFO) << "symbolicForce: " << std::bitset<64>(k) << " mapped to proc " << key2proc(k, s);
    if (t != NULL && (key2proc(k, s) == s->myrank || t->node == domainList)) {

        // the key of *t can be computed step by step in the recursion insert t->p into list plist;
        //TODO: insert t->p into list plist (where?)
        if (t->node != domainList) {
            ParticleList * current;
            current = plist;
            for (int i=0; i<pCounter; i++) {
                current = current->next;
            }
            Logger(INFO) << "symbolicForce insert x = " << t->p.x[0];
            current->p = t->p;
            current->next = new ParticleList;
            pCounter++;
        }

        float r = smallestDistance(td, t); //IMPLEMENT: smallest distance from t->p.x to cell td->box;
        //Logger(DEBUG) << "diam = " << diam << "  theta * r = " << theta * r;
        //Logger(DEBUG) << "Current node key = " << std::bitset<64>(k);
        if (diam >= theta * r) {
            for (int i = 0; i < POWDIM; i++) {
                symbolicForce(td, t->son[i], .5 * diam, plist, s, pCounter,
                              (keytype)(k | ((keytype)i << (DIM*(maxlevel-level-1)))), level+1); //TODO: is that correct?
            }
        }
    }
}

void symbolicForce(TreeNode *td, TreeNode *t, float diam, ParticleMap &pmap, SubDomainKeyTree *s,
                   keytype k, int level) {

    //Logger(INFO) << "symbolicForce: " << std::bitset<64>(k) << " mapped to proc " << key2proc(k, s);
    if (t != NULL && (key2proc(k, s) == s->myrank || t->node == domainList)) {

        // the key of *t can be computed step by step in the recursion insert t->p into list plist;
        //TODO: insert t->p into list plist (where?)
        if (t->node != domainList) {
            //Logger(INFO) << "symbolicForce insert x = " << t->p.x[0];
            bool insert = true;
            for (ParticleMap::iterator pit = pmap.begin(); pit != pmap.end(); pit++)
            {
                if (pit->second.x[0] == t->p.x[0] && pit->second.x[1] == t->p.x[1] && pit->second.x[2] == t->p.x[2]) {
                    insert = false;
                }
            }
            if (insert) {
                pmap[k] = t->p; // insert into map which has unique keys (will overwrite)
            }
        }

        float r = smallestDistance(td, t); //IMPLEMENT: smallest distance from t->p.x to cell td->box;
        //Logger(DEBUG) << "diam = " << diam << "  theta * r = " << theta * r;
        //Logger(DEBUG) << "Current node key = " << std::bitset<64>(k);
        if (diam >= theta * r) {
            for (int i = 0; i < POWDIM; i++) {
                symbolicForce(td, t->son[i], .5 * diam, pmap, s,
                              (keytype)(k | ((keytype)i << (DIM*(maxlevel-level-1)))), level+1); //TODO: is that correct?
            }
        }
    }
}

/*
 * NOTE: Force computation:
 *
 * The computation of the distance of a cell to a particle can be implemented with appropriate case distinctions.
 * One has to test whether the particle lies left, right, or inside the cell along each coordinate direction.
 * The particles to be sent are collected in lists. It could happen that a (pseudo-)particle is inserted into the list
 * several times, if several cells td are traversed for one process. Such duplicate (pseudo-)particles should be removed
 * before the communication step. This can be implemented easily via sorted lists.
 *
 */

//TODO: implement compF_BHpar (Parallel Force Computation)
/*void compF_BHpar(TreeNode *root, float diam, SubDomainKeyTree *s) {
    //allocate memory for s->numprocs particle lists in plist;
    //initialize ParticleList plist[to] for all processes to;
    //TODO: compTheta(root, root, s, plist, diam);
    for (int i=1; i<s->numprocs; i++) {
        int to = (s->myrank+i)%s->numprocs;
        int from = (s->myrank+s->numprocs-i)%s->numprocs;
        //send (pseudo-)particle data from plist[to] to process to; receive (pseudo-)particle data from process from;
        //insert all received (pseudo-)particles p into the tree using insertTree(&p, root);
    }
    //TODO: delete plist;
    //TODO: compF_BH(root, diam);
}*/

bool compareParticles(Particle p1, Particle p2) {
    /*return (p1.x[0] == p2.x[0] &&
            p1.x[1] == p2.x[1] &&
            p1.x[2] == p2.x[2]);*/
    return (sqrt(p1.x[0]*p1.x[0] + p1.x[1]*p1.x[1] + p1.x[2]*p1.x[2]) -
            sqrt(p2.x[0]*p2.x[0] + p2.x[1]*p2.x[1] + p2.x[2]*p2.x[2]));
}



//compF_BHpar analog to sendParticles()
void compF_BHpar(TreeNode *root, float diam, SubDomainKeyTree *s) {
    //allocate memory for s->numprocs particle lists in plist;
    //initialize ParticleList plist[to] for all processes to;
    ParticleList * plist;
    plist = new ParticleList[s->numprocs];

    ParticleList * uniquePlist;
    uniquePlist = new ParticleList[s->numprocs];

    ParticleMap * pmap;
    pmap = new ParticleMap[s->numprocs];

    int * pIndex;
    pIndex = new int[s->numprocs];
    //int pIndex[s->numprocs];
    for (int proc = 0; proc < s->numprocs; proc++) {
        pIndex[proc] = 0;
    }

    //compTheta(root, root, s, plist, pIndex, diam);
    compTheta(root, root, s, pmap, diam);
    //TreeNode *t, TreeNode *root, SubDomainKeyTree *s, ParticleList *plist, float diam, keytype k, int level

    for (int proc = 0; proc < s->numprocs; proc++) {
        //ParticleList *current = &plist[proc]; // needed not to 'consume' plist

        //Logger(WARN) << "pListSendLength[" << proc << "] = " << (int)pmap[proc].size();

        /*for (int i=0; i<pIndex[proc]; i++){
            Logger(DEBUG) << "BH_par2send = (" << current->p.x[0] << ", " << current->p.x[1] << ", " << current->p.x[2] << ")";
            current = current->next;
        }*/
    }

    Particle ** pArray = new Particle*[s->numprocs];

    int *plistLengthSend;
    plistLengthSend = new int[s->numprocs];
    plistLengthSend[s->myrank] = -1;

    int *plistLengthReceive;
    plistLengthReceive = new int[s->numprocs];
    plistLengthReceive[s->myrank] = -1; // nothing to receive from yourself

    for (int proc=0; proc < s->numprocs; proc++) {
        if (proc != s->myrank) {
            plistLengthSend[proc] = (int)pmap[proc].size(); //getParticleListLength(&plist[proc]);

            Logger(INFO) << "compF_BHpar(): plistLengthSend[" << proc << "] = " << plistLengthSend[proc];

            pArray[proc] = new Particle[plistLengthSend[proc]];
            ParticleList * current = &plist[proc];
            /*for (int i = 0; i < plistLengthSend[proc]; i++) {
                pArray[proc][i] = current->p;
                current = current->next;
            }*/
            int counter = 0;
            for (ParticleMap::iterator pit = pmap[proc].begin(); pit != pmap[proc].end(); pit++)
            {
                //Logger(INFO) << "pmap[" << pit->first << "] = ("
                  //           << pit->second.x[0] << ", " << pit->second.x[1] << ", " << pit->second.x[2] << "), "
                    //         << "m = " << pit->second.m;
                pArray[proc][counter] = pit->second;
                counter++;
            }
        }
    }


    int reqCounter = 0;
    MPI_Request reqMessageLengths[s->numprocs-1];
    MPI_Status statMessageLengths[s->numprocs-1];

    //send plistLengthSend and receive plistLengthReceive
    for (int proc=0; proc < s->numprocs; proc++) {
        if (proc != s->myrank) {
            MPI_Isend(&plistLengthSend[proc], 1, MPI_INT, proc, 17, MPI_COMM_WORLD, &reqMessageLengths[reqCounter]);
            MPI_Recv(&plistLengthReceive[proc], 1, MPI_INT, proc, 17, MPI_COMM_WORLD, &statMessageLengths[reqCounter]);
            reqCounter++;
        }
    }
    MPI_Waitall(s->numprocs-1, reqMessageLengths, statMessageLengths);

    //sum over to get total amount of particles to receive
    int receiveLength = 0;
    for (int proc=0; proc < s->numprocs; proc++) {
        if (proc != s->myrank) {
            receiveLength += plistLengthReceive[proc];
        }
    }

    Logger(INFO) << "compF_BHpar(): receiveLength = " << receiveLength;

    // allocate missing (sub)array for process rank
    pArray[s->myrank] = new Particle[receiveLength];

    //for (int proc=0; proc<s->numprocs; proc++) {
    //    for (int i = 0; i < plistLengthSend[proc]; i++) {
    //        Logger(INFO) << "Sending particle pArray[" << proc << "][" << i << "] : " << pArray[proc][i].x[0];
    //    }
    //}

    MPI_Request reqParticles[s->numprocs-1];
    MPI_Status statParticles[s->numprocs-1];

    //send and receive particles
    reqCounter = 0;
    int receiveOffset = 0;
    for (int proc=0; proc < s->numprocs; proc++) {
        if (proc != s->myrank) {
            MPI_Isend(pArray[proc], plistLengthSend[proc], mpiParticle, proc, 17, MPI_COMM_WORLD, &reqParticles[reqCounter]);
            MPI_Recv(pArray[s->myrank] + receiveOffset, plistLengthReceive[proc], mpiParticle, proc, 17,
                     MPI_COMM_WORLD, &statParticles[reqCounter]);
            receiveOffset += plistLengthReceive[proc];
            reqCounter++;
        }
    }
    MPI_Waitall(s->numprocs-1, reqParticles, statParticles);

    for (int i=0; i<receiveLength; i++) {
        //Logger(INFO) << "Inserting particle pArray[" << i << "] : " << pArray[s->myrank][i].x[0];
        pArray[s->myrank][i].todelete = true;
        insertTree(&pArray[s->myrank][i], root);
    }

    delete [] plist; //deleteParticleList(plist); //delete [] plist; //delete plist;
    delete [] plistLengthSend;
    delete [] plistLengthReceive;
    delete [] pmap;
    for (int proc=0; proc < s->numprocs; proc++) {
        delete [] pArray[proc];
    }
    delete [] pArray;

    compF_BH(root, root, diam, s);

}

//TODO: implement compTheta (Parallel Force Computation)
void compTheta(TreeNode *t, TreeNode *root, SubDomainKeyTree *s, ParticleList *plist, int *&pCounter, float diam, keytype k, int level) {
    //Logger(INFO) << "compTheta() ..";
    //called recursively as in Algorithm 8.1;
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++) {
            compTheta(t->son[i], root, s, plist, pCounter, diam, (keytype)(k | ((keytype)i << (DIM*(maxlevel-level-1)))), level+1);
        }
        // start of the operation on *t
        int proc;
        if ((t->node == domainList) && ((proc = key2proc(k, s)) != s->myrank)) {
            // the key of *t can be computed step by step in the recursion
            Logger(WARN) << "td x[0] = " << t->p.x[0] << ", " << std::bitset<64>(k);
            symbolicForce(t, root, diam, &plist[proc], s, pCounter[proc], 0UL, 0);
        }
    }
// end of the operation on *t
}

void compTheta(TreeNode *t, TreeNode *root, SubDomainKeyTree *s, ParticleMap *pmap, float diam, keytype k, int level) {
    //Logger(INFO) << "compTheta() ..";
    //called recursively as in Algorithm 8.1;
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++) {
            compTheta(t->son[i], root, s, pmap, diam, (keytype)(k | ((keytype)i << (DIM*(maxlevel-level-1)))), level+1);
        }
        // start of the operation on *t
        int proc;
        if ((t->node == domainList) && ((proc = key2proc(k, s)) != s->myrank)) {
            // the key of *t can be computed step by step in the recursion
            symbolicForce(t, root, diam, pmap[proc], s, 0UL, 0);
        }
    }
// end of the operation on *t
}

int gatherParticles(TreeNode *root, SubDomainKeyTree *s, Particle *&pArrayAll) {
    Particle * pArray;
    int pLength = get_particle_array(root, pArray);

    //for (int i=0; i<pCounter; i++) {
    //    Logger(DEBUG) << "pArray[" << i << "].x[0] = " << pArray[i].x[0];
    //}

    //MPI_Request reqMessageLengths[s->numprocs-1];
    //MPI_Status statMessageLengths[s->numprocs-1];

    int *pArrayReceiveLength;
    int *pArrayDisplacements;
    if (s->myrank == 0) {
        pArrayReceiveLength = new int[s->numprocs];
        pArrayDisplacements = new int[s->numprocs];
        pArrayDisplacements[0] = 0;
    }

    MPI_Gather(&pLength,
            1,
            MPI_INT,
            pArrayReceiveLength,
            1,
            MPI_INT,
            0,
            MPI_COMM_WORLD);

    int totalReceiveLength = 0;
    if (s->myrank == 0) {
        for (int i=0; i<s->numprocs; i++) {
            Logger(DEBUG) << "receiveLength[" << i << "] = " << pArrayReceiveLength[i];
            totalReceiveLength += pArrayReceiveLength[i];
        }
    }


    if (s->myrank == 0) {
        for (int i=1; i<s->numprocs; i++) {
            //Logger(DEBUG) << "receiveLength[" << i << "] = " << pArrayReceiveLength[i];
            pArrayDisplacements[i] = pArrayReceiveLength[i-1] + pArrayDisplacements[i-1];
            Logger(DEBUG) << "Displacements: " << pArrayDisplacements[i];
        }
    }

    //Particle * pArrayAll;
    if (s->myrank == 0) {
        pArrayAll = new Particle[totalReceiveLength];
    }

    MPI_Gatherv(pArray, pLength, mpiParticle, pArrayAll, pArrayReceiveLength,
                pArrayDisplacements, mpiParticle, 0, MPI_COMM_WORLD);

    /*if (s->myrank == 0) {
        for (int i = 0; i < totalReceiveLength; i++) {
            Logger(DEBUG) << "pArrayAll[" << i << "].x[0] = " << pArrayAll[i].x[0];
        }
    }*/

    delete [] pArray;
    if (s->myrank == 0) {
        delete[] pArrayReceiveLength;
        delete[] pArrayDisplacements;
    }

    return totalReceiveLength;
}

/*
 * NOTE: Remaining parts:
 * The remaining parts needed to complete the parallel program can be implemented in a straightforward way.
 * After the force computation, copies of particles from other processes have to be removed. The routine for the
 * time integration can be reused from the sequential case. It only processes all particles that belong to the process.
 * Particles are moved in two phases. First, the sequential routine is used to re-sort particles that have left their
 * cell in the local tree. Afterwards, particles that have left the process have to be sent to other processes
 * (implemented in `sendList()`).
 *
 */




