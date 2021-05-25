//
// Created by Michael Staneker on 15.03.21.
//

#include "../include/Tree.h"

/**
 * post-order traversal of the tree nodes and applying FUNCTION
 * @param t - initially called with root
void FUNCTION(TreeNode *t) {
    if (t != NULL) {
        for (int i=0; i<POWDIM; i++){
            FUNCTION(t->son[i]);
        }
        Perform the operations of the function FUNCTION on *t ;
    }
}

 * post-order traversal of the tree nodes and applying FUNCTION
 * generating key of the node in the recursion
 * @param t - initially called with root
 * @param k - current key (Lebesgue key, z-ordering)
 * @param level - current level in the tree for key generation
void FUNCTION(TreeNode *t, keytype k, int level) {
    if (t != NULL) {
        for (int i=0; i<POWDIM; i++){
            FUNCTION(t->son[i],
                     (keytype)(k | ((keytype)i << (DIM*(maxlevel-level-1)))), level+1);
        }
        Perform the operations of the function FUNCTION on *t ;
    }
}

**/


NodeList::NodeList() {
    node = particle;
    next = NULL;
}

KeyList::KeyList(){
    k = 0UL;
    next = NULL;
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

const char* getNodeType(int nodetype)
{
    switch (nodetype)
    {
        case 0: return "particle      ";
        case 1: return "pseudoParticle";
        case 2: return "domainList    ";
        default: return "not valid     ";
    }
}

const unsigned char DirTable[12][8] =
        { { 8,10, 3, 3, 4, 5, 4, 5}, { 2, 2,11, 9, 4, 5, 4, 5},
          { 7, 6, 7, 6, 8,10, 1, 1}, { 7, 6, 7, 6, 0, 0,11, 9},
          { 0, 8, 1,11, 6, 8, 6,11}, {10, 0, 9, 1,10, 7, 9, 7},
          {10, 4, 9, 4,10, 2, 9, 3}, { 5, 8, 5,11, 2, 8, 3,11},
          { 4, 9, 0, 0, 7, 9, 2, 2}, { 1, 1, 8, 5, 3, 3, 8, 6},
          {11, 5, 0, 0,11, 6, 2, 2}, { 1, 1, 4,10, 3, 3, 7,10} };
const unsigned char HilbertTable[12][8] = { {0,7,3,4,1,6,2,5}, {4,3,7,0,5,2,6,1}, {6,1,5,2,7,0,4,3},
                                            {2,5,1,6,3,4,0,7}, {0,1,7,6,3,2,4,5}, {6,7,1,0,5,4,2,3},
                                            {2,3,5,4,1,0,6,7}, {4,5,3,2,7,6,0,1}, {0,3,1,2,7,4,6,5},
                                            {2,1,3,0,5,6,4,7}, {4,7,5,6,3,0,2,1}, {6,5,7,4,1,2,0,3}};

keytype Lebesgue2Hilbert(keytype lebesgue, int level) {
    keytype hilbert = 0UL; // 0UL is our root, placeholder bit omitted
    //int level = 0, dir = 0;
    int dir = 0;
    //for (keytype tmp=lebesgue; tmp>0UL; tmp>>=DIM, level++); // obtain of key
    //if (level != 21) {
    //    Logger(DEBUG) << "Lebesgue2Hilbert: level = " << level << ", key" << lebesgue;
    //}
    //Logger(DEBUG) << "Lebesgue2Hilbert(): lebesgue = " << lebesgue << ", level = " << level;
    for (int lvl=maxlevel; lvl>0; lvl--) {
        //int cell = lebesgue >> ((level-1)*DIM) & (keytype)((1<<DIM)-1);
        int cell = (lebesgue >> ((lvl-1)*DIM)) & (keytype)((1<<DIM)-1);
        hilbert = hilbert<<DIM;
        if (lvl>maxlevel-level) {
            //Logger(DEBUG) << "Lebesgue2Hilbert(): cell = " << cell << ", dir = " << dir;
            hilbert += HilbertTable[dir][cell];
        }
        dir = DirTable[dir][cell];
    }
    //Logger(DEBUG) << "Lebesgue2Hilbert(): hilbert  = " << hilbert;
    //Logger(DEBUG) << "==============================";
    return hilbert;
}

long countParticles(TreeNode *t, long count){
     if (t != NULL){
         if (isLeaf(t) && t->node != domainList){
             return ++count;
         } else if (!isLeaf(t)) {
             for (int i = 0; i < POWDIM; i++) {
                 count = countParticles(t->son[i], count);
             }
         }
     }
     return count;
}

long countNodes(TreeNode *t, long count){
    if (t != NULL){
        if (!isLeaf(t)) {
            for (int i = 0; i < POWDIM; i++) {
                count = countNodes(t->son[i], count);
            }
        }
        return ++count;
    }
    return count;
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


void getParticleKeys(TreeNode *t, keytype *p, int &pCounter, keytype k, int level){
    if (t != NULL){
        for (int i = 0; i < POWDIM; i++) {
            if (isLeaf(t->son[i])){
                //Logger(DEBUG) << "Inserting son #" << i;
                p[pCounter] = Lebesgue2Hilbert(k | ((keytype)i << (DIM*(maxlevel-level-1))), level+1);
                        //& (KEY_MAX << DIM*(maxlevel-level-1)); // inserting key
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
    keytype *pKeys = new keytype[N]; //TODO: N+1 instead of N due to valgrind but why???
    for (int i=0; i<N; i++) {
        pKeys[i] = 0UL;
    }

    int pIndex{ 0 };
    getParticleKeys(root, pKeys, pIndex);
    // sort keys in ascending order
    std::sort(pKeys, pKeys+N);

    s->range[0] = 0UL; // range_0 = 0

    const int ppr = (N % s->numprocs != 0) ? N/s->numprocs+1 : N/s->numprocs; // particles per range

    //for (int i=0; i<N; i++){
    //    Logger(DEBUG) << pKeys[i];
    //}

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

//TODO: Code fragment 8.4: Determining current and new load distribution
void newLoadDistribution(TreeNode *root, SubDomainKeyTree *s){

    long c = countParticles(root); // count of particles in current process
    long *oldcount = new long[s->numprocs];
    // send current particles in each process to all other processes
    MPI_Allgather(&c, 1, MPI_LONG, oldcount, 1, MPI_LONG, MPI_COMM_WORLD);

    //for (int i=0; i<s->numprocs; i++){
    //    Logger(INFO) << "Load balancing: oldcount[" << i << "] = " << oldcount[i];
    //}

    long olddist[s->numprocs+1], newdist[s->numprocs+1]; // needed arrays for old and new particle distribution

    olddist[0] = 0;
    for (int i=0; i<s->numprocs; i++) {
        olddist[i + 1] = olddist[i] + oldcount[i];
    }

    for (int i=0; i<=s->numprocs; i++) {
        newdist[i] = (i * olddist[s->numprocs]) / s->numprocs;
    }

    //for (int i=0; i<=s->numprocs; i++){
    //    Logger(INFO) << "Load balancing: olddist[" << i << "] = " << olddist[i];
    //    Logger(INFO) << "Load balancing: newdist[" << i << "] = " << newdist[i];
    //}

    //for (int i=0; i<=s->numprocs; i++){
    //    Logger(DEBUG) << "Load balancing: OLD range[" << i << "] = " << s->range[i];
    //}

    for (int i=0; i<=s->numprocs; i++){
        s->range[i] = 0; // reset ranges on all processes to zero
    }

    int p = 0;
    long n = olddist[s->myrank];

    while (n > newdist[p]) {
        p++;
    }

    updateRange(root, n, p, s->range, newdist);

    s->range[0] = 0;
    s->range[s->numprocs] = KEY_MAX;

    keytype sendRange[s->numprocs+1];
    std::copy(s->range, s->range+s->numprocs+1, sendRange);

    // update new ranges on all processors
    //Logger(DEBUG) << "MPI_Allreduce() in newLoadDistribution() on ranges.";
    MPI_Allreduce(sendRange, s->range, s->numprocs+1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);

    //for (int i=0; i<=s->numprocs; i++){
    //    Logger(DEBUG) << "Load balancing: NEW range[" << i << "] = " << s->range[i];
    //}

    delete [] oldcount;
}

//TODO: To apply the space-filling Hilbert curve in our program,
// we have to insert the transformation to the Hilbert key at any place
// where the number of the process is computed from the domain key of
// a cell by key2proc.
// In addition, in the load balancing, we have to traverse all particles in order of
// increasing Hilbert keys. Here, it is no longer enough to just use a post-order
// tree traversal of all leaf nodes of the tree. Instead, we have to test
// in the recursion which son node possesses the lowest Hilbert key.
// For this, we compute the Hilbert keys of all (four or eight) son nodes, sort them,
// and then descend in the tree in the correct order.

void updateRange(TreeNode *t, long &n, int &p, keytype *range, long *newdist, keytype k, int level) {
    if(t != NULL) {
        //called recursively as in Algorithm 8.1;
        // the key of *t can be computed step by step in the recursion
        std::map<keytype, int> keymap;
        for (int i = 0; i < POWDIM; i++) {
            // sorting implicitly in ascending order
            keytype hilbert = Lebesgue2Hilbert(k | ((keytype)i << (DIM*(maxlevel-level-1))), level+1);
            keymap[hilbert] = i;
        }
        // actual recursion in correct order
        for (std::map<keytype, int>::iterator kit = keymap.begin(); kit != keymap.end(); kit++) {
            //Logger(DEBUG) << "updateRange(): hilbert  = "  << kit->first << ", i = " << kit->second << ", level = " << level;
            //Logger(DEBUG) << "updateRange(): next lebesgue = " << (k | ((keytype)kit->second << (DIM*(maxlevel-level-1))));
            updateRange(t->son[kit->second], n, p, range, newdist,
                        k | ((keytype)kit->second << (DIM*(maxlevel-level-1))), level+1);
        }
        // start of the operation on *t
        if (isLeaf(t) && t->node != domainList) {
            while (n >= newdist[p]) {
                //Logger(DEBUG) << "updateRange(): Found lebesgue = " << k << ", level = " << level;
                range[p] = Lebesgue2Hilbert(k , level); // | (KEY_MAX >> (level*DIM+1));
                //Logger(DEBUG) << "updateRange():       hilbert  = " << Lebesgue2Hilbert(k, level);
                //Logger(DEBUG) << "updateRange():       range[" << p << "] = " << range[p];
                p++;
            }
            n++;
        }
        // end of the operation on *t
    }
}

int key2proc(keytype k, SubDomainKeyTree *s) {
    for (int i=0; i<s->numprocs; i++) { //1
        if (k >= s->range[i] && k < s->range[i+1]) {
            //std::cout << "key2proc: " << i << std::endl;
            return i;
        }
    }
    Logger(ERROR) << "key2proc(k= " << k << "): -1!";
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

/* UNUSED
keytype maxHilbertSon(TreeNode *t, int level, keytype k){
    std::set<keytype> sonHilbert;
    keytype hilbert;
    for (int i = 0; i < POWDIM; i++) {
        hilbert = Lebesgue2Hilbert(k | ((keytype) i << (DIM * (maxlevel - level - 1))), level + 1);
        if (isLeaf(t->son[i]) && t->son[i]->node == particle) {
            // sorting implicitly in ascending order
            sonHilbert.insert(hilbert);
        }
    }
    return sonHilbert.empty() ? hilbert : *sonHilbert.rbegin(); // reverse iterator points to last element, i.e. largest key
}
*/

// initial call: createDomainList(root, 0, 0, s)
void createDomainList(TreeNode *t, int level, keytype k, SubDomainKeyTree *s) {
    t->node = domainList;
    keytype hilbert = Lebesgue2Hilbert(k, level); // & (KEY_MAX << DIM*(maxlevel-level));
    int p1 = key2proc(hilbert, s);
    //int p2 = key2proc(k | ~(~0L << DIM*(maxlevel-level)), s);
    //int p2 = key2proc((k | ((keytype)iMaxSon << (DIM*(maxlevel-level-1)))), s);
    //Logger(DEBUG) << "p1       k = " << k << ", level = " << level;
    //Logger(DEBUG) << "p1 hilbert = " << hilbert << ", proc = " << p1;
    //int p2 = key2proc(maxHilbertSon(t, level, k) | (KEY_MAX >> (DIM*(level+1)+1)), s);
    int p2 = key2proc(hilbert | (KEY_MAX >> (DIM*level+1)), s); // always shift the root placeholder bit to 0
    //Logger(DEBUG) << "p2 hilbert = " << (hilbert | (KEY_MAX >> (DIM*level+1))) << ", proc = " << p2;
    //Logger(DEBUG) << "=========================";
    if (p1 != p2) {
        for (int i = 0; i < POWDIM; i++) {
            if (t->son[i] == NULL) {
                t->son[i] = (TreeNode *) calloc(1, sizeof(TreeNode));
                //setSonBoxByIndex(&t->box, &t->son[i]->box, i); // somehow set in insert tree somewhere
            } else if (isLeaf(t->son[i]) && t->son[i]->node == particle){
                t->son[i]->node = domainList; // need to be set before inserting into the tree
                insertTree(&t->son[i]->p, t); // insert tree handles box
            }
            createDomainList(t->son[i], level + 1,  (keytype)(k | ((keytype)i << (DIM*(maxlevel-level-1)))), s);
        }
    }
}
/* UNUSED
void setSonBoxByIndex(Box *box, Box *sonbox, int i){
    int sonindex = i;
    for (int d = DIM - 1; d >= 0; d--) {
        if (i < 2*d || i == 0){
            sonbox->lower[d] = box->lower[d];
            sonbox->upper[d] = .5 * (box->upper[d] + box->lower[d]);
        } else {
            sonbox->lower[d] = .5 * (box->upper[d] + box->lower[d]);
            sonbox->upper[d] = box->upper[d];
            i -= 2*d;
        }
    }
    Logger(DEBUG) << "setSonByIndex(i = " << sonindex << "): box->lower = ["
                  << box->lower[0] << ", " << box->lower[1] << ", " << box->lower[2] << "]";
    Logger(DEBUG) << "                      box->upper = ["
                  << box->upper[0] << ", " << box->upper[1] << ", " << box->upper[2] << "]";
    Logger(DEBUG) << "setSonByIndex(i = " << sonindex << "): sonbox->lower = ["
                        << sonbox->lower[0] << ", " << sonbox->lower[1] << ", " << sonbox->lower[2] << "]";
    Logger(DEBUG) << "                      sonbox->upper = ["
                        << sonbox->upper[0] << ", " << sonbox->upper[1] << ", " << sonbox->upper[2] << "]";
}
*/

void clearDomainList(TreeNode *t){
    if (t != NULL){
        if (t->node == domainList){
            t->node = pseudoParticle; // former domain list node becomes pseudoParticle
            for (int i=0; i<POWDIM; i++) {
                clearDomainList(t->son[i]);
                if(isLeaf(t->son[i]) && t->son[i]->node == pseudoParticle) {
                    free(t->son[i]);
                    t->son[i] = NULL; // deleting empty domain list nodes directly
                }
            }
        }
        // no domainList node can exist below another type of particle
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
        t->son[b]->box = sonbox;
        insertTree(p, t->son[b]);
    } else {
        if (t->son[b] == NULL) {
            if (isLeaf(t) && t->node != domainList) {
                Particle p2 = t->p;
                t->node = pseudoParticle;
                t->son[b] = (TreeNode *) calloc(1, sizeof(TreeNode));
                t->son[b]->p = *p;
                t->p.todelete = false;
                t->son[b]->box = sonbox;
                insertTree(&p2, t);
            } else {
                t->son[b] = (TreeNode *) calloc(1, sizeof(TreeNode));
                t->son[b]->p = *p;
                t->son[b]->box = sonbox;
                // t->son[b]->node = particle;
            }
        } else {
            t->son[b]->box = sonbox;
            insertTree(p, t->son[b]);
            //insertTree(p, t);
        }
    }
}

// used to insert particles from other processes which should already be fine
// don't use when you're not 100% sure you can use it
// order must be ensured that parents are inserted before childs
void insertTreeFromOtherProc(Particle *p, TreeNode *t){
    Box sonbox;
    int b = sonNumber(&t->box, &sonbox, p);
    if (t->son[b] != NULL) {
        t->son[b]->box = sonbox;
        insertTreeFromOtherProc(p, t->son[b]);
    } else {
        t->son[b] = (TreeNode *) calloc(1, sizeof(TreeNode));
        t->son[b]->p = *p;
        t->son[b]->box = sonbox;
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
        if (isLeaf(t)
            && key2proc(Lebesgue2Hilbert(k, level), s) == s->myrank
            && t->node != domainList) {
            for (int d = 0; d < DIM; d++) {
                t->p.F[d] = 0;
            }
            //Logger(DEBUG) << "compF_BH(): Calculation force";
            forceTree(t, root, diam);
        }
    }
}

void forceTree(TreeNode *tl, TreeNode *t, float diam) {
    if (t != NULL) {
        float r = 0;
        for (int d=0; d<DIM; d++) {
            r += (t->p.x[d] - tl->p.x[d])*(t->p.x[d] - tl->p.x[d]);
        }
        r = sqrt(r);
        if (((isLeaf(tl)) || (diam < theta * r)) && tl->node != domainList) {
            // check if radius is zero, which means force should not be computed
            if (r == 0){
                Logger(WARN) << "Zero radius has been encoutered.";
            } else {
                force(&tl->p, &t->p);
            }
        } else {
            for (int i = 0; i < POWDIM; i++) {
                forceTree(t, t->son[i], .5 * diam);
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
        if (isLeaf(t) && t->node != domainList) {
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
        if (isLeaf(t) && t->node != domainList && (!t->p.moved)) {
            t->p.moved = true;
            if (!particleWithinBox(t->p, t->box)) {
                if (!particleWithinBox(t->p, root->box)){
                    Logger(INFO) << "Particle " << p2str(t->p) << " left system.";
                } else {
                    insertTree(&t->p, root);
                }
                //if (t->node == pseudoParticle){
                //    Logger(ERROR) << "Flagging pseudo particle for deletion in moveLeaf(). - Not good.";
                //}
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
        if (!isLeaf(t)) {
            int numberofsons = 0;
            int d;
            for (int i = 0; i < POWDIM; i++) {
                if (t->son[i] != NULL && t->son[i]->node != domainList) {
                    if (t->son[i]->p.todelete) {
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
                } else if (numberofsons == 1) {
                    // *t adopts the role of its only son node and
                    // the son node is deleted directly
                    if (t->son[d]->node != domainList) {
                        // son is an only son
                        t->p = t->son[d]->p;
                        t->node = t->son[d]->node;

                        // TODO: check why only free leaves prevents loosing particles
                        if (isLeaf(t->son[d])) {
                            free(t->son[d]);
                            t->son[d] = NULL;
                        }
                        //free(&t->son[d]->p);
                    }
                }
            }
        }
    }
}

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
 * NOTE: Remaining parts:
 * The remaining parts needed to complete the parallel program can be implemented in a straightforward way.
 * After the force computation, copies of particles from other processes have to be removed. The routine for the
 * time integration can be reused from the sequential case. It only processes all particles that belong to the process.
 * Particles are moved in two phases. First, the sequential routine is used to re-sort particles that have left their
 * cell in the local tree. Afterwards, particles that have left the process have to be sent to other processes
 * (implemented in `sendList()`).
 *
 */
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

    buildSendList(root, s, plist, pIndex, 0UL, 0); //TODO: something to be changed?

    for (int proc = 0; proc < s->numprocs; proc++) {
        ParticleList *current = &plist[proc]; // needed not to 'consume' plist
        for (int i=0; i<pIndex[proc]; i++){
            //Logger(DEBUG) << "x2send = (" << current->p.x[0] << ", " << current->p.x[1] << ", " << current->p.x[2] << ")";
            current = current->next;
        }
    }

    repairTree(root); // here, domainList nodes may not be deleted //TODO: something to be changed?

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

    delete [] plist;
    delete [] plistLengthSend;
    delete [] plistLengthReceive;
    for (int proc=0; proc < s->numprocs; proc++) {
        delete [] pArray[proc];
    }
    delete [] pArray;
}

//TODO: implement buildSendList (Sending Particles to Their Owners and Inserting Them in the Local Tree)
void buildSendList(TreeNode *t, SubDomainKeyTree *s, ParticleList *plist, int *pIndex, keytype k, int level) {
    ParticleList * current;
    if (t != NULL) {
        // start of the operation on *t
        int proc;
        if ((isLeaf(t))
            && ((proc = key2proc(Lebesgue2Hilbert(k, level), s)) != s->myrank)
            && t->node != domainList) {
            current = &plist[proc];
            for (int i=0; i<pIndex[proc]; i++) {
                current = current->next;
            }
            current->p = t->p;
            current->next = new ParticleList;
            t->p.todelete = true;
            pIndex[proc]++;
        }
        for (int i=0; i<POWDIM; i++) {
            buildSendList(t->son[i], s, plist, pIndex,
                          (keytype)(k | ((keytype)i << (DIM*(maxlevel-level-1)))), level+1);
        }
        // end of the operation on *t }
    }
}

void outputTree(TreeNode *root, bool detailed, bool onlyParticles) {

    int counterParticle = 0;
    int counterPseudoParticle = 0;
    int counterDomainList = 0;

    int nNodes = countNodes(root);
    Particle * pArray;
    nodetype * nArray;
    pArray = new Particle[nNodes];
    nArray = new nodetype[nNodes];
    getTreeArray(root, pArray, nArray);

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
                    Logger(INFO) << "\tnodetype: " << getNodeType(nArray[i]) << " ["
                                 << (pArray[i].todelete ? "true " : "false")
                                 << ", " << (pArray[i].moved ? "true " : "false")
                                 << "]  x = (" << pArray[i].x[0] << ", "
                                 << pArray[i].x[1] << ", "
                                 << pArray[i].x[2] << ")"
                                << "    v = (" << pArray[i].v[0] << ", "
                                << pArray[i].v[1] << ", "
                                << pArray[i].v[2] << ")"
                                << "    F = (" << pArray[i].F[0] << ", "
                                << pArray[i].F[1] << ", "
                                << pArray[i].F[2] << ")"
                                << "    m = " << pArray[i].m;
                }
            }
            else {
                Logger(INFO) << "\tnodetype: " << getNodeType(nArray[i]) << " ["
                             << (pArray[i].todelete ? "true " : "false")
                             << ", " << (pArray[i].moved ? "true " : "false")
                             << "]  x = (" << pArray[i].x[0] << ", "
                             << pArray[i].x[1] << ", "
                             << pArray[i].x[2] << ")"
                             << "    v = (" << pArray[i].v[0] << ", "
                             << pArray[i].v[1] << ", "
                             << pArray[i].v[2] << ")"
                             << "    F = (" << pArray[i].F[0] << ", "
                             << pArray[i].F[1] << ", "
                             << pArray[i].F[2] << ")"
                             << "    m = " << pArray[i].m;
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

void outputTree(TreeNode *root, std::string file, bool detailed, bool onlyParticles) {

    std::ofstream outf { file };

    if (!outf) {
        Logger(ERROR) << "An error occurred while opening '" << file << "'";
    } else {
        int counterParticle = 0;
        int counterPseudoParticle = 0;
        int counterDomainList = 0;

        int nNodes = countNodes(root);
        Particle * pArray;
        nodetype * nArray;
        keytype * kArray;
        pArray = new Particle[nNodes];
        nArray = new nodetype[nNodes];
        kArray = new keytype[nNodes];
        getTreeArray(root, pArray, nArray, kArray);

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
                        outf << "\tnodetype: " << getNodeType(nArray[i]) << " ["
                             << (pArray[i].todelete ? "true " : "false")
                             << ", " << (pArray[i].moved ? "true " : "false")
                             << "]  x = (" << pArray[i].x[0] << ", "
                             << pArray[i].x[1] << ", "
                             << pArray[i].x[2] << ")"
                             << "    v = (" << pArray[i].v[0] << ", "
                             << pArray[i].v[1] << ", "
                             << pArray[i].v[2] << ")"
                             << "    F = (" << pArray[i].F[0] << ", "
                             << pArray[i].F[1] << ", "
                             << pArray[i].F[2] << ")"
                             << "    m = " << pArray[i].m << ", ";
                        // converting key format
                        int levels [maxlevel];
                        for (int i = 0; i<maxlevel; i++) {
                            levels[i] = (kArray[i] >> 3*i) & (unsigned long)7;
                        }
                        std::string keyStr = "#|";
                        for (int i = maxlevel-1; i>=0; i--) {
                            keyStr += std::to_string(levels[i]);
                            keyStr += "|";
                        }
                        outf << keyStr << '\n';
                    }
                }
                else {
                    outf << "\tnodetype: " << getNodeType(nArray[i]) << " ["
                         << (pArray[i].todelete ? "true " : "false")
                         << ", " << (pArray[i].moved ? "true " : "false") << "], ";
                    // converting key format
                    int levels [maxlevel];
                    for (int lvl = 0; lvl<maxlevel; lvl++) {
                        levels[lvl] = (kArray[i] >> 3*lvl) & (unsigned long)7;
                    }
                    std::string keyStr = "#|";
                    for (int lvl = maxlevel-1; lvl>=0; lvl--) {
                        keyStr += std::to_string(levels[lvl]);
                        keyStr += "|";
                    }
                    outf << keyStr
                         << "  x = (" << pArray[i].x[0] << ", "
                         << pArray[i].x[1] << ", "
                         << pArray[i].x[2] << ")\n";
                }
            }
        }

        delete [] nArray;
        delete [] pArray;
        delete [] kArray;
    }
}

/* UNUSED
void particles2file(TreeNode *root, std::string file, SubDomainKeyTree *s){
    std::ofstream outf { file };

    if (!outf) {
        Logger(ERROR) << "An error occurred while opening '" << file << "'";
    } else {
        // writing csv header
        outf << "x" << ";"
             << "y" << ";"
             << "z" << ";"
             << "process" << ";"
             << "key" << "\n";

        int nNodes = countNodes(root);
        Particle * pArray;
        nodetype * nArray;
        keytype * kArray;
        pArray = new Particle[nNodes];
        nArray = new nodetype[nNodes];
        kArray = new keytype[nNodes];
        getTreeArray(root, pArray, nArray, kArray);

        for (int i=0; i<nNodes; i++) {
            if (nArray[i] == particle){
                outf << pArray[i].x[0] << ";"
                    << pArray[i].x[1] << ";"
                    << pArray[i].x[2] << ";"
                    << key2proc(kArray[i], s) << ";"
                    << std::bitset<64>(kArray[i]) << "\n";
            }
        }

        delete [] nArray;
        delete [] pArray;
        delete [] kArray;
    }
}
*/

void particles2file(TreeNode *root,
                    HighFive::DataSet *pos, HighFive::DataSet *vel, HighFive::DataSet *key, SubDomainKeyTree *s){
    int nNodes = countNodes(root);
    Particle * pArray;
    nodetype * nArray;
    keytype * kArray;
    pArray = new Particle[nNodes];
    nArray = new nodetype[nNodes];
    kArray = new keytype[nNodes];
    getTreeArray(root, pArray, nArray, kArray);

    std::vector<std::vector<double>> x, v; // two dimensional vector for 3D vector data
    std::vector<unsigned long> k; // one dimensional vector holding particle keys

    int nParticles = 0;

    for (int i=0; i<nNodes; i++) {
        if (nArray[i] == particle){
            x.push_back({ pArray[i].x[0], pArray[i].x[1], pArray[i].x[2] });
            v.push_back({ pArray[i].v[0], pArray[i].v[1], pArray[i].v[2] });
            k.push_back(kArray[i]);
            ++nParticles;
        }
    }

    // receive buffer
    int procN[s->numprocs];

    // send buffer
    int sendProcN[s->numprocs];
    for (int proc=0; proc<s->numprocs; proc++){
        sendProcN[proc] = s->myrank == proc ? nParticles : 0;
    }

    MPI_Allreduce(sendProcN, procN, s->numprocs, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    std::size_t nOffset = 0;
    // count total particles on other processes
    for (int proc = 0; proc < s->myrank; proc++){
        nOffset += procN[proc];
    }
    Logger(DEBUG) << "Offset to write to datasets: " << std::to_string(nOffset);

    // write to asscoiated datasets in h5 file
    // only working when load balancing has been completed and even number of particles
    pos->select({nOffset, 0},
                {std::size_t(nParticles), std::size_t(DIM)}).write(x);
    vel->select({nOffset, 0},
                {std::size_t(nParticles), std::size_t(DIM)}).write(v);
    key->select({nOffset}, {std::size_t(nParticles)}).write(k);

    delete [] nArray;
    delete [] pArray;
    delete [] kArray;
}

/* UNUSED
void outputParticles(TreeNode *t) {
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++) {
            outputParticles(t->son[i]);

        }
        if (isLeaf(t) && t->node != domainList) {
            Logger(DEBUG) << "\tparticle x = (" << t->p.x[0] << ", " << t->p.x[1] << ", " << t->p.x[2] << ")";
        }
    }
}
 */

NodeList* buildTreeList(TreeNode *t, NodeList *nLst) {
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++) {
            nLst = buildTreeList(t->son[i], nLst);
        }
        nLst->p = t->p;
        nLst->node = t->node;
        nLst->next = new NodeList;
        return nLst->next;
    }
    return nLst;
}

int getTreeArray(TreeNode *root, Particle *&p, nodetype *&n) {
    NodeList * nLst;
    nLst = new NodeList;
    int nIndex = 0;

    buildTreeList(root, nLst);
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
    return nIndex;
}

KeyList* buildTreeList(TreeNode *t, KeyList *kLst, keytype k, int level) {
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++) {
            kLst = buildTreeList(t->son[i], kLst,
                                   (keytype)(k | ((keytype)i << (DIM*(maxlevel-level-1)))), level+1);
        }
        //Logger(DEBUG) << "buildTreeList(): lebesgue = " << k << ", level = " << level;
        //Logger(DEBUG) << "buildTreeList(): hilbert  = " << Lebesgue2Hilbert(k, level);
        kLst->k = Lebesgue2Hilbert(k, level); // & (KEY_MAX << DIM*(maxlevel-level));
        kLst->next = new KeyList;
        return kLst->next;
    }
    return kLst;
}

int getTreeArray(TreeNode *root, Particle *&p, nodetype *&n, keytype *&k) {
    NodeList * nLst;
    nLst = new NodeList;
    int nIndex = 0;

    buildTreeList(root, nLst);
    if (nLst->next) {
        while (nLst->next) {
            p[nIndex] = nLst->p;
            n[nIndex] = nLst->node;
            NodeList *old = nLst;
            nLst = nLst->next;
            delete old;
            ++nIndex;
        }
    }

    KeyList * kLst;
    kLst = new KeyList;
    int kIndex = 0;

    buildTreeList(root, kLst);
    if (kLst->next) {
        while (kLst->next) {
            k[kIndex] = kLst->k;
            //Logger(DEBUG) << "k[kIndex] = " << k[kIndex];
            KeyList *old = kLst;
            kLst = kLst->next;
            delete old;
            ++kIndex;
        }
    }

    if (kIndex != nIndex){
        Logger(ERROR) << "Missmatch between kIndex and nIndex in getTreeArray(). - Very Bad.";
    }

    return nIndex;
}

ParticleList* buildParticleList(TreeNode *t, ParticleList *pLst){
    if (t != NULL) {
        if (isLeaf(t) && t->node != domainList) {
            pLst->p = t->p;
            pLst->next = new ParticleList;
            return pLst->next;
        } else {
            for (int i = 0; i < POWDIM; i++) {
                pLst = buildParticleList(t->son[i], pLst);
            }
        }
    }
    return pLst;
}

int getParticleListLength(ParticleList *plist) {
    ParticleList * current = plist;
    int count = 0;
    while (current->next) {
        count++;
        current = current->next;
    }
    return count;
}

int getParticleArray(TreeNode *root, Particle *&p){
    ParticleList * pLst;
    pLst = new ParticleList;

    buildParticleList(root, pLst);

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

/* UNUSED
void getDomainListNodes(TreeNode *t, ParticleList *pList, int &pCounter) {
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
            getDomainListNodes(t->son[i], pList, pCounter);
        }
    }
}*/

/* UNUSED
int getDomainListArray(TreeNode *root, Particle *&pArray) {
    int pCounter = 0;
    ParticleList *pList;
    pList = new ParticleList;
    getDomainListNodes(root, pList, pCounter);
    pArray = new Particle[pCounter];
    for (int i=0; i<pCounter; i++) {
        pArray[i] = pList->p;
        ParticleList* old = pList;
        pList = pList->next;
        delete old;
    }
    return pCounter;
}*/


/* UNUSED
void getLowestDomainListNodes(TreeNode *t, ParticleList *pList, int &pCounter) {
    if (t != NULL){
        ParticleList * current;
        if (t->node == domainList && isLowestDomainListNode(t)) {
            current = pList;
            for (int j=0; j<pCounter; j++) {
                current = current->next;
            }
            current->p = t->p;
            current->next = new ParticleList;
            pCounter++;
        }
        for (int i = 0; i < POWDIM; i++) {
            getLowestDomainListNodes(t->son[i], pList, pCounter);
        }
    }
}
*/

/* UNUSED
int getLowestDomainListArray(TreeNode *root, Particle *&pArray) {
    int pCounter = 0;
    ParticleList *pList;
    pList = new ParticleList;

    getLowestDomainListNodes(root, pList, pCounter);
    pArray = new Particle[pCounter];

    for (int i=0; i<pCounter; i++) {
        pArray[i] = pList->p;
        ParticleList* old = pList;
        pList = pList->next;
        delete old;
    }
    return pCounter;
}
*/

void getLowestDomainListNodes(TreeNode *t, ParticleList *pList, KeyList *kList, int &pCounter, keytype k, int level) {
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
            kCurrent->k = k; //(unsigned long)(k << DIM*(maxlevel-level-1)); //k+i ?!
            kCurrent->next = new KeyList;
            pCounter++;
        }
        for (int i = 0; i < POWDIM; i++) {
            getLowestDomainListNodes(t->son[i], pList, kList, pCounter,
                                         (keytype)(k | ((keytype)i << (DIM*(maxlevel-level-1)))), level+1);
        }
    }
}

int getLowestDomainListArray(TreeNode *root, Particle *&pArray, keytype *&kArray) {
    int pCounter = 0;
    ParticleList *pList;
    pList = new ParticleList;
    KeyList *kList;
    kList = new KeyList;

    getLowestDomainListNodes(root, pList, kList, pCounter);
    pArray = new Particle[pCounter];
    kArray = new keytype[pCounter];

    for (int i=0; i<pCounter; i++) {
        pArray[i] = pList->p;
        kArray[i] = kList->k;
        ParticleList* pOld = pList;
        KeyList* kOld = kList;
        pList = pList->next;
        kList = kList->next;
        delete pOld;
        delete kOld;
    }
    return pCounter;
}


void zeroLowestDomainListNodes(TreeNode *t) {
    if (t != NULL){
        for (int i = 0; i < POWDIM; i++) {
            zeroLowestDomainListNodes(t->son[i]);
        }
        if (t->node == domainList && isLowestDomainListNode(t)) {
            t->p.x[0] = 0;
            t->p.x[1] = 0;
            t->p.x[2] = 0;
            t->p.m = 0;
        }
    }
}

void zeroDomainListNodes(TreeNode *t) {
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++) {
            zeroDomainListNodes(t->son[i]);
        }
        if (t->node == domainList) {
            t->p.x[0] = 0;
            t->p.x[1] = 0;
            t->p.x[2] = 0;
            t->p.m = 0;
        }
    }
}

void updateLowestDomainListNodesMomentsMasses(TreeNode *t, int &pCounter, float * masses, float * moments) {
    if (t != NULL){
        if (t->node == domainList && isLowestDomainListNode(t)) {

            t->p.x[0] = moments[pCounter * 3];
            t->p.x[1] = moments[pCounter * 3 + 1];
            t->p.x[2] = moments[pCounter * 3 + 2];
            t->p.m = masses[pCounter];

            pCounter++;
        }
        for (int i = 0; i < POWDIM; i++) {
            updateLowestDomainListNodesMomentsMasses(t->son[i], pCounter, masses, moments);
        }
    }
}

void updateLowestDomainListNodesCom(TreeNode *t) {
    if (t != NULL){
        if (t->node == domainList && isLowestDomainListNode(t)) {

            if (t->p.m > 0) {
                t->p.x[0] = t->p.x[0] / t->p.m;
                t->p.x[1] = t->p.x[1] / t->p.m;
                t->p.x[2] = t->p.x[2] / t->p.m;
            }
        }
        for (int i = 0; i < POWDIM; i++) {
            updateLowestDomainListNodesCom(t->son[i]);
        }
    }
}

void updateLowestDomainListNodes(TreeNode *t, int &pCounter, float * masses, float * moments) {
    zeroLowestDomainListNodes(t);
    updateLowestDomainListNodesMomentsMasses(t, pCounter, masses, moments);
    updateLowestDomainListNodesCom(t);
}

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

//TODO: implement compPseudoParticlespar (Parallel Computation of the Values of the Pseudoparticles)
void compPseudoParticlesPar(TreeNode *root, SubDomainKeyTree *s) {

    zeroDomainListNodes(root);

    compLocalPseudoParticlesPar(root);

    Particle * pArray;
    keytype  * kArray;
    int pLength = getLowestDomainListArray(root, pArray, kArray);

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

    //Logger(DEBUG) << "MPI_Allreduce() in compPseudoParticlesPar() on moments.";
    MPI_Allreduce(&moments, &global_moments, 3*pLength, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    //Logger(DEBUG) << "MPI_Allreduce() in compPseudoParticlesPar() on masses.";
    MPI_Allreduce(&masses, &global_masses, pLength, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    //for (int i=0; i<pLength; i++) {
    //    Logger(INFO) << "masses[" << i << "] = " << masses[i];
    //    Logger(INFO) << "global_masses[" << i << "] = " << global_masses[i];
    //}

    int dCounter = 0;

    updateLowestDomainListNodes(root, dCounter, global_masses, global_moments);

    compDomainListPseudoParticlesPar(root);

    delete [] pArray;
    delete [] kArray;

}

//TODO: implement compLocalPseudoParticlespar (Parallel Computation of the Values of the Pseudoparticles)
void compLocalPseudoParticlesPar(TreeNode *t) {
    //called recursively as in Algorithm 8.1;
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++) {
            compLocalPseudoParticlesPar(t->son[i]);
        }
        // start of the operation on *t
        if (!isLeaf(t) && (t->node != domainList || isLowestDomainListNode(t))) {
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
void compDomainListPseudoParticlesPar(TreeNode *t) {
    //called recursively as in Algorithm 8.1 for the coarse domainList-tree;
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++) {
            compDomainListPseudoParticlesPar(t->son[i]);
        }
        // start of the operation on *t
        if (t->node == domainList && !isLowestDomainListNode(t)) {
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
            /*if (t->node == domainList) {
                Logger(INFO) << "!lowest domain mass: " << t->p.m << "(" << t->son[0]->p.m << ", "
                                                        << t->son[1]->p.m << ", " << t->son[2]->p.m << ", "
                                                        << t->son[3]->p.m << ", " << t->son[4]->p.m << ", "
                                                        << t->son[5]->p.m << ", " << t->son[6]->p.m << ", "
                                                        << t->son[7]->p.m << ")"; }*/
            for (int d = 0; d < DIM; d++) {
                if (t->p.m != 0) {
                    t->p.x[d] = t->p.x[d] / t->p.m;
                }
            }
        }
        // end of the operation on *t
    }
}

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
    //if (dx+dy+dz == 0.f){
        //Logger(ERROR) << "Smallest distance is zero!";
    //}

    return sqrt(dx*dx + dy*dy + dz*dz);
}


// Using naive keys, as they are unique
// td is a domain list node, belonging to other process
// level unused, keep it for now for debugging purposes
void symbolicForce(TreeNode *td, TreeNode *t, float diam, ParticleMap &pmap, SubDomainKeyTree *s,
                   keytype k, int level) {

    //Logger(INFO) << "symbolicForce: " << std::bitset<64>(k) << " mapped to proc " << key2proc(k, s);
    if (t != NULL){ // && (key2proc(Lebesgue2Hilbert(k, level), s) == s->myrank || t->node == domainList) {

        // the key of *t can be computed step by step in the recursion insert t->p into list plist;
        if (t->node != domainList) { // don't send domainList keys
            /*ParticleMap::iterator pit = pmap.find(k);
            if (pit == pmap.end()){ // not found
                Logger(DEBUG) << "td: " << getNodeType(td->node) << ", x = " << "(" << td->p.x[0] << ", ...)";
                //Logger(DEBUG) << "toSend: " << k << ", " << getNodeType(t->node) << ", x = " << t->p.x[0]
                //                    << ", level = " << level;
            }*/
            pmap[k] = t->p; // will overwrite and therefore ensures uniqueness
        }

        float r = smallestDistance(td, t);
        /*Logger(DEBUG) << "symbolicForce(): td->box.lower = ["
                      << td->box.lower[0] << ", " << td->box.lower[1] << ", " << td->box.lower[2] << "]";
        Logger(DEBUG) << "                 td->box.upper = ["
                      << td->box.upper[0] << ", " << td->box.upper[1] << ", " << td->box.upper[2] << "]";*/
        if (diam >= theta * r) {
            for (int i = 0; i < POWDIM; i++) {
                // recursion with naive key
                symbolicForce(td, t->son[i], .5 * diam, pmap, s,
                              (keytype)((k << DIM) | (keytype)i), level+1); // !!! different key for uniqueness
            }
        }
    }
}

/*void symbolicForce(TreeNode *td, TreeNode *t, float diam, ParticleList &plist, SubDomainKeyTree *s,
                   keytype k, int level){

}*/

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

//compF_BHpar analog to sendParticles()
void compF_BHpar(TreeNode *root, float diam, SubDomainKeyTree *s) {

    double t1, t2; // timing variables

    ParticleList * plist;
    plist = new ParticleList[s->numprocs];
    ParticleList * uniquePlist;
    uniquePlist = new ParticleList[s->numprocs];
    ParticleMap * pmap;
    pmap = new ParticleMap[s->numprocs];

    /*int * pIndex;
    pIndex = new int[s->numprocs];
    for (int proc = 0; proc < s->numprocs; proc++) {
        pIndex[proc] = 0;
    }*/

    t1 = MPI_Wtime();
    compTheta(root, root, s, pmap, diam);
    t2 = MPI_Wtime();
    Logger(DEBUG) << "compF_BHpar(): +++++++++++++++++++++++++ compTheta(): " << t2-t1 << "s";

    t1 = MPI_Wtime();
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

            pArray[proc] = new Particle[plistLengthSend[proc]];
            ParticleList * current = &plist[proc];
            int counter = 0;
            // parents have lesser key than children
            for (ParticleMap::iterator pit = pmap[proc].begin(); pit != pmap[proc].end(); pit++) {
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
    // allocate missing (sub)array for process rank
    pArray[s->myrank] = new Particle[receiveLength];

    // building string to output sendLengths to one line
    std::string sendLengthStr = "[";
    for (int proc = 0; proc < s->numprocs; proc++){
        sendLengthStr += std::to_string(plistLengthSend[proc]);
        if (proc < s->numprocs - 1) {
            sendLengthStr += ", ";
        } else {
            sendLengthStr += "]";
        }
    }

    Logger(DEBUG) << "compF_BHpar(): [PER PROC] sendLength = " << sendLengthStr;
    Logger(DEBUG) << "compF_BHpar(): [TOTAL] receiveLength = " << receiveLength;
    t2 = MPI_Wtime();
    Logger(DEBUG) << "compF_BHpar(): +++++++++ preparing particle exchange: " << t2-t1 << "s";

    t1 = MPI_Wtime();
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
        //Logger(DEBUG) << "Inserting particle pArray[" << i << "] : x = "
        //                    << pArray[s->myrank][i].x[0] << ", m = " << pArray[s->myrank][i].m;
        pArray[s->myrank][i].todelete = true;
        insertTreeFromOtherProc(&pArray[s->myrank][i], root);
    }

    delete [] plist;
    delete [] plistLengthSend;
    delete [] plistLengthReceive;
    delete [] pmap;
    for (int proc=0; proc < s->numprocs; proc++) {
        delete [] pArray[proc];
    }
    delete [] pArray;
    t2 = MPI_Wtime();
    Logger(DEBUG) << "compF_BHpar(): +++++ sending and receiving particles: " << t2-t1 << "s";

    t1 = MPI_Wtime();
    compF_BH(root, root, diam, s);
    t2 = MPI_Wtime();
    Logger(DEBUG) << "compF_BHpar(): +++++ force calculation on local tree: " << t2-t1 << "s";
}

/**
 * This is the used compTheta function utilizing std::map
 * @param t
 * @param root
 * @param s
 * @param pmap
 * @param diam
 * @param k
 * @param level
 */
void compTheta(TreeNode *t, TreeNode *root, SubDomainKeyTree *s, ParticleMap *pmap, float diam, keytype k, int level) {
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++) {
            compTheta(t->son[i], root, s, pmap, diam, (keytype)(k | ((keytype)i << (DIM*(maxlevel-level-1)))), level+1);
        }
        // start of the operation on *t
        int proc = key2proc(Lebesgue2Hilbert(k, level), s);
        if (t->node == domainList && proc != s->myrank && level != 0) { // exclude root
            /*Logger(DEBUG) << "symbolicForce(): t->box.lower = ["
                          << t->box.lower[0] << ", " << t->box.lower[1] << ", " << t->box.lower[2] << "], proc = " << proc;
            Logger(DEBUG) << "                 t->box.upper = ["
                          << t->box.upper[0] << ", " << t->box.upper[1] << ", " << t->box.upper[2] << "]";*/
            symbolicForce(t, root, diam, pmap[proc], s, 1UL, 0); // !!! using different keys here to ensure uniqueness
        }
    } // end of the operation on *t
}

/* UNUSED
bool compareParticles(Particle p1, Particle p2) {
    //return (p1.x[0] == p2.x[0] && p1.x[1] == p2.x[1] && p1.x[2] == p2.x[2]);
    return (sqrt(p1.x[0]*p1.x[0] + p1.x[1]*p1.x[1] + p1.x[2]*p1.x[2]) -
            sqrt(p2.x[0]*p2.x[0] + p2.x[1]*p2.x[1] + p2.x[2]*p2.x[2]));
}
*/

int gatherParticles(TreeNode *root, SubDomainKeyTree *s, Particle *&pArrayAll) {
    Particle * pArray;
    int pLength = getParticleArray(root, pArray);

    int *pArrayReceiveLength;
    int *pArrayDisplacements;
    if (s->myrank == 0) {
        pArrayReceiveLength = new int[s->numprocs];
        pArrayDisplacements = new int[s->numprocs];
        pArrayDisplacements[0] = 0;
    }

    MPI_Gather(&pLength, 1, MPI_INT, pArrayReceiveLength, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int totalReceiveLength = 0;

    if (s->myrank == 0) {
        for (int proc=0; proc<s->numprocs; proc++) {
            //Logger(DEBUG) << "receiveLength[" << proc << "] = " << pArrayReceiveLength[proc];
            totalReceiveLength += pArrayReceiveLength[proc];
            if (pArrayReceiveLength[proc] == 0){
                Logger(ERROR) << "Process " << proc << " ran out of particles. - Not good...";
            }
        }
    }

    if (s->myrank == 0) {
        for (int proc=1; proc<s->numprocs; proc++) {
            pArrayDisplacements[proc] = pArrayReceiveLength[proc-1] + pArrayDisplacements[proc-1];
            //Logger(DEBUG) << "Displacements: " << pArrayDisplacements[proc];
        }
    }

    if (s->myrank == 0) {
        pArrayAll = new Particle[totalReceiveLength];
    }

    MPI_Gatherv(pArray, pLength, mpiParticle, pArrayAll, pArrayReceiveLength,
                pArrayDisplacements, mpiParticle, 0, MPI_COMM_WORLD);

    delete [] pArray;
    if (s->myrank == 0) {
        delete[] pArrayReceiveLength;
        delete[] pArrayDisplacements;
    }

    return totalReceiveLength;
}

int gatherParticles(TreeNode *root, SubDomainKeyTree *s, Particle *&pArrayAll, int *&processNumber) {
    Particle * pArray;
    int pLength = getParticleArray(root, pArray);

    int *pArrayReceiveLength;
    int *pArrayDisplacements;
    if (s->myrank == 0) {
        pArrayReceiveLength = new int[s->numprocs];
        pArrayDisplacements = new int[s->numprocs];
        pArrayDisplacements[0] = 0;
    }

    MPI_Gather(&pLength, 1, MPI_INT, pArrayReceiveLength, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int totalReceiveLength = 0;

    if (s->myrank == 0) {
        for (int proc=0; proc<s->numprocs; proc++) {
            //Logger(DEBUG) << "receiveLength[" << proc << "] = " << pArrayReceiveLength[proc];
            totalReceiveLength += pArrayReceiveLength[proc];
            if (pArrayReceiveLength[proc] == 0){
                Logger(ERROR) << "Process " << proc << " ran out of particles. - Not good.";
            }
        }
    }

    if (s->myrank == 0) {
        int displacement = 0;
        processNumber = new int[totalReceiveLength];
        for (int proc = 0; proc < s->numprocs; proc++) {
            for (int i = 0; i < pArrayReceiveLength[proc]; i++) {
                processNumber[displacement + i] = proc;
            }
            displacement += pArrayReceiveLength[proc];
        }
    }

    if (s->myrank == 0) {
        for (int proc=1; proc<s->numprocs; proc++) {
            pArrayDisplacements[proc] = pArrayReceiveLength[proc-1] + pArrayDisplacements[proc-1];
            //Logger(DEBUG) << "Displacements: " << pArrayDisplacements[proc];
        }
    }

    if (s->myrank == 0) {
        pArrayAll = new Particle[totalReceiveLength];
    }

    MPI_Gatherv(pArray, pLength, mpiParticle, pArrayAll, pArrayReceiveLength,
                pArrayDisplacements, mpiParticle, 0, MPI_COMM_WORLD);

    delete [] pArray;
    if (s->myrank == 0) {
        delete[] pArrayReceiveLength;
        delete[] pArrayDisplacements;
    }

    return totalReceiveLength;
}
