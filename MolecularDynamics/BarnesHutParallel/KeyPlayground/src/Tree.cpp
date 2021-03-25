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

keytype key(TreeNode t){
    //TODO: implement
    return 0;
}

/** First step in the path key creation done explicitly [p. 343f]
 * - Starting at the root node 1
 * - Each level needs 3 bits => [0,7] are the labels of the sons
 * - The labels [0,7] are shifted 3 x level times
 *
 * @param t Current node in recursion, should be initialized with root
 * @param p Container to be filled with path keys of all leaves (keytype[N])
 * @param pCounter Global counter by reference
 * @param k default=1UL (root node)
 * @param level default=0
 */
void getParticleKeysSimple(TreeNode *t, keytype *p, int &pCounter, keytype k, int level){
    if (t != NULL){
        for (int i = 0; i < POWDIM; i++) {
            if (isLeaf(t->son[i])){
                p[pCounter] = k + (static_cast<keytype>(i) << level*DIM); // inserting key
                Logger(DEBUG) << "Inserted particle '" << p[pCounter] << "'@" << pCounter;
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
                p[pCounter] = k + (i << DIM*(maxlevel-level-1)); // inserting key
                Logger(DEBUG) << "Inserted particle '" << std::bitset<64>(p[pCounter]) << "'@" << pCounter;
                ++pCounter; // counting inserted particles
            } else {
                getParticleKeys(t->son[i], p, pCounter,
                                k + (i << DIM*(maxlevel-level-1)), level+1); // go deeper
            }
        }
    }
}

void createRanges(TreeNode *root, int N, SubDomainKeyTree *s, int K){
    // K-domains for debugging
    //s->range = new keytype[s->numprocs+1];
    keytype *pKeys = new keytype[N];

    int pIndex{ 0 };
    getParticleKeys(root, pKeys, pIndex);
    // sort keys in ascending order
    std::sort(pKeys, pKeys+N);

    s->range = new keytype[K+1];

    s->range[0] = 0UL; // range_0 = 0

    //const int ppr = (N % s->numprocs != 0) ? N/s->numprocs+1 : N/s->numprocs; // particles per range
    const int ppr = (N % K != 0) ? N/K+1 : N/K; // particles per range, K procs emulated

    //for (int i=1; i<s->numprocs; i++){
    for (int i=1; i<K; i++){
        s->range[i] = pKeys[i*ppr];
        Logger(DEBUG) << "Computed range[" << i << "] = " << std::bitset<64>(s->range[i]);
    }
    s->range[K] = KEY_MAX; // last range does not need to be computed
}

int key2proc(keytype k, SubDomainKeyTree *s) {
    for (int i=0; i<s->numprocs; i++) { //1
        if (k >= s->range[i]) {
            return i;
        }
    }
    return -1; // error
}

// initial call: createDomainList(root, 0, 0, s)
void createDomainList(TreeNode *t, int level, keytype k, SubDomainKeyTree *s) {
    t->node = domainList;
    int p1 = key2proc(k,s);
    int p2 = key2proc(k | ~(~0L << DIM*(maxlevel-level)),s);
    if (p1 != p2) {
        for (int i = 0; i < POWDIM; i++) {
            t->son[i] = (TreeNode *) calloc(1, sizeof(TreeNode));
            createDomainList(t->son[i], level + 1,  k + (i << DIM*(maxlevel-level-1)), s);
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
    Box sunbox;
    int b = sonNumber(&t->box, &sunbox, p);
    if (t->son[b] == NULL) {
        if (isLeaf(t)) {
            Particle p2 = t->p;
            t->son[b] = (TreeNode*)calloc(1, sizeof(TreeNode));
            t->son[b]->p = *p;
            t->son[b]->box = sunbox;
            insertTree(&p2, t);
        } else {
            t->son[b] = (TreeNode*)calloc(1, sizeof(TreeNode));
            t->son[b]->p = *p;
            t->son[b]->box = sunbox;
        }
    } else {
        if (t->son[b]->node == domainList) {
            //here???
        }
        else {
            t->son[b]->box = sunbox; //?
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
        //return b;
    }
    return b;
}

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

// adapted to parallel implementation
void compF_BH(TreeNode *t, TreeNode *root, float diam, SubDomainKeyTree *s) {
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++) {
            compF_BH(t->son[i], root, diam, s);
        }
        // start of the operation on *t
        if (isLeaf(t) && (key2proc(key(*t), s) == s->myrank)) {
            for (int d = 0; d < DIM; d++) {
                t->p.F[d] = 0;
            }
            force_tree(t, root, diam);
        }
    }
}

void force_tree(TreeNode *tl, TreeNode *t, float diam) {
    if ((t != tl) && (t != NULL)) {
        float r = 0;
        for (int d=0; d<DIM; d++) {
            r += sqrt(abs(t->p.x[d] - tl->p.x[d]));
        }
        r = sqrt(r);
        if ((isLeaf(t)) || (diam < theta * r)) {
            force(&tl->p, &t->p);
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
        if (isLeaf(t)) {
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
        if ((isLeaf(t)) && (!t->p.moved)) {
            t->p.moved = true;
            if (particleWithinBox(t->p, t->box)) {
                insertTree(&t->p, root);
                t->p.todelete = true;
            }
        }
    }
}

void repairTree(TreeNode *t) {
    if (t != NULL && t->node != domainList) {
        if (!isLeaf(t)) {
            int numberofsons = 0;
            int d;
            for (int i = 0; i < POWDIM; i++) {
                if (t->son[i] != NULL) {
                    if (t->son[i]->p.todelete)
                        free(t->son[i]);
                    else {
                        numberofsons++;
                        d = i;
                    }
                }
            }
            if (0 == numberofsons) {
                // *t is an ‘‘empty’’ leaf node and can be deleted
                t->p.todelete = true;
            } else if (1 == numberofsons) {
                // *t adopts the role of its only son node and
                // the son node is deleted directly
                t->p = t->son[d]->p;
                //std::cout << "t->son[d]->p.x[0] = " << t->son[d]->p.x[0] << std::endl;
                free(&t->son[d]->p);
            }
        }
    }
}

void output_particles(TreeNode *t) {
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++) {
            output_particles(t->son[i]);
        }
        if (isLeaf(t)) {
            //if (!isnan(t->p.x[0]))
            std::cout << "\tx = (" << t->p.x[0] << ", " << t->p.x[1] << ", " << t->p.x[2] << ")" << std::endl;
        }
    }
}

ParticleList* build_particle_list(TreeNode *t, ParticleList *pLst){
    if (t != NULL) {
        if (isLeaf(t)) {
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

void get_particle_array(TreeNode *root, Particle *p){
    auto pLst = new ParticleList;
    build_particle_list(root, pLst);
    int pIndex { 0 };
    while(pLst->next){
        p[pIndex] = pLst->p;
        Logger(DEBUG) << "Adding to *p: x = (" << p[pIndex].x[0] << ", " << p[pIndex].x[1] << ", " << p[pIndex].x[2] << ")";
        pLst = pLst->next;
        ++pIndex;
    }
}

void freeTree_BH(TreeNode *root) {
    if (root != NULL) {
        for (int i=0; i<POWDIM; i++) {
            if (root->son[i] != NULL) {
                freeTree_BH(root->son[i]);
                free(root->son[i]);
            }
        }
    }
}

//TODO: implement sendParticles (Sending Particles to Their Owners and Inserting Them in the Local Tree)
// determine right amount of memory which has to be allocated for the `buffer`,
// by e.g. communicating the length of the message as prior message or by using other MPI commands
void sendParticles(TreeNode *root, SubDomainKeyTree *s) {
    //allocate memory for s->numprocs particle lists in plist;
    //initialize ParticleList plist[to] for all processes to;
    //TODO: buildSendlist(root, s, plist);
    repairTree(root); // here, domainList nodes may not be deleted
    for (int i=1; i<s->numprocs; i++) {
        int to = (s->myrank+i)%s->numprocs;
        int from = (s->myrank+s->numprocs-i)%s->numprocs;
        //send particle data from plist[to] to process to;
        // receive particle data from process from;
        //insert all received particles p into the tree using insertTree(&p, root);
    }
    //TODO: delete plist;
}

//TODO: implement buildSendlist (Sending Particles to Their Owners and Inserting Them in the Local Tree)
void buildSendlist(TreeNode *t, SubDomainKeyTree *s, ParticleList *plist) {
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++) {
            //TODO: buildSendlist(t->son[i]);
        }
        // start of the operation on *t
        if (t != NULL) {
            for (int i = 0; i < POWDIM; i++) {
                //TODO: buildSendlist(t->son[i]);
            }
            int proc;
            if ((isLeaf(t)) && ((proc = key2proc(key(*t), s)) != s->myrank)) {
                // the key of *t can be computed step by step in the recursion
                //insert t->p into list plist[proc];
                //mark t->p as to be deleted;
            }
        }
    }
    // end of the operation on *t
}

//TODO: implement compPseudoParticlespar (Parallel Computation of the Values of the Pseudoparticles)
void compPseudoParticlespar(TreeNode *root, SubDomainKeyTree *s) {
    compLocalPseudoParticlespar(root);
    //MPI_Allreduce(..., {mass, moments} of the lowest domainList nodes, MPI_SUM, ...);
    compDomainListPseudoParticlespar(root);
}

//TODO: implement compLocalPseudoParticlespar (Parallel Computation of the Values of the Pseudoparticles)
void compLocalPseudoParticlespar(TreeNode *t) {
    //called recursively as in Algorithm 8.1;
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++) {
            compLocalPseudoParticlespar(t->son[i]);
        }
        // start of the operation on *t
        if ((!isLeaf(t)) && (t->node != domainList)) {
            // operations analogous to Algorithm 8.5 (see below)
        }
    }
    // end of the operation on *t
}

//TODO: implement compDomainListPsudoParticlespar (Parallel Computation of the Values of the Pseudoparticles)
void compDomainListPseudoParticlespar(TreeNode *t) {
    //called recursively as in Algorithm 8.1 for the coarse domainList-tree;
    // start of the operation on *t
    if (t->node == domainList) {
        // operations analogous to Algorithm 8.5 (see below)
    }
    // end of the operation on *t
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

//TODO: implement symbolicForce (Determining Subtrees that are Needed in the Parallel Force Computation)
void symbolicForce(TreeNode *td, TreeNode *t, float diam, ParticleList *plist, SubDomainKeyTree *s) {
    if ((t != NULL) && (key2proc(key(*t), s) == s->myrank)) {
        // the key of *t can be computed step by step in the recursion insert t->p into list plist;
        float r = 0; //IMPLEMENT: smallest distance from t->p.x to cell td->box;
        if (diam >= theta * r) {
            for (int i = 0; i < POWDIM; i++) {
                symbolicForce(td, t->son[i], .5 * diam, plist, s);
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
void compF_BHpar(TreeNode *root, float diam, SubDomainKeyTree *s) {
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
}

//TODO: implement compTheta (Parallel Force Computation)
void compTheta(TreeNode *t, TreeNode *root, SubDomainKeyTree *s, ParticleList *plist, float diam) {
    //called recursively as in Algorithm 8.1;
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++)
            compTheta(t->son[i], root, s, plist, diam);
        // start of the operation on *t
        int proc;
        if ((true/* TODO: *t is a domainList node*/) && ((proc = key2proc(key(*t), s)) != s->myrank)) {
            // the key of *t can be computed step by step in the recursion
            symbolicForce(t, root, diam, &plist[proc], s);
        }
    }
// end of the operation on *t
}


/*
 * NOTE: Remaining parts:
 * The remaining parts needed to complete the parallel pro- gram can be implemented in a straightforward way.
 * After the force com- putation, copies of particles from other processes have to be removed. The routine for the
 * time integration can be reused from the sequential case. It only processes all particles that belong to the process.
 * Particles are moved in two phases. First, the sequential routine is used to re-sort particles that have left their
 * cell in the local tree. Afterwards, particles that have left the process have to be sent to other processes
 * (implemented in `sendList()`).
 *
 */




