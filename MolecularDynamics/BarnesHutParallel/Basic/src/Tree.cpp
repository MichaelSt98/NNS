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
int key2proc(keytype k, SubDomainKeyTree *s) {
    for (int i=1; i<=s->numprocs; i++) {
        if (k >= s->range[i]) {
            return i - 1;
        }
    }
    return -1; // error
}

void createDomainList(TreeNode *t, int level, keytype k, SubDomainKeyTree *s) {
    t->node = domainList;
    int p1 = key2proc(k,s);
    int p2 = key2proc(k | ~(~0L << DIM*(maxlevel-level)),s); if (p1 != p2)
        for (int i=0; i<POWDIM; i++) {
            t->son[i] = (TreeNode*)calloc(1, sizeof(TreeNode));
            createDomainList(t->son[i], level+1,k + i<<DIM*(maxlevel-level-1), s);
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

void insertTree(Particle *p, TreeNode *t) {
    // determine the son b of t in which particle p is located
    // compute the boundary data of the subdomain of the son node and store it in t->son[b].box;
    Box sunbox;
    int b = sonNumber(&t->box, &sunbox, p);
    //std::cout << "b = " << b << std::endl;
    //std::cout << "sunbox.upper[0] = " << sunbox.upper[0] << std::endl;
    //t->son[b]->box = *sunbox;
    /*for (int d = DIM - 1; d >= 0; d--) {
        t->son[b]->box.upper[d] = sunbox.upper[d];
        t->son[b]->box.lower[d] = sunbox.lower[d];
    }*/
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
        t->son[b]->box = sunbox; //?
        insertTree(p, t->son[b]);
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
            compF_BH(t->son[i], root, diam);
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
    if (t != NULL) {
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

//TODO: implement sendParticles
void sendParticles(TreeNode *root, SubDomainKeyTree *s) {
    //allocate memory for s->numprocs particle lists in plist;
    //initialize ParticleList plist[to] for all processes to;
    buildSendlist(root, s, plist);
    repairTree(root); // here, domainList nodes may not be deleted
    for (int i=1; i<s->numprocs; i++) {
        int to = (s->myrank+i)%s->numprocs;
        int from = (s->myrank+s->numprocs-i)%s->numprocs;
        //send particle data from plist[to] to process to;
        // receive particle data from process from;
        //insert all received particles p into the tree using insertTree(&p, root);
    }
    delete plist;
}

//TODO: implement buildSendlist
void buildSendlist(TreeNode *t, SubDomainKeyTree *s, ParticleList *plist) {
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++) {
            buildSendlist(t->son[i]);
        }
        // start of the operation on *t
        if (t != NULL) {
            for (int i = 0; i < POWDIM; i++) {
                buildSendlist(t->son[i]);
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

//TODO: implement compPseudoParticlespar
void compPseudoParticlespar(TreeNode *root, SubDomainKeyTree *s) {
    compLocalPseudoParticlespar(root);
    //MPI_Allreduce(..., {mass, moments} of the lowest domainList nodes, MPI_SUM, ...);
    compDomainListPseudoParticlespar(root);
}

//TODO: implement compLocalPseudoParticlespar
void compLocalPseudoParticlespar(TreeNode *t) {
    //called recursively as in Algorithm 8.1;
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++) {
            compLocalPseudoParticlespar(t->son[i]);
        }
        // start of the operation on *t
        if ((!isLeaf(t)) && (t->node != domainList)) {
            // operations analogous to Algorithm 8.5
        }
    }
    // end of the operation on *t
}

//TODO: implement compDomainListPsudoParticlespar
void compDomainListPseudoParticlespar(TreeNode *t) {
    //called recursively as in Algorithm 8.1 for the coarse domainList-tree;
    // start of the operation on *t
    if (t->node == domainList) {
        // operations analogous to Algorithm 8.5
    }
    // end of the operation on *t
}

//TODO: implement symbolicForce
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

//TODO: implement compF_BHpar
void compF_BHpar(TreeNode *root, float diam, SubDomainKeyTree *s) {
    //allocate memory for s->numprocs particle lists in plist;
    //initialize ParticleList plist[to] for all processes to;
    compTheta(root, root, s, plist, diam);
    for (int i=1; i<s->numprocs; i++) {
        int to = (s->myrank+i)%s->numprocs;
        int from = (s->myrank+s->numprocs-i)%s->numprocs; send (pseudo-)particle data from plist[to] to process to; receive (pseudo-)particle data from process from;
        //insert all received (pseudo-)particles p into the tree using insertTree(&p, root);
    }
    //delete plist;
    compF_BH(root, diam);
}

//TODO: implement compTheta
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




