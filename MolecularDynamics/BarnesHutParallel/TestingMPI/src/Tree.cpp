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


/*ParticleList* build_particle_list(TreeNode *t, ParticleList *pLst){
    /*
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
     */
/*}*/

void get_particle_array(TreeNode *root, Particle *p){
    /*
    auto pLst = new ParticleList;
    build_particle_list(root, pLst);
    int pIndex = 0;
    while(pLst->next){
        p[pIndex] = pLst->p;
        //std::cout << "Adding to *p: x = (" << p[pIndex].x[0] << ", " << p[pIndex].x[1] << ", " << p[pIndex].x[2] << ")" << std::endl;
        pLst = pLst->next;
        ++pIndex;
    }
     */
}


//TODO: implement sendParticles (Sending Particles to Their Owners and Inserting Them in the Local Tree)
// determine right amount of memory which has to be allocated for the `buffer`,
// by e.g. communicating the length of the message as prior message or by using other MPI commands
void sendParticles(TreeNode *root, SubDomainKeyTree *s) {
    /*
    //allocate memory for s->numprocs particle lists in plist;
    //initialize ParticleList plist[to] for all processes to;
    ParticleList * plist;
    plist = new ParticleList[s->numprocs];
    int plistLengthSend;
    plistLengthSend = new int[s->numprocs];
    int plistLengthReceive;
    plistLengthReceive = new int[s->numprocs];
    buildSendlist(root, s, plist, plistLength);
    repairTree(root); // here, domainList nodes may not be deleted
    //convert list to array for better sending
    //TODO send and receive plistLengthReceive
    // receive and sum over to get real length
    int *pArrays;
    pArrays = new int*[s->numprocs];
    for (int i=1; i<s->numprocs; i++) {
        pArrays[i] = new int [plistLengthSend[i]];
        int counter = 0;
        while (plist[s->myrank]) {
            pArrays[i][counter] = plist[s->myrank].p;
        }
    }
    for (int i=1; i<s->numprocs; i++) {
        if (i != s->myrank) { //needed?
            int to = (s->myrank + i) % s->numprocs;
            int from = (s->myrank + s->numprocs - i) % s->numprocs;
            //send particle data from plist[to] to process to;
            MPI_Send(plist[i]);
            //receive particle data from process from;
        }
        //insert all received particles p into the tree using insertTree(&p, root);
        for (int i=0; i<0 TODO length; i++) {
            insertTree(pArrays[s->myrank][i], root);
        }
    }
    delete plist;
    */
}

//TODO: implement buildSendlist (Sending Particles to Their Owners and Inserting Them in the Local Tree)
void buildSendlist(TreeNode *t, SubDomainKeyTree *s, ParticleList *plist, int *plistLength) {
    /*
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
                //the key of *t can be computed step by step in the recursion
                //insert t->p into list plist[proc];
                plist[proc]->p = p;
                plist[proc]->next = new ParticleList; //TODO: similar problem as with get_particle_array() ?!
                plistLength[proc]++;
                //mark t->p as to be deleted;
                t->p.todelete = true;
            }
        }
    }
    // end of the operation on *t
     */
}
