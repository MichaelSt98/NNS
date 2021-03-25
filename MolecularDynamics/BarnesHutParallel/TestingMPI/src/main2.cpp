#include "../include/Particle.h"
#include "../include/Tree.h"
#include "../include/Constants.h"

#include <iostream>

int outputRank = 0;
MPI_Datatype mpiParticle;

void createParticleDatatype(MPI_Datatype *datatype) {
    //create MPI datatype for Particle struct
    int mpiParticleLengths[6] = {1, DIM, DIM, DIM, 1, 1};
    // not properly working
    //const MPI_Aint mpiParticleDisplacements[6] ={ 0, sizeof(float), 2*sizeof(float), 3*sizeof(float), 4*sizeof(float), 4*sizeof(float) + sizeof(bool) };
    // properly working
    const MPI_Aint mpiParticleDisplacements[6] ={ offsetof(Particle, m), offsetof(Particle, x), offsetof(Particle, v),
                                                  offsetof(Particle, F), offsetof(Particle, moved), offsetof(Particle, todelete) };
    MPI_Datatype mpiParticleTypes[6] = { MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_CXX_BOOL, MPI_CXX_BOOL }; // MPI_C_BOOL ?
    MPI_Type_create_struct(6, mpiParticleLengths, mpiParticleDisplacements, mpiParticleTypes, datatype);
    MPI_Type_commit(datatype);
}

void printParticleList(SubDomainKeyTree s, ParticleList *pList) {
    ParticleList * current = pList;
    int counter = 0;
    while (current) {
        if (s.myrank == outputRank) {
            std::cout << "p[" << counter << "] = (" << current->p.x[0] << ")" << std::endl;
        }
        current = current->next;
        counter++;
    }
}

void createLocalParticleList(SubDomainKeyTree s, ParticleList *pList, int proc) {

    int N = s.myrank + 10;
    if (s.myrank == outputRank) {
        std::cout << "particles within process " << s.myrank << ": " << N << std::endl;
    }

    Particle particles[N];
    float value = s.myrank * 100 + (proc*1000);
    for (int i=0; i<N; i++) {
        particles[i].x[0] = (float)i+value;
        particles[i].x[1] = (float)i+value;
        particles[i].x[2] = (float)i+value;
        particles[i].v[0] = (float)i+value;
        particles[i].v[1] = (float)i+value;
        particles[i].v[2] = (float)i+value;
        particles[i].F[0] = (float)i+value;
        particles[i].F[1] = (float)i+value;
        particles[i].F[2] = (float)i+value;
        particles[i].m = (float)(i)+value;
        particles[i].todelete = false;
        particles[i].moved = false;
    }

    ParticleList *current = pList;
    current->p = particles[0];
    for (int i=1; i<N; i++) {
        if (s.myrank == outputRank) {
            std::cout << "inserting p[" << i << "] = (" << particles[i].x[0] << ")" << std::endl;
        }
        current->next = new ParticleList;
        current = current->next;
        current->p = particles[i];
    }
}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    /** -------------------------------------- */

    SubDomainKeyTree s;
    MPI_Comm_rank(MPI_COMM_WORLD, &s.myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &s.numprocs);

    createParticleDatatype(&mpiParticle);

    if (s.myrank == outputRank) {
        std::cout << "Starting..." << std::endl;
    }


    ParticleList * plist;
    plist = new ParticleList[s.numprocs];

    //ParticleList ** plist;
    //plist = new ParticleList*[s.numprocs];
    //for (int proc= 0; proc<s.numprocs; proc++) {
    //    plist[proc] = new ParticleList;
    //}

    for (int proc = 0; proc<s.numprocs; proc++) {
        if (proc != s.myrank) {
            createLocalParticleList(s, &plist[proc], proc);
        }
    }

    for (int proc = 0; proc<s.numprocs; proc++) {
        if (proc != s.myrank) {
            if (s.myrank == outputRank) {
                std::cout << "length: " << getParticleListLength(&plist[proc]) << std::endl;
            }
        }
    }

    for (int proc = 0; proc<s.numprocs; proc++) {
        if (proc != s.myrank) {
            if (s.myrank == outputRank) {
                std::cout << "Printing particle list proc: " << proc << std::endl;
            }
            printParticleList(s, &plist[proc]);
        }
    }

    Particle ** pArray = new Particle*[s.numprocs];
    //pArray[s.myrank] = new Particle[1];

    int *plistLengthSend;
    plistLengthSend = new int[s.numprocs];
    plistLengthSend[s.myrank] = -1; // nothing to send to yourself

    int *plistLengthReceive;
    plistLengthReceive = new int[s.numprocs];
    plistLengthReceive[s.myrank] = -1; // nothing to receive from yourself

    //buildSendlist(root, s, plist);
    //repairTree(root); // here, domainList nodes may not be deleted

    //convert list to array for better sending and for lengths
    for (int proc=0; proc<s.numprocs; proc++) {
        if (proc != s.myrank) {
            plistLengthSend[proc] = getParticleListLength(&plist[proc]);
            if (s.myrank == outputRank) {
                std::cout << "plistLengthSend[" << proc << "] = " << plistLengthSend[proc] << std::endl;
            }
            pArray[proc] = new Particle[plistLengthSend[proc]];
            ParticleList * current = &plist[proc];
            for (int i = 0; i < plistLengthSend[proc]; i++) {
                pArray[proc][i] = current->p;
                current = current->next;
            }
        }
    }


    for (int proc=0; proc<s.numprocs; proc++) {
        if (proc != s.myrank) {
            for (int i = 0; i <plistLengthSend[proc]; i++) {
                if (s.myrank == outputRank) {
                    std::cout << "pArray[" << proc <<"][" << i <<"] = (" << pArray[proc][i].x[0] << ", "
                              << pArray[proc][i].x[0] << ", "
                              << pArray[proc][i].x[0] << ")" << std::endl;
                }
            }
        }
    }




    int reqCounter = 0;
    MPI_Request req[s.numprocs-1];
    MPI_Status stat[s.numprocs-1];

    //send plistLengthSend and receive plistLengthReceive
    for (int proc=0; proc<s.numprocs; proc++) {
        if (proc != s.myrank) {
            MPI_Isend(&plistLengthSend[proc], 1, MPI_INT, proc, 17, MPI_COMM_WORLD, &req[reqCounter]);
            MPI_Recv(&plistLengthReceive[proc], 1, MPI_INT, proc, 17, MPI_COMM_WORLD, &stat[reqCounter]);
            reqCounter++;
        }
    }
    MPI_Waitall(s.numprocs-1, req, stat);

    //sum over to get total amount of particles to receive
    int receiveLength = 0;
    for (int proc=0; proc<s.numprocs; proc++) {
        if (proc != s.myrank) {
            receiveLength += plistLengthReceive[proc];
        }
    }

    if (s.myrank == outputRank) {
        std::cout << "receiveLength = " << receiveLength << std::endl;
    }

    //pArrayReceive = new Particle[receiveLength];

    // allocate missing (sub)array for process rank
    pArray[s.myrank] = new Particle[receiveLength];


    //send and receive particles
    reqCounter = 0;
    int receiveOffset = 0;
    for (int proc=0; proc<s.numprocs; proc++) {
        if (proc != s.myrank) {
            MPI_Isend(pArray[proc], plistLengthSend[proc], mpiParticle, proc, 17, MPI_COMM_WORLD, &req[reqCounter]);
            MPI_Recv(pArray[s.myrank]+receiveOffset, plistLengthReceive[proc], mpiParticle, proc, 17, MPI_COMM_WORLD, &stat[reqCounter]);
            receiveOffset += plistLengthReceive[proc];
            reqCounter++;
        }
    }
    MPI_Waitall(s.numprocs-1, req, stat);

    if (s.myrank == outputRank) {
        for (int i=0; i<receiveLength; i++) {
            std::cout << "pArray[" << s.myrank << "][" << i << "]: (" << pArray[s.myrank][i].x[0] << ")" << std::endl;
        }
    }

    MPI_Finalize();

}

