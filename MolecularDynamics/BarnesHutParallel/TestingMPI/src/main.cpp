//
// Created by Michael Staneker on 18.03.21.
//

#include "../include/Particle.h"
#include "../include/Tree.h"
#include "../include/Constants.h"

#include <iostream>

int outputRank = 1;
MPI_Datatype mpiParticle;

void initializeParticles(Particle *particle, int N);
void createLocalParticleList(SubDomainKeyTree s, ParticleList *pList, Particle *particles, int numberOfParticles);
void createParticleDatatype(MPI_Datatype datatype);
void convertListToArray(SubDomainKeyTree s, ParticleList *particleList, Particle *pArray, int particlesPerProcess);

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

void initializeParticles(Particle *particles, int N) {
    for (int i=0; i<N; i++) {
        particles[i].x[0] = (float)i;
        particles[i].x[1] = (float)i;
        particles[i].x[2] = (float)i;
        particles[i].v[0] = (float)i;
        particles[i].v[1] = (float)i;
        particles[i].v[2] = (float)i;
        particles[i].F[0] = (float)i;
        particles[i].F[1] = (float)i;
        particles[i].F[2] = (float)i;
        particles[i].m = (float)(i*1000);
        particles[i].todelete = false;
        particles[i].moved = false;
    }
}

void createLocalParticleList(SubDomainKeyTree s, ParticleList *pList, Particle *particles, int numberOfParticles) {

    int particlesPerProcess = numberOfParticles/s.numprocs;
    if (s.myrank == outputRank) {
        std::cout << "particles per process: " << particlesPerProcess << std::endl;
    }

    for (int i=0; i<particlesPerProcess; i++) {
        if (s.myrank == outputRank) {
            std::cout << "create local particle list i: " << i << std::endl;
        }
        pList->p = particles[i + s.myrank*particlesPerProcess];
        pList->next = new ParticleList;
        pList = pList->next;
    }
}

void convertListToArray(SubDomainKeyTree s, ParticleList *particleList, Particle *pArray, int particlesPerProcess) {
    int counter = 0;
    while(particleList->next) {
        if (s.myrank == outputRank) {
            std::cout << "counter: " << counter << std::endl;
        }
        pArray[counter] = particleList->p;
        particleList = particleList->next;
        counter++;
    }

    if (s.myrank == outputRank) {
        std::cout << "rank: " << s.myrank << std::endl;
        for (int i=0; i < particlesPerProcess; i++) {
            std::cout << "\tparticle " << i << ": " << "(" << pArray[i].x[0] << ", " << pArray[i].x[1] << ", " << pArray[i].x[2] << ")" << std::endl;
        }
    }
}

void exchangeParticles() {

}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    /** -------------------------------------- */
    std::cout << "Starting..." << std::endl;

    SubDomainKeyTree  s;
    MPI_Comm_rank(MPI_COMM_WORLD, &s.myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &s.numprocs);
    //s.range = 0; //TODO: set range for sub domain key tree

    /** -------------------------------------- */
    if (s.myrank == outputRank) {
        std::cout << "Creating MPI Datatype for Particle struct..." << std::endl;
    }
    //create MPI datatype for Particle struct
    //MPI_Datatype mpiParticle;
    createParticleDatatype(&mpiParticle);

    /** -------------------------------------- */
    if (s.myrank == outputRank) {
        std::cout << "Creating particles for testing..." << std::endl;
    }
    //create particles for testing
    int numberOfParticles = 10;
    Particle particles[numberOfParticles];
    initializeParticles(particles, numberOfParticles);

    /** -------------------------------------- */
    if (s.myrank == outputRank) {
        std::cout << "Converting particle array to particle list..." << std::endl;
    }
    //convert array to particle list
    ParticleList *particleList;
    auto pList = new ParticleList;
    particleList = pList;
    createLocalParticleList(s, pList, particles, numberOfParticles);
    int particlesPerProcess = numberOfParticles/s.numprocs;

    /** -------------------------------------- */
    if (s.myrank == outputRank) {
        std::cout << "Converting back to an array..." << std::endl;
    }
    //convert back to array
    Particle pArray[s.numprocs][particlesPerProcess];

    convertListToArray(s, particleList, pArray[s.myrank], particlesPerProcess);

    /** -------------------------------------- */
    if (s.myrank == outputRank) {
        std::cout << "Sending and receiving particles..." << std::endl;
    }

    int messageLength[s.numprocs];
    messageLength[s.myrank] = particlesPerProcess;

    MPI_Request req;
    MPI_Status stat;

    for (int proc=0; proc < s.numprocs; proc++) {
        if (proc != s.myrank) {
            MPI_Isend(&messageLength[s.myrank], 1, MPI_INT, proc, 17, MPI_COMM_WORLD, &req);
            MPI_Recv(&messageLength[proc], 1, MPI_INT, proc, 17, MPI_COMM_WORLD, &stat);
        }
    }

    MPI_Wait(&req, &stat);

    if (s.myrank == outputRank) {
        std::cout << "Message length: " << *messageLength << std::endl;
    }

    //MPI_Request req;
    //MPI_Status stat;

    for (int proc=0; proc < s.numprocs; proc++) {
        if (proc != s.myrank) {
            MPI_Isend(pArray[s.myrank], messageLength[s.myrank], mpiParticle, proc, 17, MPI_COMM_WORLD, &req);
            MPI_Recv(pArray[proc], messageLength[proc], mpiParticle, proc, 17, MPI_COMM_WORLD, &stat);
        }
    }

    MPI_Wait(&req, &stat);
    //MPI_Request_free(&req);


    /** -------------------------------------- */
    if (s.myrank == outputRank) {
        std::cout << "Checking results..." << std::endl;
    }
    /*if (s.myrank != outputRank) {
        std::cout << "EXPECTED" << std::endl;
        std::cout << "rank: " << s.myrank << std::endl;
        for (int i=0; i < particlesPerProcess; i++) {
            std::cout << "\texpected particle " << i << ": " << "(" << pArray[s.myrank][i].x[0] << ", " << pArray[s.myrank][i].x[1] << ", " << pArray[s.myrank][i].x[2] << ")" << std::endl;
        }
    }*/
    if (s.myrank == outputRank) {
        std::cout << "RECEIVED" << std::endl;
        std::cout << "rank: " << s.myrank << std::endl;
        for (int proc = 0; proc < s.numprocs; proc++) {
            if (proc != s.myrank) {
                for (int i = 0; i < particlesPerProcess; i++) {
                    std::cout << "\tproc[" << proc <<  "] received particle " << i << ": "
                              << "\n\t\t x = " << "(" << pArray[proc][i].x[0] << ", " << pArray[proc][i].x[1] << ", "
                              << pArray[proc][i].x[2] << ")"
                              << "\n\t\t v = " << "(" << pArray[proc][i].v[0] << ", " << pArray[proc][i].v[1] << ", "
                              << pArray[proc][i].v[2] << ")" << std::endl;
                }
            }
        }
    }


    /** -------------------------------------- */
    if (s.myrank == outputRank) {
        std::cout << "Finished..." << std::endl;
    }

    MPI_Finalize();

}

