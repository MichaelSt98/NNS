//
// Created by Michael Staneker on 18.03.21.
//

#include "../include/Particle.h"
#include "../include/Tree.h"
#include "../include/Constants.h"

#include <iostream>

int outputRank = 1;

void initializeParticles(Particle *particle, int N);
void createLocalParticleList(SubDomainKeyTree s, ParticleList *pList, Particle *particles, int numberOfParticles);

void createParticleDatatype(MPI_Datatype datatype);

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
    MPI_Datatype mpiParticle;
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
    Particle pArray[particlesPerProcess];
    Particle pReceived[particlesPerProcess];

    convertListToArray(s, particleList, pArray, particlesPerProcess);

    /** -------------------------------------- */
    if (s.myrank == outputRank) {
        std::cout << "Sending and receiving particles..." << std::endl;
    }
    int from = s.myrank;
    int to;
    if (from == 1) {
        to = 0;
    }
    else {
        to = 1;
    }

    MPI_Request req;
    MPI_Status stat;

    MPI_Isend(pArray, particlesPerProcess, mpiParticle, to, 17, MPI_COMM_WORLD, &req);
    MPI_Recv(pReceived, particlesPerProcess, mpiParticle, to, 17, MPI_COMM_WORLD, &stat);

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
            std::cout << "\texpected particle " << i << ": " << "(" << pArray[i].x[0] << ", " << pArray[i].x[1] << ", " << pArray[i].x[2] << ")" << std::endl;
        }
    }*/
    if (s.myrank == outputRank) {
        std::cout << "RECEIVED" << std::endl;
        std::cout << "rank: " << s.myrank << std::endl;
        for (int i=0; i < particlesPerProcess; i++) {
            std::cout << "\treceived particle " << i << ": "
                        << "\n\t\t x = "<< "(" << pReceived[i].x[0] << ", " << pReceived[i].x[1] << ", " << pReceived[i].x[2] << ")"
                        << "\n\t\t v = "<< "(" << pReceived[i].v[0] << ", " << pReceived[i].v[1] << ", " << pReceived[i].v[2] << ")" << std::endl;
        }
    }


    /** -------------------------------------- */
    if (s.myrank == outputRank) {
        std::cout << "Finished..." << std::endl;
    }

    MPI_Finalize();

}