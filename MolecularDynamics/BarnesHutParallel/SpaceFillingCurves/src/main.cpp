#include "../include/Particle.h"
#include "../include/Integrator.h"
#include "../include/ConfigParser.h"
#include "../include/Renderer.h"
#include "../include/Logger.h"
#include "../include/H5Profiler.h"

#include <iostream>
#include <fstream>
#include <bitset>
#include <random>
#include <mpi.h>

#include <highfive/H5File.hpp>

// extern variable from Logger has to be initialized here
structlog LOGCFG = {};

MPI_Datatype mpiParticle;

// MPI output process rank number
int outputRank = 0;

// function declarations
void createParticleDatatype(MPI_Datatype *datatype);
void initParticles(SubDomainKeyTree *s, Particle *pArray, int ppp, ConfigParser &confP);
void initParticlesFromFile(SubDomainKeyTree *s, Particle *pArray, int ppp, ConfigParser &confP);
Renderer* initRenderer(ConfigParser &confP);

//create MPI datatype for Particle struct
void createParticleDatatype(MPI_Datatype *datatype) {
    int mpiParticleLengths[6] = {1, DIM, DIM, DIM, 1, 1};
    //const MPI_Aint mpiParticleDisplacements[6] ={ 0, sizeof(float), 2*sizeof(float), 3*sizeof(float), 4*sizeof(float), 4*sizeof(float) + sizeof(bool) };
    const MPI_Aint mpiParticleDisplacements[6] ={ offsetof(Particle, m), offsetof(Particle, x), offsetof(Particle, v),
                                                  offsetof(Particle, F), offsetof(Particle, moved), offsetof(Particle, todelete) };
    MPI_Datatype mpiParticleTypes[6] = { MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_CXX_BOOL, MPI_CXX_BOOL }; // MPI_C_BOOL ?
    MPI_Type_create_struct(6, mpiParticleLengths, mpiParticleDisplacements, mpiParticleTypes, datatype);
    MPI_Type_commit(datatype);
}

void initParticles(SubDomainKeyTree *s, Particle *pArray, int ppp, ConfigParser &confP) {
    using std::uniform_real_distribution;
    float systemSize{confP.getVal<float>("systemSize")};
    uniform_real_distribution<float> randAngle (0.0, 200.0 * PI);
    uniform_real_distribution<float> randRadius (0.0, systemSize/4.0);
    uniform_real_distribution<float> randHeight (0.0, systemSize/3.0);
    std::random_device rd;
    std::cout << "rd(): " << rd() << std::endl;
    //std::default_random_engine gen (rd());
    unsigned int seed = 2568239274 + s->myrank*1000;
    std::default_random_engine gen (seed);
    float angle;
    float radius;
    float radiusOffset;
    float velocity;

    Particle *current;

    for (int i=0; i<ppp; i++) { //1
        angle = randAngle(gen);
        radiusOffset = randRadius(gen);
        radius = sqrt(systemSize/2.-radiusOffset); //*sqrt(randRadius(gen));
        //velocity = pow(((G*(SOLAR_MASS+((radius)/systemSize)*SOLAR_MASS)) / (radius*TO_METERS)), 0.5);

        std::uniform_real_distribution<float> dist(-systemSize/2., systemSize/2.);

        velocity = confP.getVal<float>("initVel");
        current = &(pArray[i]);

        current->x[0] = radius * cos(angle);
        current->x[1] = radius * sin(angle);
        current->x[2] = randHeight(gen) - systemSize/6.;
        current->v[0] =  velocity*sin(angle);
        current->v[1] = -velocity*cos(angle);
        current->v[2] = 0.0; //dist(gen)/75. * velocity;

        current->F[0] = 0.0;
        current->F[1] = 0.0;
        current->F[2] = 0.0;
        current->m = confP.getVal<float>("initMass"); // SOLAR_MASS/N;
    }
}

void initParticlesFromFile(SubDomainKeyTree *s, Particle *pArray, int ppp, ConfigParser &confP) {

    HighFive::File file(confP.getVal<std::string>("initFile"), HighFive::File::ReadOnly);

    // containers to be filled
    double m;
    std::vector<std::vector<double>> x, v;

    // read datasets from file
    HighFive::DataSet mass = file.getDataSet("/m");
    HighFive::DataSet pos = file.getDataSet("/x");
    HighFive::DataSet vel = file.getDataSet("/v");

    // read data
    mass.read(m);
    pos.read(x);
    vel.read(v);

    // each process reads only a portion of the init file
    // j denotes global particle list index, i denotes local index
    for(int j=s->myrank*ppp; j < (s->myrank+1)*ppp; j++){
        int i = j-s->myrank*ppp;
        pArray[i].m = m;
        pArray[i].x[0] = x[j][0];
        pArray[i].x[1] = x[j][1];
        pArray[i].x[2] = x[j][2];
        pArray[i].v[0] = v[j][0];
        pArray[i].v[1] = v[j][1];
        pArray[i].v[2] = v[j][2];
    }
}

Renderer* initRenderer(ConfigParser &confP){

    // initialize renderer
    auto *renderer_ = new Renderer(
            confP.getVal<int>("numParticles"),
            confP.getVal<int>("width"),
            confP.getVal<int>("height"),
                    confP.getVal<int>("depth"),
            confP.getVal<double>("renderScale"),
            confP.getVal<double>("maxVelColor"),
            confP.getVal<double>("minVelColor"),
            confP.getVal<double>("particleBrightness"),
            confP.getVal<double>("particleSharpness"),
            confP.getVal<int>("dotSize"),
            confP.getVal<double>("systemSize"),
            confP.getVal<int>("renderInterval"));

    return renderer_;
}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    SubDomainKeyTree s;
    MPI_Comm_rank(MPI_COMM_WORLD, &s.myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &s.numprocs);

    ConfigParser confP{ConfigParser("config.info")};

    createParticleDatatype(&mpiParticle);

    LOGCFG.headers = true;
    LOGCFG.level = DEBUG;
    LOGCFG.myrank = s.myrank;
    LOGCFG.outputRank = confP.getVal<int>("outputRank");

    float t = 0;
    float delta_t = confP.getVal<float>("timeStep");
    float t_end = confP.getVal<float>("timeEnd");

    // check if result should be written to h5 file instead of rendering
    bool h5Dump = confP.getVal<bool>("h5Dump");

    Logger(DEBUG) << "Initialize h5 profiling file";

    int steps = (int)round(t_end/delta_t)+1;
    Logger(DEBUG) << "TOTAL STEP COUNT = " << steps;

    int loadBalancingInterval = confP.getVal<int>("loadBalancingInterval");

    H5Profiler &profiler = H5Profiler::getInstance("log/performance.h5", s.numprocs);

    // Total particle count per process
    profiler.createValueDataSet<int>("/general/numberOfParticles", steps);

    // Load balancing profiling
    int lbSteps = (int)round(steps/loadBalancingInterval)+1;
    profiler.createTimeDataSet("/loadBalancing/totalTime", lbSteps);
    //profiler.createTimeDataSet("/loadBalancing/updateRange/sort", lbSteps);
    profiler.createValueDataSet<int>("/loadBalancing/sendParticles/receiveLength",
                                     lbSteps);
    profiler.createVectorDataSet<int>("/loadBalancing/sendParticles/sendLengths",
                                      lbSteps, s.numprocs);

    // Force computation profiling
    profiler.createTimeDataSet("/forceComputation/totalTime", steps);
    profiler.createValueDataSet<int>("/compF_BHpar/receiveLength", steps);
    profiler.createVectorDataSet<int>("/compF_BHpar/sendLengths", steps, s.numprocs);

    // Updating positions and velocities profiling
    profiler.createTimeDataSet("/updatePosVel/totalTime", steps);

    Logger(DEBUG) << "TOTAL LOAD BALANCING STEPS = " << lbSteps;

    // declarations for renderer related variables, unused if h5Dump
    int width, height;
    char *image;
    double *hdImage;
    Renderer *renderer;

    if (!h5Dump) {
        // initialize renderer
        width = confP.getVal<int>("width");
        height = confP.getVal<int>("height");

        // TODO: only initialize in rank 0 process if possible
        image = new char[2 * width * height * 3];
        hdImage = new double[2 * width * height * 3];
        renderer = initRenderer(confP);
    }

    const float systemSize{confP.getVal<float>("systemSize")};
    Box domain;
    for (int i = 0; i < DIM; i++) {
        domain.lower[i] = -systemSize;
        domain.upper[i] = systemSize;
    }

    int N = confP.getVal<int>("numParticles"); //100;

    Logger(INFO) << "----------------------------------";
    Logger(INFO) << "NUMBER OF PARTICLES: " << N;
    Logger(INFO) << "----------------------------------";

    Particle *pArrayAll;
    if (s.myrank == 0) {
        pArrayAll = new Particle[N];
    }

    if(N % s.numprocs != 0){
        Logger(ERROR) << "Number of particles must have number of processes as divisor. - Aborting.";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    int ppp = N/s.numprocs;
    Particle *pArray;
    pArray = new Particle[ppp];

    if (confP.getVal<bool>("readInitDistFromFile")){
        initParticlesFromFile(&s, pArray, ppp, confP);
    } else {
        initParticles(&s, pArray, ppp, confP);
    }

    /**********/
    //TODO: skip this block and generate ranges distributed or read from file

    // collect all particles on one process
    MPI_Gather(pArray, ppp, mpiParticle, &pArrayAll[0], ppp, mpiParticle, 0, MPI_COMM_WORLD);

    if (s.myrank == 0) {

        TreeNode *rootAll;
        rootAll = (TreeNode *) calloc(1, sizeof(TreeNode));

        rootAll->p = pArrayAll[0];
        rootAll->box = domain;

        for (int i = 1; i < N; i++) {
            insertTree(&pArrayAll[i], rootAll);
        }
        createRanges(rootAll, N, &s);
        createDomainList(rootAll, 0, 0, &s);
    }

    if (s.myrank != 0) {
        s.range = new keytype[s.numprocs+1];
    }

    MPI_Bcast(s.range, s.numprocs+1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    /**********/

    /** Actual simulation **/
    TreeNode *root;
    root = (TreeNode *) calloc(1, sizeof(TreeNode));
    root->box = domain;

    createDomainList(root, 0, 0, &s);

    for (int i = 0; i < ppp; i++) {
        insertTree(&pArray[i], root);
    }

    profiler.disableWrite();
    sendParticles(root, &s);
    profiler.enableWrite();

    compPseudoParticlesPar(root, &s);

    outputTree(root, false);

    bool render = confP.getVal<bool>("render");
    bool processColoring = confP.getVal<bool>("processColoring");

    timeIntegration_BH_par(t, delta_t, t_end, root->box.upper[0] - root->box.lower[0], root, &s,
                           renderer, image, hdImage, render, processColoring,
                           h5Dump, confP.getVal<int>("h5DumpEachTimeSteps"), loadBalancingInterval);

    MPI_Finalize();
    return 0;
}


