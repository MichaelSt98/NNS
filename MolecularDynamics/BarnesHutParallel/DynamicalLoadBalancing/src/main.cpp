#include "../include/Particle.h"
#include "../include/Integrator.h"
#include "../include/ConfigParser.h"
#include "../include/Renderer.h"
#include "../include/Logger.h"

#include <iostream>
#include <fstream>
#include <bitset>
#include <random>
#include <mpi.h>

// extern variable from Logger has to be initialized here
structlog LOGCFG = {};

MPI_Datatype mpiParticle;

// MPI output process rank number
int outputRank = 0;

// function declarations
Renderer* initRenderer(ConfigParser &confP);
void createParticleDatatype(MPI_Datatype *datatype);

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
    uniform_real_distribution<float> randRadius (0.0, systemSize/2.0);
    uniform_real_distribution<float> randHeight (0.0, systemSize/5.0);
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
        radius = sqrt(systemSize-radiusOffset); //*sqrt(randRadius(gen));
        //velocity = pow(((G*(SOLAR_MASS+((radius)/systemSize)*SOLAR_MASS)) / (radius*TO_METERS)), 0.5);

        std::uniform_real_distribution<float> dist(-systemSize, systemSize);

        velocity = confP.getVal<float>("initVel");
        current = &(pArray[i]);

        current->x[0] = radius * cos(angle);
        current->x[1] = radius * sin(angle);
        current->x[2] = randHeight(gen) - systemSize/10.;
        current->v[0] =  velocity*sin(angle);
        current->v[1] = -velocity*cos(angle);
        current->v[2] = dist(gen)/75. * velocity;

        current->F[0] = 0.0;
        current->F[1] = 0.0;
        current->F[2] = 0.0;
        current->m = confP.getVal<float>("initMass"); // SOLAR_MASS/N;
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

    int width = confP.getVal<int>("width");
    int height = confP.getVal<int>("height");

    char *image;
    double *hdImage;
    Renderer *renderer;

    // TODO: only initialize in rank 0 process if possible
    image = new char[2 * width * height * 3];
    hdImage = new double[2 * width * height * 3];
    renderer = initRenderer(confP);

    const float systemSize{confP.getVal<float>("systemSize")};
    Box domain;
    for (int i = 0; i < DIM; i++) {
        domain.lower[i] = -systemSize;
        domain.upper[i] = systemSize;
    }

    int N = confP.getVal<int>("numParticles"); //100;

    Logger(ERROR) << "----------------------------------";
    Logger(ERROR) << "AMOUNT OF PARTICLES: " << N;
    Logger(ERROR) << "----------------------------------";

    Particle *pArrayAll;
    if (s.myrank == 0) {
        pArrayAll = new Particle[N];
    }

    int ppp = N/s.numprocs;
    Particle *pArray;
    pArray = new Particle[ppp];

    initParticles(&s, pArray, ppp, confP);

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

    TreeNode *root;
    root = (TreeNode *) calloc(1, sizeof(TreeNode));

    createDomainList(root, 0, 0, &s);

    root->box = domain;

    for (int i = 0; i < ppp; i++) {
        insertTree(&pArray[i], root);
    }

    sendParticles(root, &s);

    compPseudoParticlesPar(root, &s);

    outputTree(root, false);

    float delta_t = confP.getVal<float>("timeStep");
    float diam = root->box.upper[0] - root->box.lower[0];
    float t = 0;
    float t_end = confP.getVal<float>("timeEnd");

    bool render = confP.getVal<bool>("render");
    bool processColoring = confP.getVal<bool>("processColoring");

    timeIntegration_BH_par(t, delta_t, t_end, root->box.upper[0] - root->box.lower[0], root, &s,
                           renderer, image, hdImage, render, processColoring);

    MPI_Finalize();
    return 0;
}


