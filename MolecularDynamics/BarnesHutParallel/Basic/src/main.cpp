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
void initData_BH(TreeNode **root, Box *domain, SubDomainKeyTree  *s, int N, ConfigParser &confP);
Renderer* initRenderer(ConfigParser &confP);

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

//function implementations

// TODO: providing parallel program with initial data:
//  * create/read all particles by process zero
//  * create/read particles by corresponding processes
//      * redistribution necessary
//  0. values of `range` are given
//  1. every process creates a local tree using `createDomainList`
//  2. particles read/created by process are inserted into the corresponding local tree
//  3. With help of the domainList flag it can be decided to which process the particle truly belongs
//  4. every process traverses its local tree, removes particles that are not assigned to it and sends particles to the appropriate process
//      4.1 receive particles and sort into local tree
//      .
//  See: Tree::sendParticles() and Tree::buildSendList()
void initData_BH(TreeNode **root, Box *domain, SubDomainKeyTree  *s, int N, ConfigParser &confP) {

    Particle p[N];
    using std::uniform_real_distribution;
    float systemSize = getSystemSize(domain);
    uniform_real_distribution<float> randAngle (0.0, 200.0 * PI);
    uniform_real_distribution<float> randRadius (0.0, systemSize/2.0);
    uniform_real_distribution<float> randHeight (0.0, systemSize/1000.0);
    std::default_random_engine gen (0);
    float angle;
    float radius;
    float radiusOffset;
    float velocity;

    Particle *current;

    current = &p[0];
    current->x[0] = 0.0;
    current->x[1] = 0.0;
    current->x[2] = 0.0;
    current->v[0] = 0.0;
    current->v[1] = 0.0;
    current->v[2] = 0.0;
    current->F[0] = 0.0;
    current->F[1] = 0.0;
    current->F[2] = 0.0;
    current->m = (N*N)*confP.getVal<float>("initMass"); // SOLAR_MASS/N;

    for (int i=1; i<N; i++) {
        angle = randAngle(gen);
        radiusOffset = randRadius(gen);
        radius = sqrt(systemSize-radiusOffset); //*sqrt(randRadius(gen));
        //velocity = pow(((G*(SOLAR_MASS+((radius)/systemSize)*SOLAR_MASS)) / (radius*TO_METERS)), 0.5);

        velocity = confP.getVal<float>("initVel");

        current = &p[i];
        current->x[0] =  radius*cos(angle);
        current->x[1] =  radius*sin(angle);
        current->x[2] =  randHeight(gen)-systemSize/2000.0;
        current->v[0] =  velocity*sin(angle);
        current->v[1] = -velocity*cos(angle);
        current->v[2] = 0.0;
        current->F[0] = 0.0;
        current->F[1] = 0.0;
        current->F[2] = 0.0;
        current->m = confP.getVal<float>("initMass"); // SOLAR_MASS/N;
    }

    *root = (TreeNode*)calloc(1, sizeof(TreeNode));

    (*root)->p = p[0]; //(first particle with number i=1); //1
    (*root)->box = *domain;

    for (int i=1; i<N; i++) //i=2, <=N
        insertTree(&p[i], *root);

    createRanges(*root, N, s, confP.getVal<int>("dummyDomains"));

    createDomainList(*root, 0, 0, s);

    Particle *pOut = new Particle[N];
    keytype *particleKeys = new keytype[N];

    get_particle_array(*root, pOut);
    int pCounter { 0 };
    getParticleKeys(*root, particleKeys, pCounter);

    const std::string &csvFile { "./pKeysBasic.csv" };
    Logger(INFO) << "Writing particle keys to file '" << csvFile << "' ...";
    std::ofstream outf{"./pKeysBasic.csv"};
    if (!outf) {
        Logger(ERROR) << "An error occurred while opening 'particleKeysBasic'. - Aborting.";
        exit(1); //TODO: throw exception
    }
    for (int i = 0; i < N; ++i) {
        outf << std::bitset<64>(particleKeys[i]) << ";"
                << pOut[i].x[0] << ";" << pOut[i].x[1] << ";" << pOut[i].x[2] << '\n';
    }
    Logger(INFO) << "... done.";
}

void initParticles(SubDomainKeyTree *s, Box *domain, Particle *pArray, int ppp, ConfigParser &confP) {
    //pArray = new Particle[ppp];
    using std::uniform_real_distribution;
    //float systemSize = confP.//getSystemSize(domain);
    float systemSize{confP.getVal<float>("systemSize")};
    uniform_real_distribution<float> randAngle (0.0, 200.0 * PI);
    uniform_real_distribution<float> randRadius (0.0, systemSize/2.0);
    uniform_real_distribution<float> randHeight (0.0, systemSize/1000.0);
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

        velocity = confP.getVal<float>("initVel");

        current = &(pArray[i]);
        current->x[0] =  radius*cos(angle);
        current->x[1] =  radius*sin(angle);
        current->x[2] =  randHeight(gen)-systemSize/2000.0;
        current->v[0] =  velocity*sin(angle);
        current->v[1] = -velocity*cos(angle);
        current->v[2] = 0.0;
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

    //TODO: needed to be called by every process?
    // initialize logger
    LOGCFG.headers = true;
    LOGCFG.level = DEBUG;
    LOGCFG.myrank = s.myrank;

    int width = confP.getVal<int>("width");
    int height = confP.getVal<int>("height");

    //Particle rootParticle {};

    const float systemSize{confP.getVal<float>("systemSize")};
    Box domain;
    for (int i = 0; i < DIM; i++) {
        domain.lower[i] = -systemSize;
        domain.upper[i] = systemSize;
    }

    int N = confP.getVal<int>("numParticles"); //100;

    Particle *pArrayAll;
    if (s.myrank == 0) {
        pArrayAll = new Particle[N];
    }

    int ppp = N/s.numprocs;
    Particle *pArray;
    pArray = new Particle[ppp];

    initParticles(&s, &domain, pArray, ppp, confP);

    //if (s.myrank == outputRank) {
    //    for (int i=0; i<ppp; i++) {
    //        Logger(INFO) << "pArray[" << i << "].x = (" << pArray[i].x[0] << ", " << pArray[i].x[1] << ", " << pArray[i].x[2] << ")";
    //    }
    //}

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Gather(pArray, ppp, mpiParticle, &pArrayAll[0], ppp, mpiParticle, 0, MPI_COMM_WORLD);

    if (s.myrank == 0) {
        for (int i = 0; i < N; i++) {
            //Logger(INFO) << "pArrayAll[" << i << "].x = (" << pArrayAll[i].x[0] << ", " << pArrayAll[i].x[1] << ", "
              //           << pArrayAll[i].x[2] << ")";
        }


        TreeNode *rootAll;
        rootAll = (TreeNode *) calloc(1, sizeof(TreeNode));

        //rootAll->p = pArrayAll[0]; //(first particle with number i=1); //1
        rootAll->box = domain;
        //rootAll->p = rootParticle;

        for (int i = 0; i < N; i++) {
            insertTree(&pArrayAll[i], rootAll);
        }

        //createRanges(root, N, &s, confP.getVal<int>("dummyDomains"));
        createRanges(rootAll, N, &s);
        //createDomainList(rootAll, 0, 0, &s);
        //compPseudoParticles(rootAll);

        //output_tree(rootAll, false);
        //freeTree_BH(rootAll);
    }

    //send ranges to all processes
    //MPI_UNSIGNED_LONG
    //keytype *range[s.numprocs+1];

    //if (s.myrank == 0) {
    //    range = s.range;
    //}

    if (s.myrank != 0) {
        s.range = new keytype[s.numprocs+1];
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(s.range, s.numprocs+1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    //if (s.myrank == 1) {
    //    for (int i=0; i<s.numprocs+1; i++) {
    //        Logger(INFO) << std::bitset<64>(s.range[i]);
    //    }
    //}

    TreeNode *root;
    root = (TreeNode *) calloc(1, sizeof(TreeNode));

    createDomainList(root, 0, 0, &s);

    //root->p = pArray[0]; //(first particle with number i=1); //1
    root->box = domain;
    //root->p = rootParticle;

    for (int i = 0; i < ppp; i++) {
        //Logger(INFO) << "Insert particle: " << i;
        insertTree(&pArray[i], root);
    }

    //createRanges(root, N, &s, confP.getVal<int>("dummyDomains"));
    //createRanges(rootAll, N, &s);
    //createDomainList(root, 0, 0, &s);
    //compPseudoParticles(root);

    //compPseudoParticlespar(root, &s);

    //if (s.myrank == outputRank) {
    //    output_tree(root, true);
    //}

    //output_tree(root, false);

    Logger(ERROR) << "BEFORE SENDING PARTICLES";
    output_tree(root, false);

    MPI_Barrier(MPI_COMM_WORLD);

    sendParticles(root, &s);

    Logger(ERROR) << "AFTER SENDING PARTICLES";
    output_tree(root, false);
    //output_particles(root);

    //compPseudoParticlespar(root, &s);


    /*if (s.myrank == outputRank) {
        ConfigParser confP{ConfigParser("config.info")};

        int width = confP.getVal<int>("width");
        int height = confP.getVal<int>("height");

        char *image = new char[width * height * 3];
        double *hdImage = new double[width * height * 3];

        const float systemSize{confP.getVal<float>("systemSize")};
        TreeNode *root;
        Box box;
        for (int i = 0; i < DIM; i++) {
            box.lower[i] = -systemSize;
            box.upper[i] = systemSize;
        }

        const float delta_t{confP.getVal<float>("timeStep")};
        const float t_end{confP.getVal<float>("timeEnd")};
        const int N{confP.getVal<int>("numParticles")};

        Renderer *renderer = initRenderer(confP);

        //inputParameters_BH(&delta_t, &t_end, &box, &theta, &N);

        initData_BH(&root, &box, &s, N, confP);

        //TODO: timeIntegration_BH(0, delta_t, t_end, root, box, &s);

        //free resources
        freeTree_BH(root);
    }*/

    freeTree_BH(root);
    MPI_Finalize();
    return 0;
}


