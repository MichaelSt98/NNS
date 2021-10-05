#include "../include/Particle.h"
#include "../include/Integrator.h"
#include "../include/ConfigParser.h"
#include "../include/Renderer.h"
#include "../include/Logger.h"

#include <iostream>
#include <random>

#include <highfive/H5File.hpp>

// extern variable from Logger has to be initialized here
structlog LOGCFG = {};

// function declarations
void initData_BH(TreeNode **root, Box *domain, int N);
void initParticlesFromFile(TreeNode **root, Box *domain, ConfigParser &confP);
Renderer* initRenderer(ConfigParser &confP);


//function implementations
void initData_BH(TreeNode **root, Box *domain, int N, ConfigParser &confP) {

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
}

void initParticlesFromFile(TreeNode **root, Box *domain, ConfigParser &confP){

    int N = confP.getVal<int>("numParticles");

    HighFive::File file(confP.getVal<std::string>("initFile"), HighFive::File::ReadOnly);

    // containers to be filled
    double m;
    std::vector<std::vector<double>> x, v;
    Particle p[N];
    Particle *current;

    // read datasets from file
    HighFive::DataSet mass = file.getDataSet("/m");
    HighFive::DataSet pos = file.getDataSet("/x");
    HighFive::DataSet vel = file.getDataSet("/v");

    // read data
    mass.read(m);
    pos.read(x);
    vel.read(v);

    for (int i=1; i<N; i++) {
        current = &p[i];
        current->x[0] = x[i][0];
        current->x[1] = x[i][1];
        current->x[2] = x[i][2];
        current->v[0] = v[i][0];
        current->v[1] = v[i][1];
        current->v[2] = v[i][2];
        current->F[0] = 0.0;
        current->F[1] = 0.0;
        current->F[2] = 0.0;
        current->m = m;
    }

    *root = (TreeNode*)calloc(1, sizeof(TreeNode));
    (*root)->p = p[0]; //(first particle with number i=1); //1
    (*root)->box = *domain;

    for (int i=1; i<N; i++) //i=2, <=N
        insertTree(&p[i], *root);
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

int main() {

    // initialize logger
    LOGCFG.headers = true;
    LOGCFG.level = ERROR;

    ConfigParser confP { "config.info" };

    int width = confP.getVal<int>("width");
    int height = confP.getVal<int>("height");

    const double systemSize {confP.getVal<double>("systemSize")};
    const int N {confP.getVal<int>("numParticles")};
    const double delta_t {confP.getVal<double>("timeStep")};
    const double t_end {confP.getVal<double>("timeEnd")};


    char *image = new char[width*height*3];
    double *hdImage = new double[width*height*3];

    TreeNode *root;
    Box domain;

    for (int i=0; i<DIM; i++) {
        domain.lower[i] = -systemSize;
        domain.upper[i] = systemSize;
    }

    Renderer *renderer = initRenderer(confP);

    //inputParameters_BH(&delta_t, &t_end, &box, &theta, &N);

    if (confP.getVal<bool>("readInitDistFromFile")){
        initParticlesFromFile(&root, &domain, confP);
    } else {
        initData_BH(&root, &domain, N, confP);
    }
    timeIntegration_BH(0, delta_t, t_end, root, domain, renderer, image, hdImage);
    freeTree_BH(root);

    return 0;
}


