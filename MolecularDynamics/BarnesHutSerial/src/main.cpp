#include "../include/Particle.h"
#include "../include/Integrator.h"
#include "../include/ConfigParser.h"
#include "../include/Renderer.h"
#include "../include/Logger.h"

#include <iostream>
#include <random>

// extern variable from Logger has to be initialized here
structlog LOGCFG = {};

// function declarations
void initData_BH(TreeNode **root, Box *domain, int N);
Renderer* initRenderer(ConfigParser &confP);

//function implementations
void initData_BH(TreeNode **root, Box *domain, int N) {

    Particle p[N];
    using std::uniform_real_distribution;
    float systemSize = getSystemSize(domain);
    uniform_real_distribution<float> randAngle (0.0, 200.0 * 3.1415);
    uniform_real_distribution<float> randRadius (0, systemSize);
    uniform_real_distribution<float> randHeight (0.0, systemSize/1000.0);
    std::default_random_engine gen (0);
    float angle;
    float radius;
    float velocity;

    Particle *current;

    for (int i=0; i<N; i++) {
        angle = randAngle(gen);
        radius = sqrt(systemSize)*sqrt(randRadius(gen));
        //velocity = pow(((G*(SOLAR_MASS+((radius)/systemSize)*SOLAR_MASS)) / (radius*TO_METERS)), 0.5);

        velocity = 0.1;

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
        current->m = 0.1; // SOLAR_MASS/N;
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

    ConfigParser confP {ConfigParser("config.info")};

    int width = confP.getVal<int>("width");
    int height = confP.getVal<int>("height");

    char *image = new char[width*height*3];
    double *hdImage = new double[width*height*3];

    const float systemSize {confP.getVal<float>("systemSize")};
    TreeNode *root;
    Box box;

    for (int i=0; i<DIM; i++) {
        box.lower[i] = -systemSize;
        box.upper[i] = systemSize;
    }

    const float delta_t {confP.getVal<float>("timeStep")};
    const float t_end {confP.getVal<float>("timeEnd")};
    const int N {confP.getVal<int>("numParticles")};

    Renderer *renderer = initRenderer(confP);

    //inputParameters_BH(&delta_t, &t_end, &box, &theta, &N); //TODO
    initData_BH(&root, &box, N);
    timeIntegration_BH(0, delta_t, t_end, root, box, renderer, image, hdImage);
    freeTree_BH(root);

    return 0;
}


