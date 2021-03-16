#include "../include/Particle.h"
#include "../include/Integrator.h"
#include "../include/Tree.h"
#include "../include/Domain.h"

#include <iostream>
#include <random>
#include <mpi.h>

void initData_BH(TreeNode **root, Box *domain, SubDomainKeyTree  *s, int N);

void initData_BH(TreeNode **root, Box *domain, SubDomainKeyTree *s, int N) {

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

        velocity = 0.01;

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
        current->m = 0.01; // SOLAR_MASS/N;
    }

    *root = (TreeNode*)calloc(1, sizeof(TreeNode));
    createDomainList(root, 0, 0, s);
    (*root)->p = p[0]; //(first particle with number i=1); //1
    (*root)->box = *domain;

    for (int i=1; i<N; i++) //i=2, <=N
        insertTree(&p[i], *root);
}


int main(int argc, char *argv[]) {

    MPI_Init(argc, argv);

    float systemSize = 3.0;

    TreeNode *root;
    Box box;
    for (int i=0; i<DIM; i++) {
        box.lower[i] = -systemSize;
        box.upper[i] = systemSize;
    }
    SubDomainKeyTree  s;
    s.range = 0; //TODO: set range for sub domain key tree

    float delta_t = 0.01;
    float t_end = 1.0; //0.1;
    int N = 10; //1000;

    //inputParameters_BH(&delta_t, &t_end, &box, &theta, &N);

    initData_BH(&root, &box, &s, N);
    timeIntegration_BH(0, delta_t, t_end, root, box, &s);

    freeTree_BH(root);
    free(s.range);
    MPI_Finalize();
    return 0;
}


