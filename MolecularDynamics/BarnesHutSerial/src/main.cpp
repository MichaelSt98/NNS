
#include <iostream>
#include <random>

#define DIM 3
#define pwrtwo(x) (1 << (x))
#define POWDIM pwrtwo(DIM)

//#define NULL nullptr

const float PI = 3.14159265358979323846;
const float TO_METERS = 1.496e11;
const float G = 6.67408e-11;

const float SOLAR_MASS = 2.0e30;
const float theta = 0.7;

typedef struct {
    float m;
    float x[DIM];
    float v[DIM];
    float F[DIM];
    bool moved = false;
    bool todelete = false;
} Particle;

typedef struct Box {
    float lower[DIM];
    float upper[DIM];
} Box;

typedef struct TreeNode {
    Particle p;
    Box box;
    struct TreeNode *son[POWDIM];
} TreeNode;

float getSystemSize(Box *b);
bool particleWithinBox(Particle &p, Box &b);
bool isLeaf(TreeNode *t);
void insertTree(Particle *p, TreeNode *t);
int sonNumber(Box *box, Box *sonbox, Particle *p);
void initData_BH(TreeNode **root, Box *domain, int N);
void compPseudoParticles(TreeNode *t);
void compF_BH(TreeNode *t, TreeNode *root, float diam);
void force(Particle *i, Particle *j);
void force_tree(TreeNode *tl, TreeNode *t, float diam);
void updateX(Particle *p, float delta_t);
void updateV(Particle *p, float delta_t);
void compX_BH(TreeNode *t, float delta_t);
void compV_BH(TreeNode *t, float delta_t);
void moveParticles_BH(TreeNode *root);
void setFlags(TreeNode *t);
void moveLeaf(TreeNode *t, TreeNode *root);
void repairTree(TreeNode *t);
void output_particles(TreeNode *root);

/*void FUNCTION(TreeNode *t) {
    if (t != NULL) {
        for (int i=0; i<POWDIM; i++)
            FUNCTION(t->son[i]);
        Perform the operations of the function FUNCTION on *t ;
    }
}*/

float getSystemSize(Box *b) {
    float systemSize = 0.0;
    float temp = 0.0;
    for (int i=0; i<DIM; i++) {
        float temp = 0.0;
        if (b->lower[i] > b->upper[i]) {
            temp = b->lower[i];
        }
        else {
            temp = b->upper[i];
        }
        if (temp > systemSize) {
            systemSize = temp;
        }
    }
    return systemSize;
}

bool particleWithinBox(Particle &p, Box &b) {
    for (int i=0; i<DIM; i++) {
        if (p.x[i] > b.upper[i] || p.x[i] < b.lower[i]) {
            return false;
        }
    }
    return true;
}

bool isLeaf(TreeNode *t) {
    if (t != NULL) {
        for (int i=0; i<POWDIM; i++) {
            if (t->son[i] != NULL) {
                return false;
            }
        }
        return true;
    }
    return false; // TODO: ?
}

void insertTree(Particle *p, TreeNode *t) {
    // determine the son b of t in which particle p is located
    // compute the boundary data of the subdomain of the son node and store it in t->son[b].box;
    Box sunbox;
    int b = sonNumber(&t->box, &sunbox, p);
    //std::cout << "b = " << b << std::endl;
    //std::cout << "sunbox.upper[0] = " << sunbox.upper[0] << std::endl;
    //t->son[b]->box = *sunbox;
    /*for (int d = DIM - 1; d >= 0; d--) {
        t->son[b]->box.upper[d] = sunbox.upper[d];
        t->son[b]->box.lower[d] = sunbox.lower[d];
    }*/
    if (t->son[b] == NULL) {
        if (isLeaf(t)) {
            Particle p2 = t->p;
            t->son[b] = (TreeNode*)calloc(1, sizeof(TreeNode));
            t->son[b]->p = *p;
            t->son[b]->box = sunbox;
            insertTree(&p2, t);
        } else {
            t->son[b] = (TreeNode*)calloc(1, sizeof(TreeNode));
            t->son[b]->p = *p;
            t->son[b]->box = sunbox;
        }
    } else {
        t->son[b]->box = sunbox; //?
        insertTree(p, t->son[b]);
    }
}

int sonNumber(Box *box, Box *sonbox, Particle *p) {
    int b = 0;
    for (int d = DIM - 1; d >= 0; d--) {
        if (p->x[d] < 0.5 * (box->upper[d] + box->lower[d])) {
            b = 2 * b;
            sonbox->lower[d] = box->lower[d];
            sonbox->upper[d] = .5 * (box->upper[d] + box->lower[d]);
        } else {
            b = 2 * b + 1;
            sonbox->lower[d] = .5 * (box->upper[d] + box->lower[d]);
            sonbox->upper[d] = box->upper[d];
        }
        //return b;
    }
    return b;
}

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
        current->m = 1; // SOLAR_MASS/N;
    }

    *root = (TreeNode*)calloc(1, sizeof(TreeNode));
    (*root)->p = p[1]; //(first particle with number i=1);
    (*root)->box = *domain;

    for (int i=2; i<=N; i++) //i=2
        insertTree(&p[i], *root);
}

void compPseudoParticles(TreeNode *t) {
    if (t != NULL) {
        for (int i=0; i<POWDIM; i++) {
            compPseudoParticles(t->son[i]);
        }
        if (!isLeaf(t)) {
            t->p.m = 0;
            for (int d = 0; d < DIM; d++) {
                t->p.x[d] = 0;
            }
            for (int j = 0; j < POWDIM; j++) {
                if (t->son[j] != NULL) {
                    t->p.m += t->son[j]->p.m;
                    for (int d = 0; d < DIM; d++) {
                        t->p.x[d] += t->son[j]->p.m * t->son[j]->p.x[d];
                    }
                }
            }
            for (int d = 0; d < DIM; d++) {
                t->p.x[d] = t->p.x[d] / t->p.m;
            }
        }
    }
}

void compF_BH(TreeNode *t, TreeNode *root, float diam) {
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++) {
            compF_BH(t->son[i], root, diam);
        }
        // start of the operation on *t
        if (isLeaf(t)) {
            for (int d = 0; d < DIM; d++) {
                t->p.F[d] = 0;
            }
            force_tree(t, root, diam);
        }
    }
}

void force(Particle *i, Particle *j) {
    float r = 0;
    for (int d=0; d<DIM; d++)
        r += sqrt(j->x[d] - i->x[d]);
    float f = i->m * j->m /(sqrt(r) * r);
    for (int d=0; d<DIM; d++)
        i->F[d] += f * (j->x[d] - i->x[d]);
}

void force_tree(TreeNode *tl, TreeNode *t, float diam) {
    if ((t != tl) && (t != NULL)) {
        float r = 0;
        for (int d=0; d<DIM; d++) {
            r += sqrt(t->p.x[d] - tl->p.x[d]);
        }
        r = sqrt(r);
        if ((isLeaf(t)) || (diam < theta * r)) {
            force(&tl->p, &t->p);
        } else {
            for (int i = 0; i < POWDIM; i++) {
                force_tree(t, t->son[i], .5 * diam);
            }
        }
    }
}

void updateX(Particle *p, float delta_t) {
    float a = delta_t * .5 / p->m;
    for (int d=0; d<DIM; d++) {
        p->x[d] += delta_t * (p->v[d] + a * p->F[d]); // according to (3.22)
        //p->F_old[d] = p->F[d]; ?
        p->F[d] = 0;
    }
}

void updateV(Particle *p, float delta_t) {
    float a = delta_t * .5 / p->m;
    for (int d=0; d<DIM; d++)
        p->v[d] += a * (p->F[d]); //+ p->F_old[d]); // according to (3.24)
}

void compX_BH(TreeNode *t, float delta_t) {
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++)
            compX_BH(t->son[i], delta_t);
        if (isLeaf(t)) {
            updateX(&t->p, delta_t);
        }
    }
}

void compV_BH(TreeNode *t, float delta_t) {
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++)
            compV_BH(t->son[i], delta_t);
        if (isLeaf(t)) {
            updateV(&t->p, delta_t);
        }
    }
}

void moveParticles_BH(TreeNode *root) {
    setFlags(root);
    moveLeaf(root, root);
    repairTree(root);
}

void setFlags(TreeNode *t) {
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++) {
            setFlags(t->son[i]);
        }
        t->p.moved = false;
        t->p.todelete = false;
    }
}

void moveLeaf(TreeNode *t, TreeNode *root) {
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++) {
            moveLeaf(t->son[i], root);
        }
        if ((isLeaf(t)) && (!t->p.moved)) {
            t->p.moved = true;
            if (particleWithinBox(t->p, t->box)) {
                insertTree(&t->p, root);
                t->p.todelete = true;
            }
        }
    }
}

void repairTree(TreeNode *t) {
    if (t != NULL) {
        if (!isLeaf(t)) {
            int numberofsons = 0;
            int d;
            for (int i = 0; i < POWDIM; i++) {
                if (t->son[i] != NULL) {
                    if (t->son[i]->p.todelete)
                        free(t->son[i]);
                    else {
                        numberofsons++;
                        d = i;
                    }
                }
                if (0 == numberofsons) {
                    // *t is an ‘‘empty’’ leaf node and can be deleted
                    t->p.todelete = true;
                } else if (1 == numberofsons) {
                    // *t adopts the role of its only son node and
                    // the son node is deleted directly
                    t->p = t->son[d]->p;
                    //std::cout << "t->son[d]->p.x[0] = " << t->son[d]->p.x[0] << std::endl;
                    //free(&t->son[d]->p);
                }
            }
        }
    }
}

void timeIntegration_BH(float t, float delta_t, float t_end, TreeNode *root, Box box) {
    //compF_basis(p, N);
    //compF_BH(root, root, getSystemSize(&box)); //TODO: ?
    output_particles(root);
    while (t < t_end) {
        t += delta_t;
        std::cout << "t = " << t << std::endl;
        output_particles(root);
        //std::cout << "\nroot->p.x = (" << root->p.x[0] << ", " << root->p.x[1] << ", " << root->p.x[2] << ")" << std::endl;
        compX_BH(root, delta_t);
        compF_BH(root, root, getSystemSize(&box));
        compV_BH(root, delta_t);
        repairTree(root);
    }
}

void output_particles(TreeNode *t) {
    if (t != NULL) {
        for (int i = 0; i < POWDIM; i++) {
            output_particles(t->son[i]);
        }
        if (isLeaf(t)) {
            //if (!isnan(t->p.x[0]))
                std::cout << "\tx = (" << t->p.x[0] << ", " << t->p.x[1] << ", " << t->p.x[2] << ")" << std::endl;
        }
    }
}

void freeTree_BH(TreeNode *root) {
    if (root != NULL) {
        for (int i=0; i<POWDIM; i++) {
            if (root->son[i] != NULL) {
                freeTree_BH(root->son[i]);
                free(root->son[i]);
            }
        }
    }
}


int main() {

    float systemSize = 3.0;
    TreeNode *root;
    Box box;

    for (int i=0; i<DIM; i++) {
        box.lower[i] = -systemSize;
        box.upper[i] = systemSize;
    }

    float delta_t = 0.01;
    float t_end = 1.0; //0.1;
    int N = 10; //1000;

    //inputParameters_BH(&delta_t, &t_end, &box, &theta, &N); //TODO
    initData_BH(&root, &box, N);
    timeIntegration_BH(0, delta_t, t_end, root, box);
    freeTree_BH(root);

    return 0;
}


