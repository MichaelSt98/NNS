
#include "../include/Logger.h"
//#include "../include/Keytype.h"
#include "../include/ConfigParser.h"
#include "../include/Particle.h"
#include "../include/Tree.h"
#include "../include/Domain.h"
#include "../include/SubDomain.h"
#include "../include/Renderer.h"
#include "../include/ParticleDistribution.h"

#include <iostream>
#include <climits>
#include <boost/mpi.hpp>

#define KEY_MAX ULONG_MAX

structlog LOGCFG = {};

void timeIntegration(float t, float deltaT, float tEnd, float diam, SubDomain &subDomain,
                     Renderer &renderer, char *image, double *hdImage, bool render=true, bool processColoring=false);

void timeIntegration(float t, float deltaT, float tEnd, float diam, SubDomain &subDomain,
                     Renderer &renderer, char *image, double *hdImage, bool render, bool processColoring) {
    int step = 0;

    while (t < tEnd) {

        Logger(DEBUG) << " ";
        Logger(DEBUG) << "t = " << t;
        Logger(DEBUG) << "--------------------------";

        // rendering
        if (render && step % renderer.getRenderInterval() == 0)
        {
            Particle *prtcls;
            int *prtN;

            ParticleList pList;
            IntList procList;
            if (processColoring) {
                subDomain.gatherParticles(pList, procList);
                prtN = new int[(int)procList.size()];
                //prtN = &procList[0];
                std::copy(procList.begin(), procList.end(), prtN);
            }
            else {
                subDomain.gatherParticles(pList);
            }
            prtcls = new Particle[(int)pList.size()];
            //prtcls = &pList[0];
            std::copy(pList.begin(), pList.end(), prtcls);

            if (subDomain.rank == 0) {
                Logger(INFO) << "Rendering timestep #" << step << ": N = " << pList.size();
                renderer.setNumParticles((int)pList.size());
                if (processColoring) {
                    renderer.createFrame(image, hdImage, prtcls, prtN, subDomain.numProcesses, step, &subDomain.root.box);
                    delete [] prtN;
                }
                else {
                    renderer.createFrame(image, hdImage, prtcls, step, &subDomain.root.box);
                }
                delete [] prtcls;
            }
            //output_tree(root, false);
        }
        ++step;

        t += deltaT; // update timestep

        subDomain.compFParallel(diam);

        subDomain.root.repairTree();

        subDomain.root.compX(deltaT);

        subDomain.root.compV(deltaT);

        subDomain.moveParticles();

        subDomain.sendParticles();

        subDomain.compPseudoParticles();

        subDomain.root.printTreeSummary(false);

    }
    Logger(DEBUG) << "t = " << t << ", FINISHED";
}

int main(int argc, char** argv) {

    //setting up MPI
    boost::mpi::environment env{argc, argv};
    boost::mpi::communicator comm;

    ConfigParser confP{ConfigParser("config/config.info")};

    SubDomain subDomainHandler;

    //settings for Logger
    LOGCFG.headers = true;
    LOGCFG.level = DEBUG;
    LOGCFG.myrank = subDomainHandler.rank;
    LOGCFG.outputRank = confP.getVal<int>("outputRank");

    char *image;
    double *hdImage;
    int width = confP.getVal<int>("width");
    int height = confP.getVal<int>("height");
    image = new char[2 * width * height * 3];
    hdImage = new double[2 * width * height * 3];

    Renderer renderer(confP);

    const float systemSize{ confP.getVal<float>("systemSize") };
    Domain domain(systemSize);

    int numParticles = confP.getVal<int>("numParticles");

    /** create ranges */
    Particle rootParticle {{0, 0, 0} };
    TreeNode helperTree { rootParticle , domain };
    subDomainHandler.root = helperTree;
    subDomainHandler.root.node = TreeNode::domainList;

    ParticleDistribution particleDistribution(confP);
    ParticleList particleList;
    particleDistribution.initParticles(particleList, ParticleDistribution::disk);


    for (auto it = std::begin(particleList); it != std::end(particleList); ++it) {
        subDomainHandler.root.insert(*it);
    }

    subDomainHandler.createRanges();
    /** END: create ranges */


    TreeNode root { domain };
    subDomainHandler.root = root;

    subDomainHandler.createDomainList();

    for (auto it = std::begin(particleList); it != std::end(particleList); ++it) {
        subDomainHandler.root.insert(*it);
    }

    //subDomainHandler.root.printTreeSummary(true);

    float diam = subDomainHandler.root.box.upper[0] - subDomainHandler.root.box.lower[0];
    float deltaT = confP.getVal<float>("timeStep");
    float tEnd = confP.getVal<float>("timeEnd");

    bool render = confP.getVal<bool>("render");
    bool processColoring = confP.getVal<bool>("processColoring");

    timeIntegration(0.f, deltaT, tEnd, diam, subDomainHandler, renderer, image, hdImage, render, processColoring);

    return 0;
}
