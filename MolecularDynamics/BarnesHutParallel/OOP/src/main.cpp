
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

int main(int argc, char** argv) {

    //setting up MPI
    boost::mpi::environment env{argc, argv};

    ConfigParser confP{ConfigParser("config/config.info")};

    SubDomain subDomainHandler;

    //settings for Logger
    LOGCFG.headers = true;
    LOGCFG.level = DEBUG;
    LOGCFG.myrank = subDomainHandler.rank;
    LOGCFG.outputRank = confP.getVal<int>("outputRank");

    char *image;
    double *hdImage;

    Renderer renderer(confP);

    const float systemSize{ confP.getVal<float>("systemSize") };
    Domain domain(systemSize);

    int numParticles = confP.getVal<int>("numParticles");

    TreeNode helperTree { domain };
    helperTree.node = TreeNode::domainList;

    ParticleDistribution particleDistribution(confP);
    ParticleList particleList;
    particleDistribution.initParticles(particleList, ParticleDistribution::disk);

    for (auto it = std::begin(particleList); it != std::end(particleList); ++it) {
        helperTree.insert(*it);
    }

    subDomainHandler.root = helperTree;
    subDomainHandler.createRanges();

    TreeNode root;



    return 0;
}
