#include "../include/ParticleDistribution.h"


ParticleDistribution::ParticleDistribution(int numParticles, float systemSize, float initVelocity, float mass)
    : numParticles { numParticles }, systemSize { systemSize }, initVelocity { initVelocity }, mass { mass } {

}

ParticleDistribution::ParticleDistribution(ConfigParser &confP) : ParticleDistribution(
        confP.getVal<int>("numParticles"),
        confP.getVal<float>("systemSize"),
        confP.getVal<float>("initVel"),
        confP.getVal<float>("initMass")) {

}

void ParticleDistribution::initParticles(ParticleList &pList, type distributionType) {
    if (distributionType == disk) {
        initDisk(pList);
    }
    else {
        Logger(ERROR) << "Particle distribution type: " << distributionType << "not available!";
    }
}

void ParticleDistribution::initDisk(ParticleList &pList) {

    //Logger(INFO) << "Initializing " << numParticles << " particles!";

    using std::uniform_real_distribution;

    uniform_real_distribution<float> randAngle (0.f, 200.f * PI);
    uniform_real_distribution<float> randRadius (0.f, systemSize/2.f);
    uniform_real_distribution<float> randHeight (0.f, systemSize/10.f);

    std::random_device rd;
    //std::default_random_engine gen (rd());

    unsigned int seed = 2568239274 + comm.rank() * 1000;
    std::default_random_engine gen (seed);

    float angle;
    float radius;
    float radiusOffset;
    float height;

    for (int i=0; i<numParticles; i++) {
        angle = randAngle(gen);
        radiusOffset = randRadius(gen);
        radius = sqrt(systemSize - radiusOffset);
        height = randHeight(gen);

        std::uniform_real_distribution<float> dist(-systemSize, systemSize);

        pList.push_back(Particle {
            { radius * cos(angle), radius * sin(angle), height },
            { initVelocity * sin(angle), -initVelocity*cos(angle), dist(gen)/75.f * initVelocity },
            mass
        });
    }

}