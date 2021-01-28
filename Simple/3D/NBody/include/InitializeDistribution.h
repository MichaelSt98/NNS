//
// Created by Michael Staneker on 27.01.21.
//

#ifndef NBODY_INITIALIZEDISTRIBUTION_H
#define NBODY_INITIALIZEDISTRIBUTION_H

#include <random>
#include "Constants.h"
#include "Body.h"
#include "Logger.h"
#include <iostream>


class InitializeDistribution {

public:
    InitializeDistribution();

    static void starParticleDisk(Body *bods);
};


#endif //NBODY_INITIALIZEDISTRIBUTION_H
