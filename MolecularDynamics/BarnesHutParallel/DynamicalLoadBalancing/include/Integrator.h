//
// Created by Michael Staneker on 15.03.21.
//

#ifndef BARNESHUTSERIAL_INTEGRATOR_H
#define BARNESHUTSERIAL_INTEGRATOR_H

#include "Constants.h"
#include "Particle.h"
#include "Domain.h"
#include "Renderer.h"
#include "Tree.h"
#include <cmath>
#include <string>
#include <sstream>
#include <iomanip>

#include <mpi.h>

#include <highfive/H5File.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5DataSet.hpp>

#include "H5Profiler.h"

void timeIntegration_BH_par(double t, double delta_t, double t_end, float diam, TreeNode *root, SubDomainKeyTree *s,
                            Renderer *renderer, char *image, double *hdImage, bool render=true,
                            bool processColoring=false, bool h5Dump=false, int h5DumpEachTimeSteps=1,
                            int loadBalancingInterval=1);

#endif //BARNESHUTSERIAL_INTEGRATOR_H
