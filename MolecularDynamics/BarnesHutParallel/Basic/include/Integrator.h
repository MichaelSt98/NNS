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

void timeIntegration_BH(float t, float delta_t, float t_end, TreeNode *root, Box box,
                        Renderer *renderer, char *image, double *hdImage);

void timeIntegration_BH_par(float t, float delta_t, float t_end, float diam, TreeNode *root, SubDomainKeyTree *s,
                            Renderer *renderer, char *image, double *hdImage, bool processColoring=false);

#endif //BARNESHUTSERIAL_INTEGRATOR_H
