//
// Created by Michael Staneker on 15.03.21.
//

#ifndef BARNESHUTSERIAL_INTEGRATOR_H
#define BARNESHUTSERIAL_INTEGRATOR_H

#include "Constants.h"
#include "Particle.h"
#include "Domain.h"
#include "Tree.h"
#include "Renderer.h"
#include <cmath>

void timeIntegration_BH(float t, float delta_t, float t_end, TreeNode *root, Box box,
                        Renderer *renderer, char *image, double *hdImage);

#endif //BARNESHUTSERIAL_INTEGRATOR_H
