//
// Created by Michael Staneker on 25.01.21.
//

#ifndef NBODY_RENDERER_H
#define NBODY_RENDERER_H

#include <fenv.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <random>
#include "Constants.h"
#include "Body.h"
#include "Vector3D.h"

struct color
{
    double r;
    double g;
    double b;
};

class Renderer {

public:
    void createFrame(char* image, double* hdImage, Body* b, int step);
    void renderClear(char* image, double* hdImage);
    void renderBodies(Body* b, double* hdImage);
    double toPixelSpace(double p, int size);
    void colorDot(double x, double y, double vMag, double* hdImage);
    void colorAt(int x, int y, const struct color& c, double f, double* hdImage);
    unsigned char colorDepth(unsigned char x, unsigned char p, double f);
    double clamp(double x);
    void writeRender(char* data, double* hdImage, int step);
};


#endif //NBODY_RENDERER_H
