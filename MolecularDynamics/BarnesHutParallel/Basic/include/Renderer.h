#ifndef NBODY_RENDERER_H
#define NBODY_RENDERER_H

#include <fenv.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <random>
#include "Constants.h"
#include "Logger.h"
#include "Particle.h"
#include "Domain.h"

struct color
{
    double r;
    double g;
    double b;
};

class Renderer {

private:
    int numParticles; // not const as we have to change it if particles leave domain
    const int width;
    const int height;
    const int depth;
    const double renderScale;
    const double maxVelocityColor;
    const double minVelocityColor;
    const double particleBrightness;
    const double particleSharpness;
    const int dotSize;
    const double systemSize;
    const int renderInterval;

    void renderClear(char* image, double* hdImage);
    void drawDomainBox(Box* box, double* hdImage);
    void renderBodies(Particle* p, double* hdImage);
    void renderBodies(Particle* p, int* processNum, int numprocs, double* hdImage);
    double toPixelSpace(double p, int size);
    void colorDotXY(double x, double y, double vMag, double* hdImage);
    void colorDotXZ(double x, double z, double vMag, double* hdImage);
    void colorAtXY(int x, int y, const struct color& c, double f, double* hdImage);
    void colorAtXZ(int x, int z, const struct color& c, double f, double* hdImage);
    unsigned char colorDepth(unsigned char x, unsigned char p, double f);
    double clamp(double x);
    void writeRender(char* data, double* hdImage, int step);

public:

    Renderer(const int _numParticles, const int _width, const int _height, const int _depth, const double _renderScale,
             const double _maxVelocityColor, const double _minVelocityColor, const double _particleBrightness,
             const double _particleSharpness, const int _dotSize, const double _systemSize, const int _renderInterval);

    void createFrame(char* image, double* hdImage, Particle* p, int step, Box* box);
    void createFrame(char* image, double* hdImage, Particle* p, int* processNum, int numprocs, int step, Box* box);

    int getRenderInterval();

    // Getter
    int getNumParticles() const {return numParticles;}

    // Setter
    void setNumParticles(int _numParticles) { numParticles = _numParticles; }
};


#endif //NBODY_RENDERER_H
