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

struct color
{
    double r;
    double g;
    double b;
};

class Renderer {

private:
    const int numParticles;
    const int width;
    const int height;
    const double renderScale;
    const double maxVelocityColor;
    const double minVelocityColor;
    const double particleBrightness;
    const double particleSharpness;
    const int dotSize;
    const double systemSize;
    const int renderInterval;

    void renderClear(char* image, double* hdImage);
    void renderBodies(Particle* p, double* hdImage);
    double toPixelSpace(double p, int size);
    void colorDot(double x, double y, double vMag, double* hdImage);
    void colorAt(int x, int y, const struct color& c, double f, double* hdImage);
    unsigned char colorDepth(unsigned char x, unsigned char p, double f);
    double clamp(double x);
    void writeRender(char* data, double* hdImage, int step);

public:

    Renderer(const int _numParticles, const int _width, const int _height, const double _renderScale, const double _maxVelocityColor,
             const double _minVelocityColor, const double _particleBrightness, const double _particleSharpness,
             const int _dotSize, const double _systemSize, const int _renderInterval);

    void createFrame(char* image, double* hdImage, Particle* p, int step);

    int getRenderInterval();

    // Getter
    int getNumParticles() const {return numParticles;}

};


#endif //NBODY_RENDERER_H
