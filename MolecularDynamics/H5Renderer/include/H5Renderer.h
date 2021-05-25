#ifndef NBODY_H5RENDERER_H
#define NBODY_H5RENDERER_H

#include <iostream>
#include <vector>
#include <cmath>
#include <highfive/H5File.hpp>
#include <boost/filesystem.hpp>

#include <fstream>

#include "Logger.h"

namespace fs = boost::filesystem;

struct ColorRGB {
    char r;
    char g;
    char b;

    ColorRGB() : r { 0 }, g { 0 }, b { 0 }{};
    ColorRGB(char _r, char _g, char _b) : r { _r }, g { _g }, b { _b }{};
};

const ColorRGB COLORS[10] = {
        ColorRGB(~0, 0, 0), // red
        ColorRGB(~0, ~0, 0), // yellow
        ColorRGB(0, 0, ~0), // blue
        ColorRGB(~0, 0, ~0), // magenta
        ColorRGB(0, ~0, 0), // green
        ColorRGB(~0, 0, 127), // pink
        ColorRGB(~0, 127, 0), //orange
        ColorRGB(0, ~0, ~0), // turquoise
        ColorRGB(127, 127, 127), // grey
        ColorRGB(~0, ~0, ~0) //white
};

class H5Renderer {

private:
    // initializing variables for H5Renderer
    // input variables
    const std::string h5folder;
    const double systemSize;
    const int imgHeight;
    const bool processColoring; // default: true

    // constants
    const double SCALE2FIT = .9;

    // internal variables
    std::vector<fs::path> h5files;
    long psSize; // pixelSpace size
    ColorRGB* pixelSpace; // this is the container for the image, which is manipulated by private functions

    //functions
    ColorRGB procColor(unsigned long k, const std::vector<unsigned long> &ranges);
    void clearPixelSpace();
    int pos2pixel(double pos);
    void particle2Pixel(double x, double y, double z, const ColorRGB &color);
    void pixelSpace2File(const std::string &outFile);

public:
    H5Renderer(std::string _h5folder, double _systemSize, int _imgHeight, bool _processColoring=true);
    ~H5Renderer();

    void createImages(std::string outDir);
};


#endif //NBODY_H5RENDERER_H
