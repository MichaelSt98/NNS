//
// Created by Michael Staneker on 25.01.21.
//

#include "../include/Renderer.h"

void Renderer::createFrame(char* image, double* hdImage, Body* b, int step)
{
    std::cout << "\nWriting frame " << step;
    if (DEBUG_INFO)	{std::cout << "\nClearing Pixels..." << std::flush;}
    renderClear(image, hdImage);
    if (DEBUG_INFO) {std::cout << "\nRendering Particles..." << std::flush;}
    renderBodies(b, hdImage);
    if (DEBUG_INFO) {std::cout << "\nWriting frame to file..." << std::flush;}
    writeRender(image, hdImage, step);
}

void Renderer::renderClear(char* image, double* hdImage)
{
    memset(image, 0, WIDTH*HEIGHT*3);
    memset(hdImage, 0, WIDTH*HEIGHT*3*sizeof(double));
}

void Renderer::renderBodies(Body* b, double* hdImage)
{
    /// ORTHOGONAL PROJECTION
#ifdef PARALLEL_RENDER
#pragma omp parallel for
#endif
    for(int index=0; index<NUM_BODIES; index++)
    {
        Body *current = &b[index];

        int x = toPixelSpace(current->position.x, WIDTH);
        int y = toPixelSpace(current->position.y, HEIGHT);

        if (x>DOT_SIZE && x<WIDTH-DOT_SIZE &&
            y>DOT_SIZE && y<HEIGHT-DOT_SIZE)
        {
            double vMag = current->velocity.magnitude(); //magnitude(current->velocity);
            colorDot(current->position.x, current->position.y, vMag, hdImage);
        }
    }
}

double Renderer::toPixelSpace(double p, int size)
{
    return (size/2.0)*(1.0+p/(SYSTEM_SIZE*RENDER_SCALE));
}

void Renderer::colorDot(double x, double y, double vMag, double* hdImage)
{
    const double velocityMax = MAX_VEL_COLOR; //35000
    const double velocityMin = sqrt(0.8*(G*(SOLAR_MASS+EXTRA_MASS*SOLAR_MASS))/
                                    (SYSTEM_SIZE*TO_METERS)); //MIN_VEL_COLOR;
    if (vMag < velocityMin)
        return;
    const double vPortion = sqrt((vMag-velocityMin) / velocityMax);
    color c;
    c.r = clamp(4*(vPortion-0.333));
    c.g = clamp(fmin(4*vPortion,4.0*(1.0-vPortion)));
    c.b = clamp(4*(0.5-vPortion));
    for (int i=-DOT_SIZE/2; i<DOT_SIZE/2; i++)
    {
        for (int j=-DOT_SIZE/2; j<DOT_SIZE/2; j++)
        {
            double xP = floor(toPixelSpace(x, WIDTH));
            double yP = floor(toPixelSpace(y, HEIGHT));
            double cFactor = PARTICLE_BRIGHTNESS /
                             (pow(exp(pow(PARTICLE_SHARPNESS*
                                          (xP+i-toPixelSpace(x, WIDTH)),2.0))
                                  + exp(pow(PARTICLE_SHARPNESS*
                                            (yP+j-toPixelSpace(y, HEIGHT)),2.0)),/*1.25*/0.75)+1.0);
            colorAt(int(xP+i),int(yP+j),c, cFactor, hdImage);
        }
    }

}

void Renderer::colorAt(int x, int y, const struct color& c, double f, double* hdImage)
{
    int pix = 3*(x+WIDTH*y);
    hdImage[pix+0] += c.r*f;//colorDepth(c.r, image[pix+0], f);
    hdImage[pix+1] += c.g*f;//colorDepth(c.g, image[pix+1], f);
    hdImage[pix+2] += c.b*f;//colorDepth(c.b, image[pix+2], f);
}

unsigned char Renderer::colorDepth(unsigned char x, unsigned char p, double f)
{
    return fmax(fmin((x*f+p),255),0);
//	unsigned char t = fmax(fmin((x*f+p),255),0);
//	return 2*t-(t*t)/255;
}

double Renderer::clamp(double x)
{
    return fmax(fmin(x,1.0),0.0);
}

void Renderer::writeRender(char* data, double* hdImage, int step)
{

    for (int i=0; i<WIDTH*HEIGHT*3; i++)
    {
        data[i] = int(255.0*clamp(hdImage[i]));
    }

    int frame = step/RENDER_INTERVAL + 1;//RENDER_INTERVAL;
    char name[128];
    sprintf(name, "images/Step%05i.ppm", frame);
    std::ofstream file (name, std::ofstream::binary);

    if (file.is_open())
    {
//		size = file.tellg();
        file << "P6\n" << WIDTH << " " << HEIGHT << "\n" << "255\n";
        file.write(data, WIDTH*HEIGHT*3);
        file.close();
    }

}
