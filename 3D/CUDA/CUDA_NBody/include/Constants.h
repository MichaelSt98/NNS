#ifndef CONSTANTS_H_
#define CONSTANTS_H_

//#define NULL 0

typedef struct SimulationParameters
{

    bool opengl;
    bool debug;
    bool benchmark;
    bool fullscreen;
    int iterations;
    float timestep;
    float gravity;
    float dampening;

} SimulationParameters;

/// Physical constants
const double PI = 3.14159265358979323846;   //! Pi
const double TO_METERS = 1.496e11;          //! AU to meters
const double G = 6.67408e-11;               //! Gravitational constant


/// Initialization/Simulation



const int NUM_BODIES = 512*256; //512*256;//(1024*32);                   //! Number of small particles
const int NUM_SUNS = 0; //1
const double SYSTEM_SIZE = 3.5;                     //! Farthest particle (in AU)
const double SYSTEM_THICKNESS = 0.08;               //! Disk thicknes (in AU)
const double INNER_BOUND = 0.3;                     //! Closest particle to the center (in AU)
const double SOFTENING = (0.015 * TO_METERS);       //! Smooth particle interactions for small distances
const double SOLAR_MASS = 2.0e30;                   //! Solar mass (in kg)
const double EXTRA_MASS = 1.5;                      //! Disk mass as portion of center star mass
const double MAX_DISTANCE = 0.75;                   //! Barnes-Hut parameter (approximation factor)
const int TIME_STEP = (3*32*1024); //(3*3*10);                //! time step in seconds
const int STEP_COUNT = 500; //16000;                       //! amount of (simulation) steps

const bool BINARY = true;
const double BINARY_SEPARATION = 0.07; //0.07;

//#define PARALLEL_RENDER // Renders faster, but can have inaccuracies (especially when many particles occupy a small space)
#define DEBUG_INFO true // Print lots of info to the console

/// Rendering related
const int WIDTH = 1024;
const int HEIGHT = 1024;
const double RENDER_SCALE = 2.5;
const double MAX_VEL_COLOR = 40000.0;
const double MIN_VEL_COLOR = 0.0; //sqrt(0.8*(G*(SOLAR_MASS+EXTRA_MASS*SOLAR_MASS))/(SYSTEM_SIZE*TO_METERS));//14000.0;
const double PARTICLE_BRIGHTNESS = 0.35;
const double PARTICLE_SHARPNESS = 1.0;
const int DOT_SIZE = 8;
const int RENDER_INTERVAL = 1; // How many timesteps to simulate in between each frame rendered


#endif /* CONSTANTS_H_ */
