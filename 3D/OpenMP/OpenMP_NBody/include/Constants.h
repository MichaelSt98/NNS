#ifndef CONSTANTS_H_
#define CONSTANTS_H_

//#define NULL 0

/// Physical constants
const double PI = 3.14159265358979323846;   //! Pi
const double TO_METERS = 1.496e11;          //! AU to meters
const double G = 6.67408e-11;               //! Gravitational constant


/// Initialization/Simulation
// TODO: Use a config struct with const qualifier to access global variables read in from config file

extern int NUM_BODIES;                    //! Number of small particles
extern int NUM_SUNS;
extern double SYSTEM_SIZE;                //! Farthest particle (in AU)
extern double SYSTEM_THICKNESS;           //! Disk thicknes (in AU)
extern double INNER_BOUND;                //! Closest particle to the center (in AU)
extern double SOFTENING;                  //! Smooth particle interactions for small distances
extern double SOLAR_MASS;                 //! Solar mass (in kg)
extern double EXTRA_MASS;                 //! Disk mass as portion of center star mass
extern double MAX_DISTANCE;               //! Barnes-Hut parameter (approximation factor)
extern int TIME_STEP;                     //! time step in seconds
extern int STEP_COUNT;                    //! amount of (simulation) steps

extern bool BINARY;
extern double BINARY_SEPARATION;

//#define PARALLEL_RENDER // Renders faster, but can have inaccuracies (especially when many particles occupy a small space)
#define DEBUG_INFO true // Print lots of info to the console

/*
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
*/

#endif /* CONSTANTS_H_ */
