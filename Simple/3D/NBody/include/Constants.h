#ifndef CONSTANTS_H_
#define CONSTANTS_H_

//#define NULL 0
/// Physical constants
// PI      3.14159265358979323846
//#define TO_METERS 1.496e11 // Meters in an AU
//#define G 6.67408e-11 // The gravitational constant
const double PI = 3.14159265358979323846;
const double TO_METERS = 1.496e11;
const double G = 6.67408e-11;


/// Initialization/Simulation
//#define NUM_BODIES (1024*32) // Number of small particles
//#define SYSTEM_SIZE 3.5    // Farthest particles in AU
//#define SYSTEM_THICKNESS 0.08  //  Thickness in AU
//#define INNER_BOUND 0.3    // Closest particles to center in AU
//#define SOFTENING (0.015*TO_METERS) // Softens particles interactions at close distances
//#define SOLAR_MASS 2.0e30  // in kg
//#define BINARY_SEPARATION 0.07 // AU (only applies when binary code uncommented)
//#define EXTRA_MASS 1.5 // 0.02 Disk mask as a portion of center star/black hole mass
#define ENABLE_FRICTION 0 // For experimentation only. Will probably cause weird results
//#define FRICTION_FACTOR 25.0 // Only applies if friction is enabled
//#define MAX_DISTANCE 0.75 //2.0  Barnes-Hut Distance approximation factor
//#define TIME_STEP (3*32*1024) //(1*128*1024) Simulated time between integration steps, in seconds
//#define STEP_COUNT 16000 // Will automatically stop running after this many steps
const int NUM_BODIES = (1024*32);
const double SYSTEM_SIZE = 3.5;
const double SYSTEM_THICKNESS = 0.08;
const double INNER_BOUND = 0.3;
const double SOFTENING = (0.015 * TO_METERS);
const double SOLAR_MASS = 2.0e30;
const double EXTRA_MASS = 1.5;
const double FRICTION_FACTOR = 25;
const double MAX_DISTANCE = 0.75;
const int TIME_STEP = (3*32*1024);
const int STEP_COUNT = 16000;


//#define PARALLEL_RENDER // Renders faster, but can have inaccuracies (especially when many particles occupy a small space)
#define DEBUG_INFO true // Print lots of info to the console

/// Rendering related
const int WIDTH = 1024;
const int HEIGHT = 1024;
const double RENDER_SCALE = 2.5;
const double MAX_VEL_COLOR = 40000.0;
const double MIN_VEL_COLOR = sqrt(0.8*(G*(SOLAR_MASS+EXTRA_MASS*SOLAR_MASS))/(SYSTEM_SIZE*TO_METERS));//14000.0;
const double PARTICLE_BRIGHTNESS = 0.35;
const double PARTICLE_SHARPNESS = 1.0;
const int DOT_SIZE = 8;
const int RENDER_INTERVAL = 1; // How many timesteps to simulate in between each frame rendered


#endif /* CONSTANTS_H_ */
