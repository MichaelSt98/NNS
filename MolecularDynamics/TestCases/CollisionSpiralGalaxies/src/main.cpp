#include <iostream>
#include <vector>
#include <random>

#define _USE_MATH_DEFINES
#include <cmath>

#include <cxxopts.hpp>
#include <highfive/H5File.hpp>

using namespace HighFive;

//const double G = 6.67408e-11;
const double G = 1.;

int main(int argc, char *argv[]) {

    /** Reading command line options **/
    cxxopts::Options options("collision-spiral-galaxies",
                             "Generating HDF5 file with initial particle distribution for the collision of two spiral galaxies.");
    options.add_options()
            ("N,N-particles", "Number of particles", cxxopts::value<int>()->default_value("1000000"))
            ("M,M-system", "Total mass distributed in the system", cxxopts::value<double>()->default_value("1."))
            ("R,R-sphere", "Radius of spheres serving as initial galaxies", cxxopts::value<double>()->default_value("1."))
            ("s,seed", "Use given random seed", cxxopts::value<unsigned int>())
            ("g,debug", "Create only a single galaxy for debugging purposes")
            ("h,help", "Show this help");

    // read and store options provided
    auto opts = options.parse(argc, argv);
    const int N { opts["N-particles"].as<int>() };
    const double M { opts["M-system"].as<double>() };
    const double R { opts["R-sphere"].as<double>() };

    // distande of galxies
    const double deltaX = 5.*R;
    const double deltaY = 5.*R;
    const double deltaV = .2;

    // store flag for mode

    // print help on usage and exit
    if (opts.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    /** Initialize random generator **/
    unsigned int seed;

    if (opts.count("seed")) {
        // use provided seed
        seed = opts["seed"].as<unsigned int>();
    } else {
        std::random_device rd; // obtain a random number from hardware
        seed = rd();
    }
    std::mt19937 gen(seed); //Standard mersenne_twister_engine seeded with rd()

    /** Generate distribution in spherical coordinates **/
    std::uniform_real_distribution<double> rndR(0., R);
    std::uniform_real_distribution<double> rndPhi(0., 2.*M_PI);
    std::uniform_real_distribution<double> rndTheta(-M_PI, M_PI);

    // containers for particle properties, each will be written to h5 file as dataset
    std::vector<std::vector<double>> x, v; // positions and velocities

    if (opts.count("debug")){
        // Loop for particle creation
        for(int i=0; i<N; i++){
            double r, phi, theta, vmag;
            r = rndR(gen);
            phi = rndPhi(gen);
            theta = rndTheta(gen);

            vmag = sqrt(G*M*r*r/(R*R*R));

            x.push_back(std::vector<double>{ r * cos(phi) * sin(theta),
                                             r * sin(phi) * sin(theta),
                                             r * cos(theta) });
            v.push_back(std::vector<double>{ vmag * sin(phi),
                                             -vmag * cos(phi), 0.} );
        }
    } else {
        // filling containers with adequate particles
        for(int i=0; i<N/2; i++){
            double r1, phi1, theta1, v1, r2, phi2, theta2, v2;
            // generate random particle in galaxy 1
            r1 = rndR(gen);
            phi1 = rndPhi(gen);
            theta1 = rndTheta(gen);

            // calculate velocity from radius with z as rotation axis and kepler velocities
            v1 = sqrt(G*M*r1*r1/(R*R*R));

            // store particle position of galaxy 1 in cartesian coordinates and move it to the upper left
            x.push_back(std::vector<double>{ r1 * cos(phi1) * sin(theta1) - deltaX/2.,
                                             r1 * sin(phi1) * sin(theta1) + deltaY/2.,
                                             r1 * cos(theta1)});
            // store particle velocity of galaxy 1 in cartesian coordinates and move it rightwards
            v.push_back(std::vector<double>{ v1 * sin(phi1) + deltaV/2.,
                                             -v1 * cos(phi1), 0.});

            // generate random particle in galaxy 2
            r2 = rndR(gen);
            phi2 = rndPhi(gen);
            theta2 = rndTheta(gen);

            // calculate velocity from radius with z as rotation axis and kepler velocities
            v2 = sqrt(G*M*r2*r2/(R*R*R));

            // store particle of galaxy 2 in cartesian coordinates and move it to the lower right
            x.push_back(std::vector<double>{ r2 * cos(phi2) * sin(theta2) + deltaX/2.,
                                             r2 * sin(phi2) * sin(theta2) - deltaY/2.,
                                             r2 * cos(theta2)});
            // store particle velocity of galaxy 2 in cartesian coordinates and move it leftwards
            v.push_back(std::vector<double>{ v2 * sin(phi2) - deltaV/2.,
                                             -v2 * cos(phi2), 0.});
        }
    }

    /** Write distribution to file **/

    // open h5 file with default property list
    File file("output/N" + std::to_string(N) + "seed" + std::to_string(seed) + ".h5",
              File::ReadWrite | File::Create | File::Truncate);

    // prepare particle mass to be written to file
    const double m = M/(double)N;

    // create data sets
    DataSet mass = file.createDataSet<double>("/m", DataSpace::From(m));
    DataSet pos = file.createDataSet<double>("/x",  DataSpace::From(x));
    DataSet vel = file.createDataSet<double>("/v",  DataSpace::From(v));

    // write data
    mass.write(m);
    pos.write(x);
    vel.write(v);

    return 0;
}
