# 3D

The implementations in the subdirectories are similar, but 

* **Serial/NBody** is a serial implementation 
* **MPI/MPI_NBody** is a parallel implementation using MPI
* **OpenMP/OpenMP_NBody** is a parallel implementation using OpenMP

## Basic implementation

> Note: `.h` files are within *include/* and `.cpp` files within *src/* 

* **Body.h & Body.cpp** describing a body/particle including position, velocity, acceleration and mass
* **Constants.h** constants and initial values
* **InitializeDistribution.h & InitializeDistribution.cpp** functions to generate (different) initial body/particle distribution
* **Interaction.h & Interaction.cpp** handling interactions
* **Logger.h & Logger.cpp** logger class including *INFO*, *WARN*, *ERROR*, *DEBUG*
* **Octant.h & Octant.cpp** describing a octant including the center position and the length
* **Renderer.h & Renderer.cpp** class for creating images (`*.ppm`) from the particle positions, which can be used to generate a movie
* **Timer.h & Timer.cpp** observe time consumption of different parts of the code
* **Tree.h & Tree.cpp** recursive Oct-Tree implementation
* **Utils.h & Utils.cpp**
* **Vector3D.h** simple class for 3 dimensional vectors
* **main.cpp** **executes the code**

## Compiling

Use the `Makefile` for compiling the project and find the binary in *bin/* called *runner*.

> Note: Only tested for MacOS yet.


## Examples

![sample gif](resources/Sample.gif)

![binary sample gif](resources/BinarySample.gif)

