# Basic

This subdirectory implements the **parallel Barnes-Hut method/algorithm** from 

> M. Griebel, S. Knapek, and G. Zumbusch. **Numerical Simulation in Molecular Dynamics**: Numerics, Algorithms, Parallelization, Applications. 1st. Springer Pub- lishing Company, Incorporated, 2010. isbn: 3642087760

described in the sections:

* 8 Tree Algorithms for Long-Range Potentials 
	* 8.1 Series Expansion of the Potential 
	* 8.2 Tree Structures for the Decomposition of the Far Field 
	* 8.3 Particle-Cluster Interactions and the Barnes-Hut Method 
		* 8.3.1 Method 
		* 8.3.2 Implementation
		* 8.3.3 Applications from Astrophysics
	* 8.4 **ParallelTreeMethods**
		* 8.4.1 **An Implementation with Keys** 
		* 8.4.2 Dynamical Load Balancing
		* 8.4.3 Data Distribution with Space-Filling Curves

as an extension/**parallel version of [BarnesHutSerial](../BarnesHutSerial/)**
	
## Usage

* **Compilation** via `make`
* **Running** via `mpirun -np <number of processes> ./bin/runner`
* **Cleaning** via
	* `make clean`
	* `make cleaner`
	
## TODO

* Found 17 TODO items in 5 files
* BarnesHutParallel
* Basic
	* include
		* Integrator.h
			* (14, 3) //TODO: Box *box or Box box ?
	* src
		* ConfigParser.cpp
			* (15, 11) //TODO: Throw exception here
		* Integrator.cpp
			* (7, 3) //TODO: adapt to parallel method (use SubDomainKeyTree)
		* main.cpp
			* (20, 4) // TODO: providing parallel program with initial data:
			* (116, 20) s.range = 0; //TODO: set range for sub domain key tree
			* (118, 7) //TODO: needed to be called by every process?
		* Tree.cpp
			* (16, 3) //TODO: create key (!?)
			* (62, 3) //TODO: do not insert particle data within domainList nodes, instead:
			* (302, 3) //TODO: implement sendParticles (Sending Particles to Their Owners and Inserting Them in the Local Tree)
			* (320, 3) //TODO: implement buildSendlist (Sending Particles to Their Owners and Inserting Them in the Local Tree)
			* (342, 3) //TODO: implement compPseudoParticlespar (Parallel Computation of the Values of the Pseudoparticles)
			* (349, 3) //TODO: implement compLocalPseudoParticlespar (Parallel Computation of the Values of the Pseudoparticles)
			* (364, 3) //TODO: implement compDomainListPsudoParticlespar (Parallel Computation of the Values of the Pseudoparticles)
			* (402, 3) //TODO: implement symbolicForce (Determining Subtrees that are Needed in the Parallel Force Computation)
			* (426, 3) //TODO: implement compF_BHpar (Parallel Force Computation)
			* (441, 3) //TODO: implement compTheta (Parallel Force Computation)
			* (449, 21) if ((true/* TODO: *t is a domainList node*/) && ((proc = key2proc(key(*t), s)) != s->myrank)) {
