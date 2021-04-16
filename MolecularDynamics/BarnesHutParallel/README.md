# Barnes-Hut parallel

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
	* 8.4 **Parallel Tree Methods**
		* 8.4.1 **An Implementation with Keys** 
		* 8.4.2 **Dynamical Load Balancing** 
		* 8.4.3 **Data Distribution with Space-Filling Curves**

as an extension/**parallel version of [BarnesHutSerial](../BarnesHutSerial/)**
	
There are **3 parallel implementations** which build on each other and represent additions or rather improvements to the previous version:

* **[Basic](Basic/)** 
	* corresponding to **Parallel Tree Methods - An Implementation with Keys**
* **[DynamicalLoadBalancing](DynamicalLoadBalancing/)** 
	* corresponding to **Parallel Tree Methods - Dynamical Load Balancing**
* **[SpaceFillingCurves](SpaceFillingCurves/)** 
	* corresponding to **Parallel Tree Methods - Data Distribution with Space Filling Curves**

## Comparison of Lebuesgue and Hilbert space-filling curves

> N=100, m=1.1e-4, v=0.05, delta_t=1.0, t_end=300

Parallelized on two processes:
* Particles on process 0: *blue* dots 
* Particles on process 1: *red* dots

### Lebesgue space-filling curve
![](DynamicalLoadBalancing/N100m1_1e-4v0_05.mp4)

### Hilbert space-filling curve
![](SpacefillingCurves/N100m1_1e-4v0_05.mp4)






