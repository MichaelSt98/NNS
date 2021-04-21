# Fixed-radius near neighbors

In computational geometry, the [fixed-radius near neighbor problem](https://en.wikipedia.org/wiki/Fixed-radius_near_neighbors) is a variant of the nearest neighbor search problem. In the fixed-radius near neighbor problem, one is given as input a set of points in d-dimensional Euclidean space and a fixed distance $r$. One must design a data structure that, given a query point $\vec{q}$, efficiently reports the points of the data structure that are within distance $r$ of $\vec{q}$. 

**In other words:** retriving all neighbors for an arbitrary query point $\vec{q} \in R^3$ and radius $r \in R$ with respect to a norm $||\cdot||$,
that are within the search ball $S(\vec{q}, r)$.


## Serial implementation

The overall code/project is about 

* **SPH**: short range forces in dependence of smoothing length 
* including **self-gravity**: long range force over whole domain
	* implemented via the **Barnes-Hut tree method** (as approximative method)

Therefore: **utilize the Barnes-Hut tree or rather Octree structure for fixed-radius near(est) neighbor search!**

## Parallel implementation

### Problems

* Possible neighbors or rather interaction partners could be in another process
	* therefore not accessible
* **How to determine if and which particles from other processes needed for SPH/short-range forces?**	

### (Possible) Solutions

#### Exchange particles

**Utilize the common coarse tree to determine particles to be exchanged for short range forces**

for all particles $p_i$ at $\vec{x}_i$ (with $0 \leq i < N$, whereas $N$ is the amount of particles)
and the smoothing length $h_i$:

* check whether $S(\vec{x}_i, h_i)$ is completely within the corresponding subdomain (using the common coarse tree)
	* **yes:** continue as in the serial case
	* **no:** request particle/information from overlapping processes/subdomains
		* process information (e.g. insert in local tree)
		* continue as in the serial case
		* (get rid of *additional* particles)
		

#### Halo

**Add a redundant halo to each subdomain for short range forces**

for all particles $p_i$ at $\vec{x}_i$ (with $0 \leq i < N$, whereas $N$ is the amount of particles)
and the smoothing length $h_i$ within a subdomain/process:

* determine $max(h_i)$
* adapt range (of particle keys), so that one layer of cells around each
subdomain is exchanged among neighboring subdomains, in order to have all information 
within the local subdomain/tree

> May adapt NBody algorithm to take this into account


## Summaries

* [FDPS](FDPS/FDPS.md)
* [ThreeDimensionalPointClouds](ThreeDimensionalPointClouds/ThreeDimensionalPointClouds.md)
* [LargeScaleSPH](LargeScaleSPH/LargeScaleSPH.md)
* [Miluphcuda](Miluphcuda/Miluphcuda.md)

## Links

### Papers

* [Efficient Radius Neighbor Search in Three-dimensional Point Clouds](http://jbehley.github.io/papers/behley2015icra.pdf) by J. Behley, V. Steinhage, and A.B. Cremers via Octrees
* [Fast and Efficient Nearest Neighbor Search for Particle Simulations](https://diglib.eg.org/bitstream/handle/10.2312/cgvc20191258/055-063.pdf) by J. Groß
, M. Köster and A. Krüger via unfirom grids
* [An Optimal Algorithm for Approximate Nearest
Neighbor Searching in Fixed Dimensions](https://graphics.stanford.edu/courses/cs468-06-fall/Papers/03%20AMNSW%20-%20JACM.pdf)


### Implementations

* [jbehley/octree: Efficient Radius Neighbor Search in Three-dimensional Point Clouds](https://github.com/jbehley/octree) as header only and fully templated
* [Point Cloud Library (PCL): octree module](https://pointclouds.org/documentation/group__octree.html)
	* [GitHub: PointCloudLibrary](https://github.com/PointCloudLibrary/pcl) 	

### Miscellaneous

* [FAST FIXED-RADIUS NEAREST NEIGHBORS:
INTERACTIVE MILLION-PARTICLE FLUIDS](https://on-demand.gputechconf.com/gtc/2014/presentations/S4117-fast-fixed-radius-nearest-neighbor-gpu.pdf) by Rama C. Hoetzlein, Graphics Devtech, NVIDIA
* [FDPS](https://academic.oup.com/pasj/article/68/4/54/2223184) (see [FDPS summary](FDPS/FDPS.md))

