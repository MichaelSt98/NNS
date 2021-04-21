# Miluphcuda

**Single GPU code using NVIDA CUDA**

## Neighbor search

Implemented in `tree.h/cu` 

* [miluphcuda/tree.cu](https://github.com/christophmschaefer/miluphcuda/blob/main/tree.cu)
* [miluphcuda/tree.h](https://github.com/christophmschaefer/miluphcuda/blob/main/tree.h)

Functions:

* `__global__ void nearNeighbourSearch(int *interactions)`
	* search interaction partners for each particle 
* `__global__ void nearNeighbourSearch_modify_sml(int *interactions)`
	* search interaction partners for each particle, but the smoothing length is changed if `MAX_NUM_INTERACTIONS` is reached 	
* `__global__ void knnNeighbourSearch(int *interactions)`
	* search interaction partners with variable smoothing length
* `__global__ void symmetrizeInteractions(int *interactions)`
	* checks interaction list for symmetry
		* e.g.: removes particle j from particle i's interaction list if particle i is not in particles j's interaction list  
* `__device__ void redo_NeighbourSearch(int particle_id, int *interactions)`
	* redo NeighbourSearch for particular particle only: search for interaction partners 


> Note: removed everything for $DIM \neq 3$

```cpp
/* search interaction partners for each particle */
__global__ void nearNeighbourSearch(int *interactions)
{
	register int i, inc, nodeIndex, depth, childNumber, child;
	register double x, interactionDistance, dx, r, d;
    register double y, dy;
	register int currentNodeIndex[MAXDEPTH];
	register int currentChildNumber[MAXDEPTH];
	register int numberOfInteractions;
	register double z, dz;
	
	inc = blockDim.x * gridDim.x;
	for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i += inc) {
		x = p.x[i];
		y = p.y[i];
		z = p.z[i];

	    double sml; /* smoothing length of particle */
        double smlj; /* smoothing length of potential interaction partner */
		// start at root
		depth = 0;
		currentNodeIndex[depth] = numNodes - 1;
		currentChildNumber[depth] = 0;
		numberOfInteractions = 0;
		r = radius * 0.5; // because we start with root children
        sml = p.h[i];
        p.noi[i] = 0;
		interactionDistance = (r + sml);

		do { //while (depth >= 0)
		
			childNumber = currentChildNumber[depth];
			nodeIndex = currentNodeIndex[depth];

			while (childNumber < numChildren) {
				child = childList[childListIndex(nodeIndex, childNumber)];
				childNumber++;
				if (child != EMPTY && child != i) {
					dx = x - p.x[child];
					dy = y - p.y[child];
					dz = z - p.z[child];

					if (child < numParticles) {
                      
						d = dx*dx;
                       	d += dy*dy;
						d += dz*dz;

                       smlj = p.h[child];

						if (d < sml*sml && d < smlj*smlj) {
							interactions[i * MAX_NUM_INTERACTIONS + numberOfInteractions] = child;
							numberOfInteractions++;
						}
					} else if (fabs(dx) < interactionDistance && fabs(dy) < interactionDistance && fabs(dz) < interactionDistance) {
						// put child on stack
						currentChildNumber[depth] = childNumber;
						currentNodeIndex[depth] = nodeIndex;
						depth++;
						r *= 0.5;
						interactionDistance = (r + sml);
						if (depth >= MAXDEPTH) {
							printf("Error, maxdepth reached!");
                           	assert(depth < MAXDEPTH);
						}
						childNumber = 0;
						nodeIndex = child;
					}
				}
			}

			depth--;
			r *= 2.0;
			interactionDistance = (r + sml);
		} while (depth >= 0);
	}
}
```
