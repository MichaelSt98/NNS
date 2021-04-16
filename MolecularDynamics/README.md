# Molecular Dynamics

This subdirectory implements algorithms/pseudocode from 

> M. Griebel, S. Knapek, and G. Zumbusch. **Numerical Simulation in Molecular Dynamics**: Numerics, Algorithms, Parallelization, Applications. 1st. Springer Pub- lishing Company, Incorporated, 2010. isbn: 3642087760

## Serial Barnes-Hut Algorithm
> [BarnesHutSerial](./BarnesHutSerial)

## Parallel Barnes-Hut Algorithm
> [BarnesHutParallel](./BarnesHutParallel)

### Basic
Static load balancing via constant domain decomposition
> [Basic](./BarnesHutParallel/Basic)

### DynamicalLoadBalancing
Dynamical load balancing via implicit Lebesgue space-filling curves
> [DynamicalLoadBalancing](./BarnesHutParallel/DynamicalLoadBalancing)

### SpaceFillingCurves
Dynamical load balancing via Hilbert space-filling curves
> [SpaceFillingCurves](./BarnesHutParallel/SpaceFillingcurves)