# CUDA_CPP

CUDA (Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) model created by Nvidia. It allows software developers and software engineers to use a CUDA-enabled graphics processing unit (GPU) for general purpose processing – an approach termed GPGPU (General-Purpose computing on Graphics Processing Units). The CUDA platform is a software layer that gives direct access to the GPU’s virtual instruction set and parallel computational elements, for the execution of compute kernels.


## CUDA

### GPUs

The **graphics processing unit (GPU)** is a specialized computer processor. GPUs addresses the demands of real-time high-resolution 3D graphics compute-intensive tasks. The GPUs design is more effective than **(CPUs)** for algorithms in situations where processing large blocks of data is done in parallel.

### General-purpose computing on GPUs (GPGPU)

General-purpose computing on graphics processing units (GPGPU) is a fairly recent trend in computer engineer- ing research. GPUs are co-processors that have been heavily optimized for computer graphics processing. Computer graphics processing is a field dominated by data parallel operations—particularly linear algebra matrix operations.


### Links

* [CUDA developer zone](https://developer.nvidia.com/cuda-zone)
* [CUDA Toolkit documentation](https://docs.nvidia.com/cuda/)
* [CUDA C programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
* [Easy introduction](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) 
* [CUDA C programming guide (PDF)](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf)
* [CUDA by example (PDF)](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)
* [libcu++: The C++ Standard Library](https://nvidia.github.io/libcudacxx/)
* [C++ language support](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-cplusplus-language-support
)

### Kernels

Kernels are functions running on the device (GPU) and are executed N ties in parallel by N different CUDA threads.

```cpp
// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
	int i = threadIdx.x;
   C[i] = A[i] + B[i];
}
int main() {
...
    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(A, B, C);
	... 
}
```

### CUDA Debugging

Flags:

* no optimization flags for debugging
* `-G`
* `-lineinfo`

#### CUDA-MEMCHECK

* included in CUDA Toolkit: `bin/cuda-memcheck`
* usage: `cuda-memcheck ./executable`

See [NVIDIA: CUDA-MEMCHECK](https://docs.nvidia.com/cuda/cuda-memcheck/index.html)

#### CUDA-GDB

* included in CUDA Toolkit: `bin/gdb`
* usage: `cuda-gdb ./executable`
	* `cuda-gdb --args ./executable -arg1 -arg2` 

See [NVIDIA: CUDA-GDB](https://docs.nvidia.com/cuda/cuda-gdb/index.html)


