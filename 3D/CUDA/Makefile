
#define variables
objects= main.o kernels.o BarnesHutParticleSystem.o Particle_cuda.o debug.o
NVCC= nvcc              	#CUDA C compiler
CPP= g++                 	#C++ compiler
CPP_FLAGS= -O2 -g -G     	#C++ flags
ARCH= -arch=sm_30        	#CUDA compute capability

execname= bin/main

build: $(objects)
	$(NVCC) $(CPP_FLAGS) -o $(execname) $(objects) $(LIBS) 


kernels.o: kernels.cu
	$(NVCC) $(CPP_FLAGS) $(ARCH) -maxrregcount=32 -c src/kernels.cu
BarnesHutParticleSystem.o: BarnesHutParticleSystem.cpp
	$(NVCC) $(CPP_FLAGS) -std=c++11 -c src/BarnesHutParticleSystem.cpp
Particle_cuda.o: Particle_cuda.cu
	$(NVCC) $(CPP_FLAGS) $(ARCH) -c src/Particle_cuda.cu
debug.o: debug.cpp
	$(NVCC) $(CPP_FLAGS) $(ARCH) -c src/debug.cpp
main.o: main.cpp 
	$(NVCC) $(CPP_FLAGS) $(ARCH) -c $(INCLUDE) src/main.cpp

clean:
	rm $(objects)