#define variables
objects= main.o Kernels.o ParticleSystem.o Particle.o Debug.o
NVCC= /usr/local/cuda-10.1/bin/nvcc              #cuda c compiler
CPP= g++                 #c++ compiler
opt= -O2 -g -G           #optimization flag
ARCH= -arch=sm_30        #cuda compute capability

execname= main

build: $(objects)
	$(NVCC) $(opt) -o $(execname) $(objects) $(LIBS) 


Kernels.o: Kernels.cu
	$(NVCC) $(opt) $(ARCH) -maxrregcount=32 -c Kernels.cu
ParticleSystem.o: ParticleSystem.cpp
	$(NVCC) $(opt) -std=c++11 -c ParticleSystem.cpp 
Particle.o: Particle.cu
	$(NVCC) $(opt) $(ARCH) -c Particle.cu 
Debug.o: Debug.cpp
	$(NVCC) $(opt) $(ARCH) -c Debug.cpp 
main.o: main.cpp 
	$(NVCC) $(opt) $(ARCH) -c $(INCLUDE) main.cpp


clean:
	rm $(objects)
