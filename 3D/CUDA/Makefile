
#define variables

SRCDIR      := ./src
BUILDDIR    := ./build

objects= main.o Kernels.o ParticleSystem.o Particle.o Debug.o
NVCC= /usr/local/cuda-10.1/bin/nvcc              	#CUDA C compiler
CPP= g++                 	#C++ compiler
CPP_FLAGS= -O2 -g -G     	#C++ flags
ARCH= -arch=sm_30        	#CUDA compute capability

execname= bin/main

build: $(BUILDDIR)/$(objects)
	$(NVCC) $(CPP_FLAGS) -o $(execname) $(BUILDDIR)/$(objects) $(LIBS) 


$(BUILDDIR)/Kernels.o: $(SRCDIR)/Kernels.cu
	$(NVCC) $(CPP_FLAGS) $(ARCH) -maxrregcount=32 -c $(SRCDIR)/Kernels.cu
$(BUILDIR)/ParticleSystem.o: $(SRCDIR)/ParticleSystem.cpp
	$(NVCC) $(CPP_FLAGS) -std=c++11 -c $(SRCDIR)/ParticleSystem.cpp
$(BUILDDIR)/Particle.o: $(SRCDIR)/Particle.cu
	$(NVCC) $(CPP_FLAGS) $(ARCH) -c $(SRCDIR)/Particle.cu
$(BUILDDIR)/Debug.o: $(SRCDIR)/Debug.cpp
	$(NVCC) $(CPP_FLAGS) $(ARCH) -c $(SRCDIR)/Debug.cpp
$(BUILDDIR)/main.o: $(SRCDIR)/main.cpp 
	$(NVCC) $(CPP_FLAGS) $(ARCH) -c $(INCLUDE) $(SRCDIR)/main.cpp

clean:
	rm $(BUILDDIR)/$(objects)
