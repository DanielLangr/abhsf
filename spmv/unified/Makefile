#PROGRAMS = spmv_haswell-gnu spmv_haswell-intel spmv_mic-intel 
PROGRAMS = spmv_cuda-gnu

RM = rm -f

CXXFLAGS += -I../../include
CXXFLAGS += -std=c++11
CXXFLAGS += -O3 -DNDEBUG

MKLFLAGSIOMP = -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread
MKLFLAGSGOMP = -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread
MKLFLAGSMIC = -L${MKLROOT}/lib/mic -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread

CUDAFLAGS = -lcuda -lcudart -lcusparse

.PHONY: all
all: $(PROGRAMS)

spmv_haswell-gnu: spmv.cpp
	g++ $(CXXFLAGS) -DHAVE_MKL -fopenmp -march=haswell $(MKLFLAGSGOMP) -o $@ $<

spmv_haswell-intel: spmv.cpp
	icpc $(CXXFLAGS) -DHAVE_MKL -qopenmp -xCORE-AVX2 -restrict $(MKLFLAGSIOMP) -o $@ $<

spmv_mic-intel: spmv.cpp
	icpc $(CXXFLAGS) -DHAVE_MKL -qopenmp -mmic -restrict $(MKLFLAGSMIC) -o $@ $<

spmv_cuda-gnu: spmv.cpp
	nvcc $(CXXFLAGS) -DHAVE_CUDA -arch=sm_35 -Xcompiler -Wno-unused-result -lcusparse -o $@ $<
#	g++ $(CXXFLAGS) -DHAVE_CUDA $(CUDAFLAGS) -Wno-unused-result -o $@ $<

.PHONY: clean
clean:
	$(RM) $(PROGRAMS)
