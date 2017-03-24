PROGRAMS = spmv_haswell-gnu 
PROGRAMS += spmv_haswell-intel

RM = rm -f

CXXFLAGS = -I../../include
CXXFLAGS += -std=c++11
CXXFLAGS += -O3 -DNDEBUG

CXXFLAGS += -DHAVE_PAPI -lpapi -lpthread
CXXFLAGS += -DHAVE_MKL

MKLFLAGSIOMP = -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread
MKLFLAGSGOMP = -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread

.PHONY: all
all: $(PROGRAMS)

spmv_haswell-gnu: spmv.cpp
	g++ $(CXXFLAGS) -fopenmp -march=haswell $(MKLFLAGSGOMP) -o $@ $<

spmv_haswell-intel: spmv.cpp
	icpc $(CXXFLAGS) -qopenmp -xCORE-AVX2 -restrict $(MKLFLAGSIOMP) -o $@ $<

.PHONY: clean
clean:
	$(RM) $(PROGRAMS)
