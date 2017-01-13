PROGRAMS += coo csr

#CXX = icpc
#CXX = g++

RM = rm -f

#CXXFLAGS = -std=c++11 -O3 -DNDEBUG 
CXXFLAGS = -std=c++11 
#CXXFLAGS = -std=c++11 -Og -g
#CXXFLAGS = -std=c++11 -O0 -g
#CXXFLAGS += -fopt-info-vec
#CXXFLAGS += -fopt-info-loop
#CXXFLAGS += -march=haswell

CXXFLAGS += -I../../include

#CXXFLAGS += -fopenmp -D_GLIBCXX_PARALLEL

LDFLAGS = -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core 
LDFLAGSOMP = -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread
LDFLAGSMIC = -L${MKLROOT}/lib/mic -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread

.PHONY: all
all: $(PROGRAMS)

coo: coo.cpp
	icpc $(CXXFLAGS) -O3 -xCORE-AVX2 -restrict -wd3180 $(LDFLAGS) -o $@-intel $<
	icpc $(CXXFLAGS) -O3 -xCORE-AVX2 -qopenmp -restrict $(LDFLAGSOMP) -o $@-intel_omp $<

csr: csr.cpp
	icpc $(CXXFLAGS) -O3 -xCORE-AVX2 -restrict -wd3180 $(LDFLAGS) -o $@-intel $<
	icpc $(CXXFLAGS) -O3 -xCORE-AVX2 -qopenmp -restrict $(LDFLAGSOMP) -o $@-intel_omp $<
#	icpc $(CXXFLAGS) -O3 -mmic -qopenmp -restrict $(LDFLAGSMIC) -o $@-intel_mic $<

.PHONY: clean
clean:
	$(RM) $(PROGRAMS)
