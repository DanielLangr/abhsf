#PROGRAMS += 88 88_bitmap morton spmv
#PROGRAMS += 44_bitmap 88_bitmap 
PROGRAMS += delta mb

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

CXXFLAGS += -I../include

#CXXFLAGS += -fopenmp -D_GLIBCXX_PARALLEL

#LDFLAGS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm
LDFLAGS = -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core 
LDFLAGSOMP = -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread
LDFLAGSMIC = -L${MKLROOT}/lib/mic -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread

.PHONY: all
all: $(PROGRAMS)

delta: delta.cpp
	g++ $(CXXFLAGS) -O3 -march=haswell $(LDFLAGS) -o $@-gnu $<
	g++ $(CXXFLAGS) -O3 -march=haswell $(LDFLAGS) -fno-inline -g -o $@-gnu_noinline $<
#	g++ $(CXXFLAGS) -g -O0 -march=haswell $(LDFLAGS) -o $@-gnu_debug $<
	icpc $(CXXFLAGS) -O3 -xCORE-AVX2 -restrict -wd3180 $(LDFLAGS) -o $@-intel $<
	icpc $(CXXFLAGS) -O3 -xCORE-AVX2 -restrict -wd3180 -fno-inline-functions $(LDFLAGS) -o $@-intel_noinline $<
	icpc $(CXXFLAGS) -O3 -xCORE-AVX2 -qopenmp -restrict $(LDFLAGSOMP) -o $@-intel_omp $<
	icpc $(CXXFLAGS) -O3 -mmic -qopenmp -restrict $(LDFLAGSMIC) -o $@-intel_mic $<

mb: mb.cpp
	icpc -std=c++11 -O3 -xCORE-AVX2 -qopenmp -restrict -o $@-intel_omp $<
	icpc -std=c++11 -O3 -mmic -qopenmp -restrict -o $@-intel_mic $<

.PHONY: clean
clean:
	$(RM) $(PROGRAMS)
