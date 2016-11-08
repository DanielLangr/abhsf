#PROGRAMS += 88 88_bitmap morton spmv
PROGRAMS += 88_bitmap

#CXX = icpc
#CXX = g++

RM = rm -f

#CXXFLAGS = -std=c++11 -O3 -DNDEBUG 
CXXFLAGS = -std=c++11 -O3
#CXXFLAGS = -std=c++11 -Og -g
#CXXFLAGS = -std=c++11 -O0 -g
#CXXFLAGS += -fopt-info-vec
#CXXFLAGS += -fopt-info-loop
#CXXFLAGS += -march=haswell

CXXFLAGS += -I../include

#CXXFLAGS += -fopenmp -D_GLIBCXX_PARALLEL

#LDFLAGS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm
#LDFLAGS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core 

.PHONY: all
all: $(PROGRAMS)

$(PROGRAMS): %: %.cpp
	g++ $(CXXFLAGS) -march=haswell $(LDFLAGS) -o $@-gnu $<
	icpc $(CXXFLAGS) -march=core-avx2 -restrict $(LDFLAGS) -o $@-intel $<

.PHONY: clean
clean:
	$(RM) $(PROGRAMS)
