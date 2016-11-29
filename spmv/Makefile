#PROGRAMS += 88 88_bitmap morton spmv
#PROGRAMS += 44_bitmap 88_bitmap 
PROGRAMS += delta

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
LDFLAGS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core 

.PHONY: all
all: $(PROGRAMS)

$(PROGRAMS): %: %.cpp
	g++ $(CXXFLAGS) -O3 -march=haswell $(LDFLAGS) -o $@-gnu $<
#	g++ $(CXXFLAGS) -O3 -march=haswell $(LDFLAGS) -fno-inline -g -o $@-gnu_noinline $<
#	g++ $(CXXFLAGS) -g -O0 -march=haswell $(LDFLAGS) -o $@-gnu_debug $<
	icpc $(CXXFLAGS) -O3 -march=core-avx2 -restrict $(LDFLAGS) -o $@-intel $<

.PHONY: clean
clean:
	$(RM) $(PROGRAMS)
