PROGRAMS = mb 

RM = rm -f

CXXFLAGS = -std=c++11 -O3 -DNDEBUG
CXXFLAGS += -I../include

.PHONY: all
all: $(PROGRAMS)

mb: mb.cpp
	icpc $(CXXFLAGS) -xCORE-AVX2 -qopenmp -restrict -o $@-intel_omp $<
	g++ $(CXXFLAGS) -march=haswell -fopenmp -o $@-gnu_omp $<
#	icpc -std=c++11 -O3 -mmic -qopenmp -restrict -o $@-intel_mic $<

.PHONY: clean
clean:
	$(RM) $(PROGRAMS)
