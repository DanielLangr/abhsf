PROGRAMS += test-clang test-gnu test-intel

RM = rm -f

CXXFLAGS = -std=c++11 -O3 -DNDEBUG
#CXXFLAGS = -std=c++11 -O0 -g
#CXXFLAGS = -std=c++11 -O3 -g

CXXFLAGS += -I../utils/include
CXXFLAGS += -I$(AQSORT_ROOT)/include

.PHONY: all
all: $(PROGRAMS)

test-clang: test.cpp
	clang++ $(CXXFLAGS) -fopenmp -D_OPENMP -Wno-tautological-compare -o $@ $<

test-gnu: test.cpp
	g++ $(CXXFLAGS) -fopenmp -o $@ $<

test-intel: test.cpp
	icpc $(CXXFLAGS) -openmp -o $@ $<

.PHONY: clean
clean:
	$(RM) $(PROGRAMS)
