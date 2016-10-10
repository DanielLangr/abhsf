PROGRAMS += 88 spmv

#CXX = icpc
CXX = g++

RM = rm -f

CXXFLAGS = -std=c++11 -O3 -DNDEBUG
#CXXFLAGS = -std=c++11 -O3
#CXXFLAGS = -std=c++11 -Og -g
#CXXFLAGS = -std=c++11 -O0 -g

CXXFLAGS += -I../include

#CXXFLAGS += -fopenmp -D_GLIBCXX_PARALLEL

LDFLAGS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm
#LDFLAGS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core 

.PHONY: all
all: $(PROGRAMS)

$(PROGRAMS): %: %.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

.PHONY: clean
clean:
	$(RM) $(PROGRAMS)
