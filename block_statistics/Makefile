PROGRAMS += block_stats msmf test

CXX = g++

RM = rm -f

#CXXFLAGS = -std=c++11 -O3 -DNDEBUG
#CXXFLAGS = -std=c++11 -O3
CXXFLAGS = -std=c++11 -O0 -g

CXXFLAGS += -I../include

.PHONY: all
all: $(PROGRAMS)

$(PROGRAMS): %: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

.PHONY: clean
clean:
	$(RM) $(PROGRAMS)
