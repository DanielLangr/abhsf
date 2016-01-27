PROGRAMS += mtx2mm

RM = rm -f

CXXFLAGS = -std=c++11 -O2 -DNDEBUG

CXXFLAGS += -I../include

.PHONY: all
all: $(PROGRAMS)

mtx2mm: %: %.cpp
	g++ $(CXXFLAGS) -o $@ $<

.PHONY: clean
clean:
	$(RM) $(PROGRAMS)
