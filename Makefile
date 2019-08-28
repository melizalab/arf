
PREFIX=/usr/local
CXX ?= g++
CFLAGS = -g -Wall -Ic++ -I/opt/local/include `pkg-config --cflags hdf5`
LDFLAGS = -L/opt/local/lib `pkg-config --libs-only-L hdf5`

all: test

%.o : %.cpp
	$(CXX) $(CFLAGS) -o $@ -c $<

test:	tests/test_arf.o
	$(CXX) $(LDFLAGS) -o tests/test_arf tests/test_arf.o -lhdf5 -lhdf5_hl

clean:
	rm -f test.arf tests/*.o tests/test_arf tests/test.arf

install:
	install -d $(PREFIX)/include/arf
	install -m 644 -o root c++/*.hpp $(PREFIX)/include/
	install -m 644 -o root c++/arf/*.hpp $(PREFIX)/include/arf/

