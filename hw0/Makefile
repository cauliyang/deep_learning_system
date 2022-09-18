# NOTE: on MacOS you need to add an addition flag: -undefined dynamic_lookup
default: src/simple_ml_ext.cpp
	c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup -fPIC $$(python -m pybind11 --includes) src/simple_ml_ext.cpp -o src/simple_ml_ext.so

test: src/test_simple_ml_ext.cpp src/simple_ml_ext.hpp
	$(CXX) -o test src/test_simple_ml_ext.cpp -I/usr/local/include $$(python -m pybind11 --includes)
	./test

clean:
	rm -f src/simple_ml_ext.so
	rm -f test

all: clean default test
