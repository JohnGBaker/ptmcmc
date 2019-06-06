CFLAGS = -fopenmp -O2
CXX = g++

socket_sampler: socket_sampler.cc
	${CXX} $(CFLAGS) -std=c++11 -o $@ $< -lprobdist -lptmcmc -L../../lib -I../../include $(LDFLAGS)
