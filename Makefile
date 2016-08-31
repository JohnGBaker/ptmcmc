#set CXX to g++
MCMC_OFILES = states.o chain.o probability_function.o proposal_distribution.o  ptmcmc.o
LIB ?= ${CURDIR}/lib
INCLUDE ?= ${CURDIR}/include
CFLAGS ?= -fopenmp
#eg CXX = /opt/local/bin/g++-mp-5

export LIB INCLUDE CFLAGS

default:test

test: testMH

DUMMY:
	@true

#Make library if needed.
${LIB}:
	mkdir ${LIB}

${INCLUDE}:
	mkdir ${INCLUDE}

#Hacky way to handle subdirectories
${LIB}/libprobdist.a: ${LIB} ${INCLUDE}
	@echo "LIB="${LIB}
	@echo "INCLUDE="${INCLUDE}
	@echo "Descending to ProbabilityDist"
	@cd ProbabilityDist;${MAKE} ${MFLAGS}

chain.o: chain.cc chain.hh states.hh probability_function.hh proposal_distribution.hh ${LIB}/libprobdist.a
proposal_distribution.o: proposal_distribution.cc states.hh probability_function.hh proposal_distribution.hh ${LIB}/libprobdist.a
probability_function.o: probability_function.cc states.hh probability_function.hh ${INCLUDE}/newran.h
ptmcmc.o: ptmcmc.cc bayesian.hh states.hh ptmcmc.hh chain.hh options.hh probability_function.hh proposal_distribution.hh
states.o: states.hh options.hh

${LIB}/libptmcmc.a: ${MCMC_OFILES}
	@echo "archiving"
	cp *.hh ${INCLUDE}/
	ar rv ${LIB}/libptmcmc.a ${MCMC_OFILES} 

clean:
	@echo "Cleaning ptMCMC"
	rm -f *.o *.a 
	rm -f ${LIB}/*.a
	@cd ProbabilityDist;${MAKE} ${MFLAGS} clean;cd -

docs:
	doxygen dox.cfg

testMH: testMH.cpp ${LIB}/libprobdist.a ${LIB}/libptmcmc.a
	${CXX} $(CFLAGS) -std=c++11 -o testMH $< -lprobdist -lptmcmc -L${LIB} -I${INCLUDE} $(LDFLAGS)

testPT: testPT.cpp ${LIB}/libprobdist.a ${LIB}/libptmcmc.a
	${CXX} $(CFLAGS) -std=c++11 -o testPT $< -lprobdist -lptmcmc -L${LIB} -I${INCLUDE} $(LDFLAGS)

example: example.cc testPT.cpp ${LIB}/libprobdist.a ${LIB}/libptmcmc.a
	${CXX} $(CFLAGS) -std=c++11 -o example -lprobdist -lptmcmc -L${LIB} -I${INCLUDE} $<

#linear_example: linear_example.cc ${LIB}/libprobdist.a ${LIB}/libptmcmc.a
#	${CXX} $(CFLAGS) -std=c++11 -o linear_example -lprobdist -lptmcmc -L${LIB} -I${INCLUDE} $<

poly_example: poly_example.cc ${LIB}/libprobdist.a ${LIB}/libptmcmc.a
	${CXX} $(CFLAGS) -std=c++11 -o poly_example -lprobdist -lptmcmc -L${LIB} -I${INCLUDE} $<

.SUFFIXES: .c .cc .o

.cc.o: 
	${CXX} -c ${CFLAGS} -std=c++11 -I${INCLUDE} $<

.c.o: 
	${CC} -c ${CFLAGS} $<

