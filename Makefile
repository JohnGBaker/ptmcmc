MCMC_OFILES = bayesian.o chain.o probability_function.o proposal_distribution.o  ptmcmc.o

test: testMH

DUMMY:
	@true

#Make library if needed.
${LIB}:
	mkdir ${LIB}

#Hacky way to handle subdirectories
${LIB}/libprobdist.a: DUMMY ${LIB}
	@echo "Descending to ProbabilityDist"
	@cd ProbabilityDist;${MAKE} ${MFLAGS}

chain.o: chain.cc chain.hh bayesian.hh probability_function.hh proposal_distribution.hh
proposal_distribution.o: proposal_distribution.cc bayesian.hh probability_function.hh proposal_distribution.hh
probability_function.o: probability_function.cc bayesian.hh probability_function.hh
ptmcmc.o: ptmcmc.cc bayesian.hh ptmcmc.hh chain.hh options.hh

${LIB}/libptmcmc.a: ${MCMC_OFILES}
	@echo "archiving"
	ar rv ${LIB}/libptmcmc.a ${MCMC_OFILES} 

clean:
	@echo "Cleaning ptMCMC"
	rm -f *.o *.a 
	@cd ProbabilityDist;${MAKE} ${MFLAGS} clean;cd -

docs:
	doxygen dox.cfg

testMH: testMH.cpp ${LIB}/libprobdist.a ${LIB}/libptmcmc.a
	${CXX} $(CFLAGS) -o testMH -lprobdist -lptmcmc -L${LIB} $<

testPT: testPT.cpp ${LIB}/libprobdist.a ${LIB}/libptmcmc.a
	${CXX} $(CFLAGS) -o testPT -lprobdist -lptmcmc -L${LIB} $<

.SUFFIXES: .c .cc .o

.cc.o: 
	${CXX} -c ${CFLAGS} -std=c++11 $<

.c.o: 
	${CC} -c ${CFLAGS} $<

