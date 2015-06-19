MCMC_OFILES = mcmc.o chain.o probability_function.o proposal_distribution.o 

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

chain.o: chain.cc chain.hh mcmc.hh probability_function.hh proposal_distribution.hh
proposal_distribution.o: proposal_distribution.cc mcmc.hh probability_function.hh proposal_distribution.hh
probability_function.o: probability_function.cc mcmc.hh probability_function.hh

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

