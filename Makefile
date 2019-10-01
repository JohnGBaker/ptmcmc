#set CXX to g++
MCMC_OFILES = states.o chain.o probability_function.o proposal_distribution.o  ptmcmc.o
EIGEN=$(CURDIR)/eigen-eigen-323c052e1731/Eigen #version 3.3.7

ifeq ($(CFLAGS),)
	include Makefile.ac
else
	INCDIR = ${INCLUDE}
endif
INCDIR ?= ${CURDIR}/include
LIB ?= ${CURDIR}/lib

export LIB INCDIR CFLAGS

default:test

test: testMH

DUMMY:
	@true

#Make library if needed.
${LIB}:
	mkdir ${LIB}

${INCDIR}: 
	mkdir ${INCDIR}
	cd ${INCDIR}

${INCDIR}/Eigen: | ${INCDIR}
	@echo "Making symbolic link to Eigen library in ${INCDIR}"
	@pwd
	@cd ${INCDIR};if test -L Eigen ; then echo "exists" ; else ln -s ${EIGEN} Eigen ; fi


#Hacky way to handle subdirectories
${LIB}/libprobdist.a: ${LIB} ${INCDIR}
	@echo "LIB="${LIB}
	@echo "INCDIR="${INCDIR}
	@echo "Descending to ProbabilityDist"
	@${MAKE} CFLAGS="${CFLAGS}" CXX="${CXX}" INCDIR="${INCDIR}" -C ProbabilityDist
#@cd ProbabilityDist;${MAKE} ${MFLAGS}

chain.o: chain.cc chain.hh states.hh probability_function.hh proposal_distribution.hh ${LIB}/libprobdist.a restart.hh ${INCDIR}/Eigen
proposal_distribution.o: proposal_distribution.cc states.hh probability_function.hh proposal_distribution.hh ${LIB}/libprobdist.a ${INCDIR}/Eigen
probability_function.o: probability_function.cc states.hh probability_function.hh ${INCDIR}/newran.h
ptmcmc.o: ptmcmc.cc bayesian.hh states.hh ptmcmc.hh chain.hh options.hh probability_function.hh proposal_distribution.hh
states.o: states.hh options.hh restart.hh

${LIB}/libptmcmc.a: ${MCMC_OFILES}
	@echo "archiving"
	cp *.hh ${INCDIR}/
	ar rv ${LIB}/libptmcmc.a ${MCMC_OFILES} 

clean:
	@echo "Cleaning ptMCMC"
	rm -f *.o *.a 
	rm -f ${LIB}/*.a
	rm -f ${INCDIR}/Eigen
	@cd ProbabilityDist;${MAKE} ${MFLAGS} clean;cd -

docs:
	doxygen dox.cfg

testMH: testMH.cpp ${LIB}/libprobdist.a ${LIB}/libptmcmc.a
	${CXX} $< $(CFLAGS) -std=c++11 -o testMH  -lprobdist -lptmcmc -L${LIB} -I${INCDIR} $(LDFLAGS)

testPT: testPT.cpp ${LIB}/libprobdist.a ${LIB}/libptmcmc.a
	${CXX} $(CFLAGS) -std=c++11 -o testPT $< -lprobdist -lptmcmc -L${LIB} -I${INCDIR} $(LDFLAGS)

testGaussian: testGaussian.cc ${LIB}/libprobdist.a ${LIB}/libptmcmc.a
	${CXX} $(CFLAGS) -g -std=c++11 -o testGaussian $< -lprobdist -lptmcmc -L${LIB} -I${INCDIR} $(LDFLAGS)

example: example.cc ${LIB}/libprobdist.a ${LIB}/libptmcmc.a
	${CXX} $(CFLAGS) -std=c++11 -o example -lprobdist -lptmcmc -L${LIB} -I${INCDIR} $<

exampleLISA: exampleLISA.cc ${LIB}/libprobdist.a ${LIB}/libptmcmc.a
	${CXX}  $< $(CFLAGS) -std=c++11 -o exampleLISA -lprobdist -lptmcmc -L${LIB} -I${INCDIR}

testKL: testKL.cc ${LIB}/libprobdist.a ${LIB}/libptmcmc.a test_proposal.hh
	${CXX}  $< $(CFLAGS) -std=c++11 -o testKL -lprobdist -lptmcmc -L${LIB} -I${INCDIR}

#linear_example: linear_example.cc ${LIB}/libprobdist.a ${LIB}/libptmcmc.a
#	${CXX} $(CFLAGS) -std=c++11 -o linear_example -lprobdist -lptmcmc -L${LIB} -I${INCDIR} $<

poly_example: poly_example.cc ${LIB}/libprobdist.a ${LIB}/libptmcmc.a
	${CXX} $(CFLAGS) -std=c++11 -o poly_example -lprobdist -lptmcmc -L${LIB} -I${INCDIR} $<

.SUFFIXES: .c .cc .o

.cc.o: 
	${CXX} -c ${CFLAGS} -std=c++11 -I${INCDIR} $<

.c.o: 
	${CC} -c ${CFLAGS} $<

