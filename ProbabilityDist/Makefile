PROBDIST_OFILES = ProbabilityDist.o
NEWRAN_OFILES   = newran1.o newran2.o myexcept.o simpstr.o extreal.o
ifeq ($(CFLAGS),)
	include ../Makefile.ac
	INCDIR = .
	#CFLAGS = -g
	#CXX = /opt/local/bin/g++-mp-4.7
endif


all: ${LIB}/libprobdist.a  ${INCDIR}/ProbabilityDist.h
	@echo LIB=${LIB}
	@echo INCDIR=${INCDIR}

headers: ProbabilityDist.h

${LIB}/libprobdist.a: ${PROBDIST_OFILES} ${NEWRAN_OFILES} ${INCDIR}/ProbabilityDist.h
	ar rv ${LIB}/libprobdist.a ${PROBDIST_OFILES} ${NEWRAN_OFILES}

${INCDIR}/ProbabilityDist.h: ProbabilityDist.h include.h newran.h myexcept.h extreal.h simpstr.h
	@echo "Installing headers in "${INCDIR}
	cp ProbabilityDist.h include.h newran.h myexcept.h extreal.h simpstr.h ${INCDIR}/



testProbDist: ${NEWRAN_OFILES} testProbDist.cxx ProbabilityDist.o
	${CXX} ${CFLAGS} -o testProbDist testProbDist.cxx ProbabilityDist.o ${NEWRAN_OFILES}

clean:
	@echo "Cleaning ProbabilityDist"
	rm -f *.o *.a 

ProbabilityDist.o: ProbabilityDist.cxx ProbabilityDist.h
newran1.o: newran1.cxx include.h newran.h myexcept.h extreal.h simpstr.h
newran2.o: newran2.cxx include.h newran.h myexcept.h extreal.h simpstr.h
simpstr.o: simpstr.cxx include.h myexcept.h simpstr.h
extreal.o: extreal.cxx include.h extreal.h


.SUFFIXES: .c .cxx .o

.cxx.o: 
	${CXX} -c ${CFLAGS} $<

.c.o: 
	${CC} -c ${CFLAGS} $<

