from distutils.core import setup,Extension
from Cython.Build import cythonize
import os
import numpy

locale=None

#defaults
compiler=None
usempi=True

if locale=="laptop":
    #compiler="g++-mp-8"
    compiler='/opt/local/bin/mpicxx-mpich-gcc8'
    usempi=True
elif locale=="discover":
    compiler="mpicxx"
    usempi=True
elif locale is None:
    compiler=os.environ.get("CXX","")
    if "mpic" in compiler: usempi=True
    print("Generic locale: compiler:",compiler,"usempi:",usempi)

if locale is not None:
    os.environ["CC"] = compiler
    os.environ["CXX"] = compiler
    os.environ['LD'] = compiler
    
    #os.environ["CXXFLAGS"] = "-mmacosx-version-min=10.14"

comp_args=["-fopenmp","--std=c++11"]
if usempi:comp_args.append("-DUSE_MPI")

extensions = [
    Extension("test_proposal",['test_proposal.pyx','../chain.cc','../proposal_distribution.cc','../probability_function.cc','../ProbabilityDist/ProbabilityDist.cxx'],
              language='c++',
              extra_compile_args=comp_args,
              extra_link_args=comp_args,
               include_dirs=[numpy.get_include(),'..','../ProbabilityDist/','../eigen-eigen-323c052e1731/'],
              ),

    Extension("ptmcmc",['ptmcmc.pyx','../chain.cc','../proposal_distribution.cc','../probability_function.cc','../ProbabilityDist/ProbabilityDist.cxx'],
              language='c++',
              extra_compile_args=comp_args,
              extra_link_args=comp_args,
              include_dirs=[numpy.get_include(),'..','../ProbabilityDist/','../eigen-eigen-323c052e1731/'],
              #library_dirs=['../lib'],
              #libraries=['probdist','ptmcmc']
              ),    

    ]

setup(
    name="ptmcmc",
    ext_modules=cythonize(extensions, gdb_debug=True),
    )

