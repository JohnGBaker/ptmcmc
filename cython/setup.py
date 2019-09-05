from distutils.core import setup,Extension
from Cython.Build import cythonize
import os
import numpy

#compiler="g++-mp-8"
compiler='/opt/local/bin/mpicxx-mpich-gcc8'
usempi=True

os.environ["CC"] = compiler
os.environ["CXX"] = compiler
os.environ['LD'] = compiler
#os.environ["CXXFLAGS"] = "-mmacosx-version-min=10.14"

comp_args=["-fopenmp","--std=c++11"]
if usempi:comp_args.append("-DUSE_MPI")

extensions = [
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
