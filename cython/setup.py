from distutils.core import setup,Extension
from Cython.Build import cythonize
import os
import numpy

os.environ["CC"] = "g++-mp-8"
os.environ["CXX"] = "g++-mp-8"
#os.environ["CXXFLAGS"] = "-mmacosx-version-min=10.14"

extensions = [
    Extension("ptmcmc",['ptmcmc.pyx','../chain.cc','../proposal_distribution.cc','../probability_function.cc','../ProbabilityDist/ProbabilityDist.cxx'],
              language='c++',
              extra_compile_args=["-fopenmp","--std=c++11"],
              extra_link_args=['-fopenmp',"--std=c++11"],
               include_dirs=[numpy.get_include(),'..','../ProbabilityDist/','../eigen-eigen-323c052e1731/'],
              #library_dirs=['../lib'],
              #libraries=['probdist','ptmcmc']
              ),    
    ]

setup(
    name="ptmcmc",
    ext_modules=cythonize(extensions, gdb_debug=True),
    )
