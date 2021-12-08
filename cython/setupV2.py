from distutils.core import setup,Extension
from Cython.Build import cythonize
import os
import numpy

here=os.path.dirname(os.path.abspath(__file__))+'/'
#locale='discover'
locale=None

#defaults
compiler=None
usempi=False

if locale=="laptop":
    #compiler="g++-mp-8"
    compiler='/opt/local/bin/mpicxx-mpich-gcc8'
    usempi=True
elif locale=="discover":
    compiler="mpicxx"
    usempi=True
else:
    locale=None
    
if locale is not None:
    os.environ["CC"] = compiler
    os.environ["CXX"] = compiler
    os.environ['LD'] = compiler
    os.environ["CXXFLAGS"] = "-mmacosx-version-min=10.14"

comp_args=["-fopenmp","--std=c++11"]
if usempi:comp_args.append("-DUSE_MPI")

extensions = [
    Extension("ptmcmc",[here+'ptmcmc.pyx'],
              depends=[here+'../lib/'+file for file in ['libptmcmc.a','libprobdist.a']],
              language='c++',
              extra_compile_args=comp_args,
              extra_link_args=comp_args,
              include_dirs=[numpy.get_include(),here+'..',here+'../ProbabilityDist/',here+'../eigen-eigen-323c052e1731/'],
              library_dirs=[here+'../lib'],
              libraries=['probdist','ptmcmc']
              ),    

    ]

setup(
    name="ptmcmc",
    ext_modules=cythonize(extensions, gdb_debug=True),
    )

