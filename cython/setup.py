from distutils.core import setup,Extension
from Cython.Build import cythonize

extensions = [
    Extension("ptmcmc",['ptmcmc.pyx'],
              language='c++',
              include_dirs=[],
              libraries=[],
              libary_dirs=[]),    
    ]

setup(
    name="PTMCMC",
    ext_modules=cythonize(extensions),
    )
