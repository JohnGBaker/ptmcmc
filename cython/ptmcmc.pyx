# distutils: language = c++
# cython: language_level = 3
#This is just an initial test, probably don't want a separate pyx file for this

from states cimport boundary as boundary_cppclass 
#cimport states
cdef dict boundary_types={'open':0,'limit':1,'reflect':2,'wrap':2}
from libcpp cimport bool

cdef class boundary:
    #start with ref to c++ instance, need a pointer if supporting inheritance?
    cdef boundary_cppclass cppboundary
    def __cinit__(self, str lowertype='open', str uppertype='open', float xmin=float('-inf'), float xmax=float('inf')):
        self.cppboundary=boundary_cppclass(
            boundary_types[lowertype],
            boundary_types[uppertype],
            xmin,xmax)
    cpdef bool enforce(self, double & x):
        return self.cppboundary.enforce(x)
    cpdef str show(self):
        return str(self.cppboundary.show())
    cpdef getDomainLimits(self):
        cdef double xmin,xmax
        self.cppboundary.getDomainLimits(xmin,xmax)
        return xmin,xmax


    
        
