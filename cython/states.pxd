#This is a header providing cython access to the relevant C++ states.hh content

from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "../states.cc":
    pass

cdef extern from '../states.hh' :
    cdef cppclass boundary:
        boundary() except +
        boundary(int,int ,double, double) except +
        bool enforce(double &)const
        string show()const
        void getDomainLimits(double &xmin_, double &xmax_)const

#cdef dict boundary_types={'open':0,'limit':1,'reflect':2,'wrap':2}

