#This is a header providing cython access to the relevant C++ options.hh content

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

#Move this to separate file?
cdef extern from "../options.hh":
    cdef cppclass Options:
       Options()
       bool parse(vector[string]&,bool verbose)
       string print_usage()
       string report()
