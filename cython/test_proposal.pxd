#This is a header providing cython access to the relevant C++ states.hh content
# distutils: language = c++
# cython: language_level = 3

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

cimport states

cdef extern from '../test_proposal.hh' :
    cdef cppclass test_proposal:
        test_proposal()
        double KL_divergence(vector[states.state] &samplesP,vector[states.state] &samplesQ, bool approx_nn)
        double fake_KL_divergence(vector[states.state] &samplesP,vector[states.state] &samplesQ)
        int one_nnd2(states.state &s,vector[states.state] &samples)
        vector[double] all_nnd2(vector[states.state] &samples)

        
