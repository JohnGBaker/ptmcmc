# distutils: language = c++
# cython: language_level = 3

from libcpp cimport bool
from typing import List
from libcpp.vector cimport vector
cimport numpy as np
import numpy as np
cimport ptmcmc
import ptmcmc
cimport states
import copy
from libc.stdio cimport printf

cdef class test:
    """
    Define the parameter state.
    """
    
    #start with ref to c++ instance, need a pointer if supporting inheritance?
    cdef test_proposal ctp
    def __cinit__(self):
       self.ctp=test_proposal()
    
    cpdef double KL_divergence(self, list samplesP, list samplesQ, bool approx_nn):

        cdef vector[states.state] samplesPvec=get_states_vector_from_list(samplesP)
        cdef vector[states.state] samplesQvec=get_states_vector_from_list(samplesQ)
        cdef double result=self.ctp.KL_divergence(samplesPvec,samplesQvec, approx_nn)
        return result
    
    cpdef double fake_KL_divergence(self, list samplesP, list samplesQ):

        cdef vector[states.state] samplesPvec=get_states_vector_from_list(samplesP)
        cdef vector[states.state] samplesQvec=get_states_vector_from_list(samplesQ)
        cdef double result=self.ctp.fake_KL_divergence(samplesPvec,samplesQvec)
        return result
    
cdef vector[states.state] get_states_vector_from_list(list pystates):
    cdef vector[states.state] result;
    result.resize(len(pystates))
    for i in range(len(pystates)):
        st=<ptmcmc.state>(pystates[i])
        result[i]=st.cstate
    return result


