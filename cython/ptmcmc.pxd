#This is a header providing cython access to the relevant C++ ptmcmc.hh content
# distutils: language = c++
# cython: language_level = 3

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
cimport states 
cimport bayesian
cimport options
cimport numpy as np

cdef extern from "../ptmcmc.cc":
    pass
       
cdef extern from "../ptmcmc.hh":
    cdef cppclass ptmcmc_sampler:
        ptmcmc_sampler()
        void select_proposal()
        void addOptions(options.Options &opt)
        int run(const string & base, int ic) nogil
        void setup(bayesian.bayes_likelihood &llike)
        int initialize()nogil;
        #int analyze(const string & base, int ic, int Nsigma, int Nbest, bayes_likelihood &like);
        ptmcmc_sampler * clone_ptmcmc_sampler()
        states.state getState();	
        bool reporting();

    void ptmcmc_sampler_Init "ptmcmc_sampler::Init" ()
    void ptmcmc_sampler_Quit "ptmcmc_sampler::Quit" ();

cdef class boundary:
    cdef states.boundary bound
    cpdef bool enforce(self, double & x)
    cpdef str show(self)
    cpdef getDomainLimits(self)

cdef class stateSpace:
    #start with ref to c++ instance, need a pointer if supporting inheritance?
    cdef states.stateSpace space  #should this be changed to a pointer?
    cdef const states.stateSpace *spaceptr
    cdef bool constpointer
    cdef object potentialSyms
    cpdef void set_bound(self, str name, boundary b)
    cpdef void set_names(self, list stringnames)
    cpdef int size(self)
    cpdef int requireIndex(self, str name)
    cdef void point(self, const states.stateSpace *sp)
    cpdef str show(self)
    cpdef bool addSymmetry(self, involution sym)

cdef class state:
    cdef states.state cstate
    cdef set_from_list(self,stateSpace sp,list values)
    cdef wrap(self,states.state obj)
    cpdef str get_string(self)
    cpdef object get_params(self)
    cpdef np.ndarray[np.npy_double, ndim=1, mode='c'] get_params_np(self)
    cpdef stateSpace getSpace(self)
    cpdef str show(self)

cdef class involution:
    cdef states.stateSpaceInvolution *cinv
    cdef object transformState
    cdef object jacobian
    cdef str label
    cdef states.timing_data timer
    cdef states.state call_transformState(self, const states.state &s, const vector[double] &randoms)with gil
    cdef double call_jacobian(self, const states.state &s, const vector[double] &randoms)with gil
    


