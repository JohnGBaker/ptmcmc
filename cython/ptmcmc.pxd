#This is a header providing cython access to the relevant C++ ptmcmc.hh content

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
cimport states 
cimport bayesian
cimport options

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
        #static void Quit();
        bool reporting();
