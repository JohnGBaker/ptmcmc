# distutils: language = c++
# cython: language_level = 3
#This is just an initial test, probably don't want a separate pyx file for this

#from states cimport boundary as boundary_cppclass 
#cimport states
cimport bayesian
cdef dict boundary_types={'open':0,'limit':1,'reflect':2,'wrap':3}
from libcpp cimport bool
from typing import List
from libcpp.string cimport string
from libcpp.vector cimport vector
cimport numpy as np
import numpy as np
import random
import argparse
cimport ptmcmc
cimport options
import copy
cimport openmp
#cimport cython.parallel

from libc.stdio cimport printf

  
#cdef extern from '../ProbabilityDist/newran.h' :
#    cdef cppclass Random:
#        pass

#cdef extern from 'python.h':
#    cdef void PyEval_InitThreads()
#    cdef int PyEval_ThreadsInitialized()nogil

cdef extern from '../ProbabilityDist/ProbabilityDist.h' :
    cdef cppclass ProbabilityDist:
        #static Random *getPRNG()
        @staticmethod
        void setSeed(double)

cpdef resetRNGseed(double seed):
    ProbabilityDist.setSeed(seed)

cpdef Init():
    ptmcmc_sampler_Init()
        
cpdef Quit():
    ptmcmc_sampler_Quit()
        
#Some initialization
random.seed()
resetRNGseed(random.random())



cdef class boundary:
    """
    Define boundary options for a parameter space dimension.
    
    Options are: 'open'=no limit, 'limit'=closed at value, 'reflect'=reflect around value, 'wrap'=wrap at value, onto other end
    """

    #start with ref to c++ instance, need a pointer if supporting inheritance?
    cdef states.boundary bound
    def __cinit__(self, str lowertype='open', str uppertype='open', float xmin=float('-inf'), float xmax=float('inf')):
        self.bound=states.boundary(
            boundary_types[lowertype],
            boundary_types[uppertype],
            xmin,xmax)
    cpdef bool enforce(self, double & x):
        return self.bound.enforce(x)
    cpdef str show(self):
        return str(self.bound.show())
    cpdef getDomainLimits(self):
        cdef double xmin,xmax
        self.bound.getDomainLimits(xmin,xmax)
        return xmin,xmax

cdef class stateSpace:
    """
    Define the parameter state space.    
    """

    #start with ref to c++ instance, need a pointer if supporting inheritance?
    cdef states.stateSpace space  #should this be changed to a pointer?
    cdef const states.stateSpace *spaceptr
    cdef bool constpointer;
    def __cinit__(self, **kwargs): #by default we construct and own the stateSpace object
        """
        Versions of constructor:
          stateSpace()                       :  empty space
          stateSpace(dim=n)                  :  unspecified space of dimension n
          stateSpace(pars={name0:bound0,...} :  set pars by a dictionary of {name:bound}
          stateSpace(ptr=cpp_pointer)        :  (within cython) set by pointer to cpp stateSpace
        """

        if 'pars' in kwargs:
            pars=kwargs['pars']
            dim=len(pars)
            self.space=states.stateSpace(<int>dim)
            self.spaceptr=&self.space
            self.set_names(list(pars.keys()))
            #print('names:',list(pars.keys()))
            for name in pars.keys():
                bound=<boundary>pars[name]
                #print(name,bound)
                #print('bound type',type(bound))
                #print('  '+name+':',bound.show())
                self.set_bound(name,bound)
        else:
            dim=0
            if 'dim' in kwargs:dim=kwargs['dim']
            self.space=states.stateSpace(<int>dim)
            self.spaceptr=&self.space
            self.constpointer=False
            
    cpdef void set_bound(self, str name, boundary b):
        #print('set_bound for ',name)
        if self.constpointer: return #maybe raise an exception
        cdef int idx=self.requireIndex(name)
        #print('idx=',idx)
        self.space.set_bound(idx,b.bound)
    cpdef void set_names(self, list stringnames):
        if self.constpointer: return #maybe raise an exception
        cdef int n=len(stringnames)
        cdef vector[string] namesvec
        namesvec.resize(n)
        for i in range(n):namesvec[i]=(<str>stringnames[i]).encode('UTF-8')
        self.space.set_names(namesvec)
    cpdef int size(self):
        return self.spaceptr.size()
    cpdef int requireIndex(self, str name):
        cdef string sname = name.encode('UTF-8')
        cdef int idx=self.spaceptr.get_index(sname)  
        if(idx<0):raise ValueError("Index not found")
        return idx
        #return self.spaceptr.requireIndex(sname)
        #bool enforce(valarray<double> &params)const;
    cdef void point(self, const states.stateSpace *sp):
        """
        Forget about the constructed space and set this class to point (as const)
        to an independently defined (and owned) object
        """
        
        self.constpointer=True
        self.spaceptr=sp
    cpdef str show(self):
        return (self.spaceptr.show()).decode('UTF-8')

"""
cdef double nada(void * obj,const states.state &s)nogil:
    printf(" th in:%i\n",PyEval_ThreadsInitialized())
    cdef double result=0
    with gil:
        result=0
    return result;
"""
cdef class state:
    """
    Define the parameter state.
    """
    
    #start with ref to c++ instance, need a pointer if supporting inheritance?
    cdef states.state cstate
    def __cinit__(self, stateSpace space=None, list values=None):
        """
        Versions of constructor:
          stateSpace(space=stateSpace,values=list)  :  set state from list of values
        """

        if values is not None  and space is not None:
            self.set_from_list(space,values)
    cdef set_from_list(self,stateSpace sp,list values):
            cdef vector[double] valuesvec
            valuesvec.resize(len(values))
            for i,val in enumerate(values):valuesvec[i]=<double>values[i]
            self.cstate=states.state(sp.spaceptr,valuesvec)
    cdef wrap(self,states.state obj):
        self.cstate=states.state(obj)
    cpdef str get_string(self):
        return (self.cstate.get_string()).decode('UTF-8')
    cpdef np.ndarray[np.npy_double, ndim=1, mode='c'] get_params(self):
        cdef vector[double] params=self.cstate.get_params_vector()
        result=np.zeros(params.size())
        for i in range(params.size()):result[i]=params[i]
        return result
    cpdef stateSpace getSpace(self):
        s = stateSpace()
        s.point(self.cstate.getSpace())
        return s
    cpdef str show(self):
        return (self.cstate.show()).decode('UTF-8')


cdef class likelihood:
    """
    User should define a class to inherit from this one overriding evaluate_log()
    """

    cdef bayesian.bayes_likelihood *like
    def __cinit__(self):
        self.like=new bayesian.bayes_likelihood()
        #check_posterior #User can set to false to skip checks for unreasonable posterior values
        self.like.register_reference_object(<void*>self)
        #self.like.register_evaluate_log(nada)
        self.like.register_evaluate_log(<double (*)(void *object, const states.state &s)>self.call_evaluate_log)
            #register_defWorkingStateSpace(<void (*)(void *object, const stateSpace &sp)>self.call_defWorkingStateSpace)
    def __dealloc__(self):
        del self.like
    cdef double call_evaluate_log(self, const states.state &s) with gil:
        cdef int tid=openmp.omp_get_thread_num()
        #print('Acquired GIL: Thread',tid)
        cdef double result
        #print("calling evaluate_log on thread",tid)
        st=state()
        st.cstate=states.state(s)
        #result=0
        result=self.evaluate_log(st)
        #print("returning from evaluate_log on thread",tid)
        #print('ReReleasing GIL: Thread',tid)
        return result
    
    cpdef double evaluate_post(self, state s):
        #This is something of a hacky interface to get the posterior with a minimal interface
        #first reset the likelihood "best" posterior
        self.like.reset()
        #then evaluate the likelihood (which will also update the best posterior)
        self.like.evaluate_log(s.cstate)
        return self.like.bestPost()
        
    cpdef void basic_setup( self, stateSpace space, list types, list centers, list scales):
        cdef int n=space.size()
        cdef vector[string] typesvec
        cdef vector[double] centersvec
        cdef vector[double] scalesvec
        typesvec.resize(n)
        centersvec.resize(n)
        scalesvec.resize(n)
        for i in range(n):
            typesvec[i]=(<str>types[i]).encode('UTF-8')
            centersvec[i]=<double>centers[i]
            scalesvec[i]=<double>scales[i]
        if self.like==NULL: raise UnboundLocalError
        else: self.like.basic_setup(space.spaceptr, typesvec, centersvec, scalesvec)
    cpdef state draw_from_prior(self):
        if self.like==NULL: raise UnboundLocalError
        cdef states.state s=self.like.draw_from_prior();
        st=state()
        st.cstate=states.state(s)
        return st
    cpdef stateSpace getObjectStateSpace(self):
       sp=stateSpace()
       sp.point(self.like.getObjectStateSpace())
       return sp


cdef class Options:
    '''
    Class for handling options for program subcomponents. 

    For wrapped C++ code the options are passed down into the C++ optioned interface.
    Options not processed in the C++ code are presumed to be handled at the python level.
    Options defined by python native code are processed, stored and accessed here.
    '''
    cdef options.Options Opt
    cdef list names
    cdef list  descrips
    cdef list defaults
    cdef dict argsdict
    cdef bool have_dict
    
    def __cinit__(self):
        self.names   =[]
        self.descrips=[]
        self.defaults=[]
        self.Opt =options.Options()
        self.have_dict=False
        
    def add(self,name,descrip,default):
        self.names.append(name)
        self.descrips.append(descrip)
        self.defaults.append(default)
    def parse(self,list argv):
        cdef vector[string] ccargv
        argv=['ptmcmc']+argv
        ccargv.resize(len(argv))
        if '--help' in argv:
            parser=self.make_parser()
            parser.print_help()
            print(self.Opt.print_usage().decode('UTF-8'))
            parser.exit()
        for i in range(len(argv)):ccargv[i]=(<str>argv[i]).encode('UTF-8')
        #pass to Options.parse with verbose=false
        self.Opt.parse(ccargv,verbose=False)
        argv=[]
        #process residual char** array back to list of str
        for ccarg in ccargv[1:]:
            argv.append(ccarg.decode('UTF-8'))
        print('argv=',argv)
        parser=self.make_parser()
        self.argsdict=vars(parser.parse_args(argv))
    def make_parser(self):
        parser=argparse.ArgumentParser()
        for i in range(len(self.names)):
            #print('adding argument: ',self.names[i],"'"+self.descrips[i]+"'","["+self.defaults[i]+"]")
            parser.add_argument('--'+self.names[i], help=self.descrips[i], default=self.defaults[i])
        #print('making parser:\n',parser.print_help())
        return parser
    def value(self, argname):
        return self.argsdict[argname]

#######
cdef class sampler:
    cdef ptmcmc_sampler *mcmcsampler
    cdef Options opt
    def __cinit__(self,Options opts=None):
        self.mcmcsampler=new ptmcmc_sampler()
        if opts is not None:
            self.opt=opts
            self.mcmcsampler.addOptions(self.opt.Opt)
#            self.initialize()
#    cdef initialize(self):
#        PyEval_InitThreads() #This is essential to initilize the GIL
    def __dealloc__(self):
        del self.mcmcsampler
    cpdef void setup(self,likelihood like):
        self.mcmcsampler.setup(like.like[0])
        self.mcmcsampler.select_proposal()
    cpdef int run(self, str base, int ic=0):
        cdef int result
        cdef string basestring=base.encode('UTF-8')
        #PyEval_InitThreads() #This may be essential to initilize the GIL

        #print('Releasing GIL: Thread',openmp.omp_get_thread_num())
        with nogil: 
           result=self.mcmcsampler.run(basestring, ic)
        #print('Re-acquiring GIL: Thread',openmp.omp_get_thread_num())
        return result
    cpdef int initialize(self):
        with nogil:
            self.mcmcsampler.initialize()
    cpdef sampler clone(self):
        cdef sampler new_sampler=sampler()
        new_sampler.opt=self.opt
        new_sampler.mcmcsampler=self.mcmcsampler.clone_ptmcmc_sampler()
        return new_sampler
    #state getState();
    #static void Quit();
    cpdef bool reporting(self):
        return self.mcmcsampler.reporting()
