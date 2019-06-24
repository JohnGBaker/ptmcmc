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
import random
import argparse
cimport ptmcmc
cimport options

#cdef extern from '../ProbabilityDist/newran.h' :
#    cdef cppclass Random:
#        pass

cdef extern from '../ProbabilityDist/ProbabilityDist.h' :
    cdef cppclass ProbabilityDist:
        #static Random *getPRNG()
        @staticmethod
        void setSeed(double)

cpdef resetRNGseed(double seed):
    ProbabilityDist.setSeed(seed)


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
        return self.cstate.get_params_vector()
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
        self.like.register_evaluate_log(<double (*)(void *object, const states.state &s)>self.call_evaluate_log)
            #register_defWorkingStateSpace(<void (*)(void *object, const stateSpace &sp)>self.call_defWorkingStateSpace)
    def __dealloc__(self):
        del self.like
    cdef double call_evaluate_log(self, const states.state &s):
        st=state()
        st.cstate=states.state(s)
        return self.evaluate_log(st)
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
    
    def __init__(self):
        self.names   =[]
        self.descrips=[]
        self.defaults=[]
    def add(self,name,descrip,default):
        self.names.append(name)
        self.descrips.append(descrip)
        self.defaults.append(default)
    def parse(self,list argv):
        cdef vector[string] ccargv
        ccargv.resize(len(argv))
        for i in range(len(argv)):ccargv[i]=(<str>argv[i]).encode('UTF-8')
        #pass to Options.parse with verbose=false
        self.Opt.parse(ccargv,verbose=False)
        argv=[]
        #process residual char** array back to list of str
        for ccarg in ccargv:
            argv.append(ccarg.decode('UTF-8'))
        parser=argparse.ArgumentParser()
        for i in range(len(self.names)):
            print('adding argument: ',self.names[i],"'"+self.descrips[i]+"'","["+self.defaults[i]+"]")
            parser.add_argument('--'+self.names[i], help=self.descrips[i], default=self.defaults[i])
        print('argv=',argv)
        self.argsdict=vars(parser.parse_args(argv))
    def value(self, argname):
        return self.argsdict[argname]

#######
cdef class sampler:
    cdef ptmcmc_sampler *mcmcsampler
    def __cinit__(self):
        self.mcmcsampler=new ptmcmc_sampler()
    def __dealloc__(self):
        del self.ptsampler
    cpdef void setup(self,likelihood like):
        self.mcmcsampler.setup(like.like[0])
        self.mcmcmsampler.select_proposal()
    cpdef void addOptions(self, Options opt):
        self.mcmcsampler.addOptions(opt.Opt)
    cpdef int run(self, str base, int ic=0):
        return self.mcmcsampler.run(base.encode('UTF-8'), ic)
    cpdef int initialize(self):
        self.mcmcsampler.initialize()
    #bayes_sampler * clone()
    #state getState();
    #static void Quit();
    #bool reporting();
