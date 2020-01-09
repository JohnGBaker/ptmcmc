# distutils: language = c++
# cython: language_level = 3

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

#General purpose
#cdef np.ndarray[np.npy_double, ndim=1, mode='c'] get_vector(vector[double] vals):
#    result=np.zeros(vals.size())
#    for i in range(vals.size()):result[i]=vals[i]
#    return result
cdef object get_vector(vector[double] vals):
    result=[0]*vals.size()
    for i in range(vals.size()):result[i]=vals[i]
    return result


cdef class boundary:
    """
    Define boundary options for a parameter space dimension.
    
    Options are: 'open'=no limit, 'limit'=closed at value, 'reflect'=reflect around value, 'wrap'=wrap at value, onto other end
    """

    #start with ref to c++ instance, need a pointer if supporting inheritance?
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
        self.potentialSyms=[]
        
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

    cpdef bool addSymmetry(self, involution sym):
        self.potentialSyms.append(sym)
        return self.space.addSymmetry(sym.cinv[0])
                      

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
    def __cinit__(self, object space_or_state=None, list values=None):

        """
        Versions of constructor:
          stateSpace(space=stateSpace,values=list)  :  set state from list of values
        """
        cdef stateSpace space
        if space_or_state is not None:
            if type(space_or_state) is state:
                space=space_or_state.getSpace()
                if values is None:values=space_or_state.get_params()
            elif type(space_or_state) is stateSpace:
                space=space_or_state
            else:
                raise ValueError("state constructor: space_or_state argument should be a state or stateSpace. Got type '"+str(type(space_or_state))+"'")
            if values is not None:
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
    cpdef object get_params(self):
        cdef vector[double] params=self.cstate.get_params_vector()
        return get_vector(params)
    cpdef np.ndarray[np.npy_double, ndim=1, mode='c'] get_params_np(self):
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

cdef class involution:
    """
    User should specify involution function which may depend on nrand random numbers in the range [-1:1]
    provided in list argument "randoms" as well as a Jacobian, if nontrivial.  The function should be an 
    involution, meaning a map onto the same space which is its own inverse, when considering the product 
    of the stateSpace and the space of randoms, and assuming that the function flips the sign of the randoms.
    An approximate reflection symmetry, and be implemented, for instance with nrand=0, and a rotation symmetry
    can be implemented with nrand=1.  The user implemetation should be tested by running ptmcmc with the
    prop_test_index flag.
    
    """

    #cdef states.stateSpaceInvolution *cinv
    #cdef have_init

    def __cinit__(self, stateSpace sp, str label,int nrand, transformState_func,jacobian_func=None, timing_every=0):
        #print("Constructing involution '"+label+"' = "+str(self)+"  nrand="+str(nrand)+"\n  transformState_func type="+str(type(transformState_func)))
        self.label=label
        cdef string clabel=label.encode('UTF-8')
        if(timing_every>0):
            self.timer.every=timing_every
            self.cinv=new states.stateSpaceInvolution(sp.space,clabel,nrand,&self.timer)
        else:
            self.cinv=new states.stateSpaceInvolution(sp.space,clabel,nrand)

        self.cinv.register_transformState(<states.state (*)(void *object, const states.state &s, const vector[double] &randoms)>self.call_transformState)
        self.transformState=transformState_func
        if jacobian_func is not None:
            self.cinv.register_jacobian(<double (*)(void *object, const states.state &s, const vector[double] &randoms)>self.call_jacobian)
            self.jacobian=jacobian_func
        self.cinv.register_reference_object(<void*>self)


    def __dealloc__(self):
        print("deallocating involution '"+self.label+"' = "+str(self))
        del self.cinv
        
    cdef states.state call_transformState(self, const states.state &s, const vector[double] &randoms) with gil:
        #A note on performance: Testing involution proposals via python is typically
        #much slower than in C++. In an example where C++ realizes the state transform
        #in 100ns, even doing nothing here requires 250ns.  Doing nothing in
        #self.transformState require 700ns (7x slower than C++) while a basic python
        #implementation of the same thing takes 2600ns.  This makes testing slow, but
        #these times are still very small compared to typical likelihood evals.
        #
        #print("Transforming state")
        #print("'"+self.label+"'  transformState_func type="+str(type(self.transformState)))
        #cdef int tid=openmp.omp_get_thread_num()
        #print('Acquired GIL: Thread',tid)
        st=state()
        st.cstate=states.state(s)
        #+200ns to here
        #cdef np.ndarray[np.npy_double, ndim=1, mode='c'] rnd=get_vector(randoms)
        #return st.cstate
        #listresult=self.transformState(st)
        #cdef state result=state(st.getSpace(),listresult)
        cdef state result=self.transformState(st,get_vector(randoms))
        #print("Transformed state")
        return result.cstate
    
    cdef double call_jacobian(self, const states.state &s, const vector[double] &randoms) with gil:
        st=state()
        st.cstate=states.state(s)
        result=self.jacobian(st,get_vector(randoms)) #should retern a list of param values
        return result
    


cdef class likelihood:
    """
    User should define a class to inherit from this one overriding evaluate_log()
    """

    cdef bayesian.bayes_likelihood *like
    cdef stateSpace space
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
        
    cpdef void basic_setup( self, stateSpace space, list types, list centers, list priorScales, list reScales=None):
        cdef int n=space.size()
        self.space=space #We hold on to this so it doesn't go out of scope
        cdef vector[string] typesvec
        cdef vector[double] centersvec
        cdef vector[double] scalesvec
        cdef vector[double] rescalesvec
        typesvec.resize(n)
        centersvec.resize(n)
        scalesvec.resize(n)
        rescalesvec.resize(n)
        for i in range(n):
            typesvec[i]=(<str>types[i]).encode('UTF-8')
            centersvec[i]=<double>centers[i]
            scalesvec[i]=<double>priorScales[i]
            if reScales is not None:
                rescalesvec[i]=<double>reScales[i]
            else:
                rescalesvec[i]=1.0
        if self.like==NULL: raise UnboundLocalError
        else: self.like.basic_setup(space.spaceptr, typesvec, centersvec, scalesvec, rescalesvec)
        #else: self.like.basic_setup(space.spaceptr, typesvec, centersvec, scalesvec)
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
        #print('argv=',argv)
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
    def report(self):
        result=self.Opt.report().decode('UTF-8')
        if len(self.argsdict)>0:
            result+=" --"
            for name in self.argsdict:
                result+="\n "+name+":"+str(self.argsdict[name])
        return  result

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
