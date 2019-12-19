#This is a header providing cython access to the relevant C++ bayesian.hh content

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from states cimport *

#cdef extern from "../probability_function.cc":
#    pass

#cdef extern from "../ProbabilityDist/ProbabilityDist.cxx":
#    pass

cdef extern from "../ProbabilityDist/simpstr.cxx":
    pass

cdef extern from "../ProbabilityDist/newran1.cxx":
    pass

cdef extern from "../ProbabilityDist/newran2.cxx":
    pass

cdef extern from "../ProbabilityDist/myexcept.cxx":
    pass

cdef extern from "../ProbabilityDist/extreal.cxx":
    pass

cdef extern from '../bayesian.hh' :
    cdef cppclass bayes_likelihood:
         #bayes_likelihood(stateSpace *sp,bayes_data *data,bayes_signal *signal)
         #^Note: no python interface for bayes_data, bayes_signal yet
         bayes_likelihood() except +
         #void checkPointers()const
         #^need?
         #void basic_setup(stateSpace *sp, sampleable_probability_function *prior )
         #^ sampleable_probability_function not yet interfaced in python
         void basic_setup(const stateSpace *sp,const vector[string] &types, const vector[double] &centers,const vector[double] &priorScales)                       
         void basic_setup(const stateSpace *sp,const vector[string] &types, const vector[double] &centers,const vector[double] &priorScales,const vector[double] &reScales)                       
         void getScales(vector[double] &scales)
         #int size()const
         #virtual void reset()
         #virtual state bestState();
         #virtual double bestPost();
         #^^^^not clear if these will be needed
         #virtual double getFisher(const state &s0, vector<vector<double> >&fisher_matrix)
         #virtual vector<double> getVariances(const state &st,vector<double>&svar)const
         #virtual void write(ostream &out,state &st)
         #^ may need to override, perhaps trivally
         #virtual void writeFine(ostream &out,state &st,int samples=-1, double xstart=0, double xend=0)
         #^ may need to override, perhaps trivally
         #void getFineGrid(int & nfine, double &xfinestart, double &xfineend)const

         ### Inherited from bayes_component:
         #stateSpace nativeSpace;
         #virtual void setState(const state &st){working_state=&st; checkWorkingStateSpace();};
         #shared_ptr<const sampleable_probability_function> nativePrior;
         #void haveSetup(){have_setup=true;};
         #void setPrior(const sampleable_probability_function* prior)
         #^^^ not needed for simplified interface
         #void setNoParams()
         #bool checkSetup(bool quiet=false)const
         # ^unnec?
         #virtual void panic(string message="")
         #virtual void alert(string message="")
         #^^ maybe useful to override??

         ###Inherited from Optioned
         #virtual void addOptions(Options &opt)
             
         ###Simple interface
         bool check_posterior #User can set to false to skip checks for unreasonable posterior values
         void register_reference_object(void *object)    
         void register_evaluate_log(double (*function)(void *object, const state &s))
         void register_defWorkingStateSpace(void (*function)(void *object, const stateSpace &sp))
         void defWorkingStateSpace(const stateSpace &sp)  
         double evaluate_log(state &s)           
         state draw_from_prior()
         const stateSpace* getObjectStateSpace()const
         void reset()
         double bestPost()


