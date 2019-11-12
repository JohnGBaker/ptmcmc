#This is a header providing cython access to the relevant C++ states.hh content

from libcpp.string cimport string
from libcpp.vector cimport vector
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

    cdef cppclass stateSpace:
        stateSpace() except +
        stateSpace(int dim) except +             #need in python
        stateSpace(const stateSpace &s) except +
        int size()const
        void set_bound(int i, const boundary &b) #need in python
        #boundary get_bound(int i)const{
        #void set_names(string stringnames[])     
        void set_names(vector[string] &stringnames) #need in python
        #string get_name(int i)const
        int get_index(const string &name)const
        int requireIndex(const string name)const
        #bool enforce(valarray<double> &params)const;
        string show()const                       #need in python
        #void replaceParam(int i, const string &newname, const boundary &newbound)
        #^ TBD
        #void attach(const stateSpace &other)
        bool addSymmetry(stateSpaceInvolution &involution);

    cdef cppclass state:
        
        #const state& operator=(const state model)
        state()
        state(const stateSpace *space,int n);          #need in python ??
        state(const state &st)
        #do we need to declare copy constructor?
        #state(const stateSpace *sp, const valarray<double>&array);
        state(const stateSpace *sp, const vector[double]&array);
        #int size()const{return params.size();
        #virtual state add(const state &other)const;
        #virtual state scalar_mult(double x)const;
        #virtual double innerprod(state other)const;
        string get_string()const     #likely need in python
        #virtual string get_string(int prec)const     
        #virtual void get_params_array(valarray<double> &outarray)const
        #virtual valarray<double> get_params()const{return params;};
        vector[double] get_params_vector()const #likely need in python
        #double get_param(const int i)
        #double get_param(const string name)const
        #void set_param(const int i,const double v)
        const stateSpace * getSpace()const
        string show()const                              #likely need in python  
        #bool invalid()const

        ####From restartable interface:
        
        #void checkpoint(string path)override
        #void restart(string path)override
        #string save_string()const
        #void restore_string(const string s)

    cdef cppclass stateSpaceInvolution:
        stateSpaceInvolution() except+
        stateSpaceInvolution(const stateSpace &sp,string label,int nrand) except+
        stateSpaceInvolution(const stateSpace &sp,string label,int nrand,timing_data * timer) except+
        
        void register_reference_object(void *object)
        void register_transformState(state (*function)(void *object, const state &s, const vector[double] &randoms))
        void register_jacobian(double (*function)(void *object, const state &s, const vector[double] &randoms))
        void register_defWorkingStateSpace(void (*function)(void *object, const stateSpace &s))

    ctypedef struct timing_data:
        int every
    
        


    
