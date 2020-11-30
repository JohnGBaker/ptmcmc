#This is a header providing cython access to the relevant C++ proposal_distribution.hh content

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
cimport states

cdef extern from "../proposal_distribution.hh":
    cdef cppclass proposal_distribution:
        proposal_distirbution()
        
    cdef cppclass user_gaussian_prop(proposal_distribution):
        user_gaussian_prop(void *user_parent_object, bool (*function)(const void *parent_object, void* instance_object, const states.state &, double,  const vector[double] &randoms, vector[double] &covarvec),const states.stateSpace &sp,const vector[double] &covarvec, int nrand, const string label,void * (*new_user_instance_object_function)(void*object,int id)) except +
        user_gaussian_prop(void *user_parent_object, bool (*function)(const void *parent_object, void* instance_object, const states.state &, double, const vector[double] &randoms, vector[double] &covarvec),const states.stateSpace &sp,const vector[double] &covarvec, int nrand, const string label) except +
        #user_gaussian_prop(void *object, bool (*function)(void *object, const states.state &, const vector[double] &randoms, vector[double] &covarvec),const states.stateSpace &sp,const vector[double] &covarvec, int nrand, const string label) except +
        #void register_reference_object(void *object)
        #void register_check_update(bool (*function)(void *object, const states.state &, const vector[double] &randoms, vector[double] &covarvec))
        #string show()


    
        


    
