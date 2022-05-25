///State and state space structures for Bayesian analysis
///
///state objects are tied to a stateSpace object which specifies to domain.
///boundary objects allow specification of the domain bounraies in each dimension.
///John G Baker - NASA-GSFC (2013-2015)

#ifndef PTMCMC_STATES_HH
#define PTMCMC_STATES_HH
#include <map>
#include <valarray>
#include <vector>
#include <sstream>
//#include <cmath>
#include <iostream>
//#include <utility>
//#include <memory>
#include "restart.hh"
#include "newran.h"
#include <chrono>

using namespace std;

typedef unsigned int uint;

class stateSpaceInvolution;

//********* BASE CLASSES *************
/// A class for specifying 1-D boundary info. Used for stateSpace definition.
class boundary {
  int lowertype;
  int uppertype;
  double xmin;
  double xmax;
public:
  static const int open=0;
  static const int limit=1;
  static const int reflect=2;
  static const int wrap=3;
  boundary(int lowertype=open,int uppertype=open,double min=-INFINITY,double max=INFINITY):lowertype(lowertype),uppertype(uppertype),xmin(min),xmax(max){
    //if(lowertype==open)xmin=-INFINITY; //Should be cosmetic since no checking will be done
    //if(uppertype==open)xmax=-INFINITY; //but not fully cosmetic because effect of getDomainLimits changes... what TODO?
};
  ///enforce boundary condition. If consistent enforcement was achieved return true.
  bool enforce(double &x)const;
  ///Show structural info
  string show()const;
  void getDomainLimits(double &xmin_, double &xmax_)const{xmin_=xmin;xmax_=xmax;};
  bool isWrapped()const{return lowertype==wrap && uppertype==wrap;};
};

/// State space class allows reference to overall limits and structure of the space.
/// Can provide support for boundary conditions, maybe special flows that can be used in proposals...
/// Plan to expand to transdimensional:
///   -Allow compound state spaces with an undetermined number multiple copies of identical space
///   -Potentially allow named subspaces
///   -Allow compound state-spaces built of subspaces and/or including multi-copy subspaces
///   -Develop a naming scheme for mapping through the hierarchy and contexting components
///   -Possibly move prior information into subspace structure.

class stateSpace {
  int dim;
  vector<boundary> bounds;
  vector<string> names;
  map<string,int> index;
  bool have_names;
public:
  stateSpace(int dim=0):dim(dim){
    bounds.resize(dim,boundary());//default to default boundaries (ie allow all reals)
    have_names=false;    
  };
  int size()const{return dim;};
  void set_bound(int i, const boundary &b){
    if(i<dim)bounds[i]=b;
    else{
      cout<<"stateSpace::set_bound: Index out of range, "<<i<<">="<<dim<<"."<<endl;
      exit(1);
    }
  };      
  boundary get_bound(int i)const{
    if(i<0||i>=dim){
      cout<<"stateSpace::set_bound: Index out of range, "<<i<<">="<<dim<<"."<<endl;
      exit(1);
    }
    return bounds[i];
  };      
  void set_names(const string stringnames[]){
    names.resize(dim,"");
    for(uint i=0;i<dim;i++){
      names[i]=stringnames[i];
      index[names[i]]=i;
    }
    have_names=true;
  };
  void set_names(const vector<string> &stringnames){
    if(stringnames.size()<dim){
      cout<<"stateSpace::set_names: Vector of param names is too short. Quitting."<<endl;
      exit(-1);
    }
    names.resize(dim,"");
    for(uint i=0;i<dim;i++){
      names[i]=stringnames[i];
      index[names[i]]=i;
    }
    have_names=true;
  };
  string get_name(int i)const {
    if(have_names&&i<dim)return names[i];
    else return "[unnamed]";
  };
  int get_index(const string &name)const{
    //cout<<"get_index: name="<<name<<endl;
    //cout<<"index.size="<<index.size()<<endl;
    if(have_names&&index.count(name)>0){
      //cout<<"index.size()="<<index.size()<<endl;
      //cout<<"index.at(name)="<<index.at(name)<<endl;
      return index.at(name);
    }
    else return -1;
  };      
  ///This is the same as get_index but fails with an error message if not found.
  int requireIndex(const string &name)const{
    //cout<<"requireIndex:name="<<name<<endl;
    int idx=get_index(name);
    //cout<<"requireIndex:idx="<<idx<<endl;
    if(idx<0){
      //cout<<"Oops:name="<<name<<", have_names="<<have_names<<endl;     
      cout<<"stateSpace::checkNames(): Name '"<<name<<"' not found in state:"<<show()<<endl;
      exit(1);
    }
    return idx;
  };
  ///Test whether another space is a subspace of this one.
  bool contains(const stateSpace &other);
  bool enforce(valarray<double> &params,bool verbose=false)const;
  ///Show structural info
  string show()const;
  ///replace a parameter in one space dimension
  void replaceParam(int i, const string &newname, const boundary &newbound){
    if(!have_names){
      cout<<"stateSpaceTransform::replaceParam: Cannot apply transform to a space for which names are not defined."<<endl;
      exit(1);
    }
    index.erase(names[i]);
    index[newname]=i;
    names[i]=newname;
    bounds[i]=newbound;
  };
  ///join this space to another space
  void attach(const stateSpace &other);
  
  ///Construct a subspace based on a vector of parameter names.  Not that potential symmetries are dropped.
  stateSpace subspace_by_name(const vector<string>subspace_names)const;

  ///Support for optional additional list of PotentialSymmetries
  ///These are understood to be useful involutions on the space which may interesting in certain applications, like proposals
  ///It is not assumed that the map provides an actual isomorphism of the space.
private:
  vector<stateSpaceInvolution > potentialSyms;
public:
  bool addSymmetry(stateSpaceInvolution &involution);
  const vector<stateSpaceInvolution >get_potentialSyms()const{
    return potentialSyms;
  };
};
      
///Class for holding and manipulating bayesian parameter states
class state :restartable {
  bool valid;
  const stateSpace *space;
  valarray<double> params;
public:
  //Need assignment operator since default valarray assignment is problematic
  const state& operator=(const state model){space=model.space,valid=model.valid,params.resize(model.size(),0);params=model.params;return *this;};
  state(const stateSpace *space=nullptr,int n=0);
  state(const stateSpace *sp, const valarray<double>&array);
  state(const stateSpace *sp, const vector<double>&array);
  void checkpoint(string path)override{//probably won't want to use this; woudl need a separate dir for each state...
    ostringstream ss;
    ss<<path<<"/state.cp";
    ofstream os;
    openWrite(os,ss.str());
    writeString(os,save_string());
  };
  void restart(string path)override{//probably won't want to use this; woudl need a separate dir for each state...
    ostringstream ss;
    ss<<path<<"/state.cp";
    ifstream os;
    openRead(os,ss.str());
    string s;
    readString(os,s);
    restore_string(s);
  };
  string save_string()const{
    //We assume *space is known independently
    ostringstream oss;
    writeInt(oss,valid);
    writeInt(oss,params.size());
    for(int i=0;i<params.size();i++)writeDouble(oss,params[i]);
    return oss.str();
  };
  void restore_string(const string s){
    //We assume *space is known independently
    istringstream iss(s);
    int n;
    readInt(iss,n);valid=n;
    readInt(iss,n);
    params.resize(n,0);
    for(int i=0;i<params.size();i++)readDouble(iss,params[i]);
  };
  int size()const{return params.size();}
  void enforce(bool verbose=false);
  //some algorithms rely on using the states as a vector space
  virtual state add(const state &other)const;
  virtual state scalar_mult(double x)const;
  ///For some applications it is necessary to have an inner product on the state space. Probably should move this out to stateSpace.
  virtual double innerprod(state other,bool constrained=false)const;
  ///Compute the squared distance between two states.  Should account for wrapped dimensions.
  virtual double dist2(const state &other)const;
  virtual string get_string(int prec=-1)const;
  virtual void get_params_array(valarray<double> &outarray)const{
    //outarray.resize(params.size());
    //for(size_t i=0;i<size();i++)outarray[i]=params[i];
      outarray=std::move(params);
    return;
  }
  virtual valarray<double> get_params()const{return params;};
  virtual vector<double> get_params_vector()const{vector<double> v;v.assign(begin(params),end(params));return v;};
  double get_param(const int i)const{return params[i];};
  double get_param(const string name)const{if(space)return params[space->requireIndex(name)];else{cout<<"state::get_param(name):Need a stateSpace to get param by name."<<endl;exit(1);};};
  void set_param(const int i,const double v){params[i]=v;};
  const stateSpace * getSpace()const{return space;};
  ///Get indicies for projection down to a subspace
  vector<int> projection_indices_by_name(const stateSpace *subspace)const;
  ///Show param info
  string show()const;
  bool invalid()const{return !valid;};
};

///Interface class for objects using stateSpace to describe their content
class stateSpaceInterface {
  bool have_working_space;
public:
  stateSpaceInterface(){have_working_space=0;};
  virtual void defWorkingStateSpace(const stateSpace &sp)=0;
  virtual void haveWorkingStateSpace(){have_working_space=true;};
  //virtual stateSpace getObjectStateSpace()const{return stateSpace();};
  virtual const stateSpace* getObjectStateSpace()const=0;
  void checkWorkingStateSpace()const{
    if(!have_working_space){
      cout<<"stateSpaceTransform::checkWorkingSpace: [this="<<this<<"] Must define working space (defWorkingStateSpace) before applying it."<<endl;
      exit(1);
    }
  };
};

///Base class for stateSpace transformations
class stateSpaceTransform {
protected:
  bool have_working_space;
public:
  stateSpaceTransform(){have_working_space=false;};
  virtual stateSpace transform(const stateSpace &sp)=0;
  virtual stateSpace inverse_transform(const stateSpace &sp){cout<<"stateSpaceTransform:No inverse transform available"<<endl;exit(1);};
  ///jacobian determinant function
  virtual double jacobian(const state &s)const {cout<<"stateSpaceTransform:No Jacobian available"<<endl;exit(1);};
  virtual state transformState(const state &s)const=0;
  virtual void defWorkingStateSpace(const stateSpace &sp)=0;
};

///Dynamically generated one-to-one generic state space transform class
///
///Perhaps a way to do this dynamically, rather than hand-coding each one as a separate derived class..
///This is just an idea, not yet developed for use....
class stateSpaceTransform1D : public stateSpaceTransform {
  string in;
  string out;
  double (*func)(double);
  boundary bound;
  bool have_bound;
  int idx_inout;
  public:
  stateSpaceTransform1D(string in="",string out="",double (*func)(double)=[](double a){return a;}):in(in),out(out),func(func){have_bound=false;};
  void set_bound(const boundary &b){
    have_bound=true;
    bound=b;
  }
  virtual stateSpace transform(const stateSpace &sp){
    
    stateSpace outsp=sp;
    int ind=sp.get_index(in);
    if(ind<0){
      cout<<"stateSpaceTransform1D::transform: Parameter name '"<<in<<"' not found."<<endl;
      exit(1);
    } else {
      boundary b;
      if(have_bound)b=bound;
      else b=sp.get_bound(ind);
      outsp.replaceParam(ind, out, b);
    }
    defWorkingStateSpace(sp);
    return outsp;    
  };
  virtual state transformState(const state &s)const{
    if(!have_working_space){
      cout<<"stateSpaceTransform1D::transformParams: Must call deWorkingStateSpace before transformParams."<<endl;
      exit(1);
    }
    state st=s;
    st.set_param(idx_inout,(*func)(s.get_param(idx_inout)));
    return st;
  };
  virtual void defWorkingStateSpace(const stateSpace &sp){
    int ind=sp.get_index(in);
    if(ind<0){
      cout<<"stateSpaceTransform1D::defWorkingStateSpace: Parameter name '"<<in<<"' not found."<<endl;
      exit(1);
    } else {
      idx_inout=ind;
      have_working_space=true;
    }
  };
};

///Dynamically generated n-to-n generic state space transform class
///
///Perhaps a way to do this dynamically, rather than hand-coding each one as a separate derived class..
///This is just an idea, not yet developed for use....
class stateSpaceTransformND : public stateSpaceTransform {
  int dim;
  vector<string> ins;
  vector<string> outs;
  vector<double> (*func)(vector<double>&a);
  vector<boundary> bounds;
  bool have_bounds;
  int idxs[2];
  public:
  stateSpaceTransformND(int dim=0,vector<string> ins={},vector<string> outs={},vector<double> (*func)(vector<double>&a)=[](vector<double> &a){return a;}):ins(ins),outs(outs),func(func),dim(dim){
    have_bounds=false;
    if(ins.size()!=dim||outs.size()!=dim){
      cout<<"stateSpaceTransformND::stateSpaceTransformND: Dimensions do not match."<<endl;
      exit(1);
    }
  };
  void set_bounds(const vector<boundary> &b){
    have_bounds=true;
    bounds=b;
  }
  virtual stateSpace transform(const stateSpace &sp){
    stateSpace outsp=sp;
    for(int i=0;i<dim;i++){
      int ind=sp.get_index(ins[i]);
      if(ind<0){
	cout<<"stateSpaceTransformND::transform: Parameter name '"<<ins[i]<<"' not found."<<endl;
	exit(1);
      } else {
	boundary b;
	if(have_bounds)b=bounds[i];
	else b=sp.get_bound(ind);
	outsp.replaceParam(ind, outs[i], b);
      }
    }
    defWorkingStateSpace(sp);
    return outsp;    
  };
  virtual state transformState(const state &s)const{
    if(!have_working_space){
      cout<<"stateSpaceTransformND::transformParams: Must call defWorkingStateSpace before transformParams."<<endl;
      exit(1);
    }
    state st=s;
    vector<double>inpars(2),outpars(2);
    for(int i=0;i<dim;i++)inpars[i]=s.get_param(idxs[i]);
    outpars=(*func)(inpars);
    for(int i=0;i<dim;i++)st.set_param(idxs[i],outpars[i]);
    return st;
  };
  virtual void defWorkingStateSpace(const stateSpace &sp){
    for(int i=0;i<dim;i++){
      int ind=sp.get_index(ins[i]);
      if(ind<0){
	cout<<"stateSpaceTransformND::defWorkingStateSpace: Parameter name '"<<ins[i]<<"' not found."<<endl;
	exit(1);
      } else {
	idxs[i]=ind;
      }
    }
    have_working_space=true;
  };
};

struct timing_data {
  int count;
  int every;
  double time;
};

///Involutions are a special class of state space transforms representing symmetries.  They are automorphisms
///where the function is its own inverse, so the transformed space is identical to the domain and the inverse
///state transform is the same as the forward state transform.  We provide a test to verify this property.
///
///We also allow functions which depend on an additional vector of uniform random numbers of length nrand, with
///values in (-1,1).  These can represent continuous symmetries.  In this case, the transform is an involution
///on the product of the stateSpace together with nrand copies of the signed unit interval. It need not be its
//own inverse on the stateSpace alone.  Rather the inverse should be acheived by reversing the sign on the random
///vector.  If a family of point-wise involutions is required, then the function should depend only on the
///absolute value of each random number. With nrand>0 the Jacobian function should provide the Jacobian on the
///full space including the random components.
///
///Note that this is still in development and we may generalize this further beyond involutions or even beyond
///endomorphisms.  The interface should be considered volatile for now.
///
///Some clean-up needed of hacky overloading of user_object for random numbers
class stateSpaceInvolution : public stateSpaceTransform {
  string label;
  int nrand;
  timing_data * performance_data;
protected:
  const stateSpace *domainSpace; //Note the Transform can be applied as long as this can be identified as a subspace.
  bool have_working_space;
public:
  stateSpaceInvolution(const stateSpace &sp,string label="",int nrand=0,timing_data *perf_data=nullptr):label(label),nrand(nrand),performance_data(perf_data){
    have_working_space=false;
    transformState_registered=false;
    jacobian_registered=false;
    defWorkingStateSpace_registered=false;
    randoms.resize(nrand);
    user_object=nullptr;
    domainSpace=&sp;
    if(performance_data){
      performance_data->count=0;
      performance_data->time=0;
    }
  };
  virtual void set_random(Random &rng){
    for(auto & x : randoms)x=(rng.Next()*2.0-1.0);
  };
  virtual string get_label()const {return label;};
  virtual stateSpace transform(const stateSpace &sp)override{return stateSpace(sp);};//Should we enforce that domain is subspace of sp?
  virtual stateSpace inverse_transform(const stateSpace &sp)override{return stateSpace(sp);};
  virtual state transformState(const state &s)const override{    
    if(transformState_registered){
      if(have_working_space){
        auto start=chrono::high_resolution_clock::now();
	state transformed=(*user_transformState)(user_object,s,randoms);
	//Enforce stateSpace limits
	transformed.enforce();
	if(performance_data){
	  performance_data->count++;
	  auto stop=chrono::high_resolution_clock::now();
	  int dtime=chrono::duration_cast<chrono::nanoseconds>(stop - start).count();
	  performance_data->time+=dtime;
	  if(performance_data->count%performance_data->every==0)
	    cout<<"transformState "<<performance_data->time/performance_data->count<<" ns per eval."<<endl;
	}
	return transformed;
      } else {
	cout<<"stateSpaceInvolution:: No working state space."<<endl;
	exit(-1);
      }
    } else {
      cout<<"stateSpaceInvolution:: No transform registered."<<endl;
      exit(-1);
    }
  };
  virtual double jacobian(const state &s)const override{
    if(transformState_registered){
      if(jacobian_registered){
	if(have_working_space)
	  return (*user_jacobian)(user_object,s,randoms);
	else {
	  cout<<"stateSpaceInvolution:: No working state space."<<endl;
	  exit(-1);
	}
      }
      else return 1; //trivial by default
    } else {
      cout<<"stateSpaceInvolution:: No transform registered."<<endl;
      exit(-1);
    }
  };
  virtual void defWorkingStateSpace(const stateSpace &sp){
    if(transformState_registered){
      if(defWorkingStateSpace_registered)(*user_defWorkingStateSpace)(user_object,sp);
      have_working_space=true;
      return;
    }
  };
  virtual double test_involution(const state &s, double verbose_lev=0, ostream &out=cout){
    //should return 0 if the function is indeed its own inverse and jacobian(x)*jacobian(f(x))==1; 
    state image=transformState(s);
    double jac=jacobian(s);
    //For random seeded involutions, drawing on nrand random doubles in (-1,1),
    //The inverse transformation should be realized by the negative of the
    //random vector value.  In case a point-wise self inverse map is applied,
    //it should depend only the absolute value of the random number.
    //Here we flip the sign on the random vector for the inverse transf.
    for(auto &x : randoms)x*=-1;
    state image2=transformState(image);
    double jac2=jacobian(image);
    for(auto &x : randoms)x*=-1;//Flip the sign back restore the original random vector
    //state diff = s.add(image2.scalar_mult(-1));
    //double err=diff.innerprod(diff);
    double err=s.dist2(image2);
    double jacdiff=jac*jac2-1;
    double result=err+jacdiff*jacdiff;
    if(verbose_lev<0 or result*verbose_lev>1){
      out<<"test_involution: s="<<s.get_string()<<endl;
      out<<"                s'="<<image.get_string()<<endl;
      out<<"               s''="<<image2.get_string()<<"  err="<<err<<endl;
      out<<"   J="<<jac<<endl;
      out<<"  J'="<<jac2<<"  err="<<jacdiff*jacdiff<<endl;
      
      if(nrand>0){
	out<<"randoms: [";
	for(int i=0;i<nrand;i++){
	  if(i>0)cout<<",";
	  out<<randoms[i];
	}
	out<<"]"<<endl;
      }
    }
    return result;
  };

  ///This section provides data and functions for a minimal interface not requiring inheritance
  ///To apply the base class, the user must at minimum provide a transformState() function and a
  ///jacobian function (if the jacobian is non-trival) using the register_transformState() and
  ///register_jacobian hooks.  If that function requires some reference data, then the
  ///user must also provide a pointer to reference object (with access to the data) using the
  ///function register_reference_object().  The user has the option to register a
  ///defWorkingStateSpace() function to preset the location of data within the state object.
private:
  state (*user_transformState)(void *object, const state &s, const vector<double> &randoms);
  double (*user_jacobian)(void *object, const state &s, const vector<double> &randoms);
  void (*user_defWorkingStateSpace)(void *object, const stateSpace &d);
  void * user_object;
  vector<double>randoms;
  bool transformState_registered,jacobian_registered,defWorkingStateSpace_registered;
  friend class stateSpace;
public:
  void register_reference_object(void *object){
    user_object=object;};    
  void register_transformState(state (*function)(void *object, const state &s, const vector<double> &randoms)){
    user_transformState=function;
    transformState_registered=true;
    have_working_space=true;//Assume the working space is known unless defWorkingStateSpace is registered
  };
  void register_jacobian(double (*function)(void *object, const state &s, const vector<double> &randoms)){
    user_jacobian=function;
    jacobian_registered=true;
  };
  void register_defWorkingStateSpace(void (*function)(void *object, const stateSpace &s)){
    user_defWorkingStateSpace=function;
    defWorkingStateSpace_registered=true;
    have_working_space=false;
  };
};


#endif

