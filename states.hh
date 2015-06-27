///State and state space structures for Bayesian analysis
///
///state objects are tied to a stateSpace object which specifies to domain.
///boundary objects allow specification of the domain bounraies in each dimension.
///John G Baker - NASA-GSFC (2013-2015)

#ifndef PTMCMC_STATES_HH
#define PTMCMC_STATES_HH
#include <valarray>
#include <vector>
#include <sstream>
//#include <cmath>
#include <iostream>
//#include <utility>
//#include <memory>

using namespace std;

typedef unsigned int uint;


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
  boundary(int lowertype=open,int uppertype=open,double min=-INFINITY,double max=INFINITY):lowertype(lowertype),uppertype(uppertype),xmin(min),xmax(max){};
  ///enforce boundary condition. If consistent enforcement was achieved return true.
  bool enforce(double &x);
  ///Show structural info
  string show();
};

/// State space class allows reference to overall limits and structure of the space.
/// Can provide support for boundary conditions, maybe special flows that can be used in proposals...
/// Should inner product be defined here?  probably...
class stateSpace {
  const int dim;
  valarray<boundary> bounds;
  valarray<string> names;
  bool have_names;
public:
  stateSpace(int dim=0):dim(dim){
    bounds.resize(dim,boundary());//default to default boundaries (ie allow all reals)
    have_names=false;    
  };
  void set_bound(int i, const boundary &b){
    if(i<dim)bounds[i]=b;
    else{
      cout<<"stateSpace::set_bound: Index out of range, "<<i<<">="<<dim<<"."<<endl;
      exit(1);
    }
  };      
  void set_names(string stringnames[]){
    names.resize(dim,"");
    for(uint i=0;i<dim;i++)names[i]=stringnames[i];
    have_names=true;
  };
  string get_name(int i){
    if(have_names&&i<dim)return names[i];
    else return "[unnamed]";
  };      
  bool enforce(valarray<double> &params);
  ///Show structural info
  string show();
};
      
///Class for holding and manipulating bayesian parameter states
class state {
  bool valid;
  stateSpace *space;
  valarray<double> params;
  void enforce();
public:
  //Need assignment operator since default valarray assignment is problematic
  const state& operator=(const state model){space=model.space;valid=model.valid,params.resize(model.size(),0);params=model.params;return *this;};
  state(stateSpace *space=nullptr,int n=0);
  state(stateSpace *sp, const valarray<double>&array);
  int size()const{return params.size();}
  //some algorithms rely on using the states as a vector space
  virtual state add(const state &other)const;
  virtual state scalar_mult(double x)const;
  ///For some applications it is necessary to have an inner product on the state space. Probably should move this out to stateSpace.
  virtual double innerprod(state other)const;
  virtual string get_string()const;
  virtual void get_params_array(valarray<double> &outarray)const{
    //outarray.resize(params.size());
    //for(size_t i=0;i<size();i++)outarray[i]=params[i];
      outarray=std::move(params);
    return;
  }
  virtual valarray<double> get_params()const{return params;};
  virtual vector<double> get_params_vector(){vector<double> v;v.assign(begin(params),end(params));return v;};
  stateSpace * getSpace(){return space;};
  ///Show param info
  string show();
  bool invalid()const{return !valid;};
};

#endif

