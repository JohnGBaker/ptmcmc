///General structures for Bayesian analysis
///
///state objects are tied to a stateSpace object which specifies to domain.
///boundary objects allow specification of the domain bounraies in each dimension.
///John G Baker - NASA-GSFC (2013-2015)

#ifndef PTMCMC_BAYESIAN_HH
#define PTMCMC_BAYESIAN_HH
#include <valarray>
#include <vector>
#include <sstream>
#include <cmath>
#include <iostream>
//#include <utility>
#include <memory>
#include "ProbabilityDist.h"
#include "newran.h"
#include "options.hh"

using namespace std;

typedef unsigned int uint;

extern shared_ptr<Random>globalRNG;

class probability_function;

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

/// Base class for defining likelihoods/priors/etc (nonnormalized)
/// Default version is flat.
class probability_function {
protected:
  stateSpace *space;
public:
  virtual ~probability_function(){};
  probability_function(stateSpace *space):space(space){};
  virtual double evaluate(state &s){return exp(evaluate_log(s));};
  virtual double evaluate_log(state &s){return 0;};
  virtual string show(){return "UnspecifiedProb()";};
};

// A general (abstract) class for defining eg priors/etc 
// from which we can draw samples.
class sampleable_probability_function: public probability_function{
  ///Sometimes we need to know the largest relevant dimension
  void fail(){cout<<"sampleable_probability_function: This should be used strictly as a parent class, and it's virtual functions should be overridden in a base clas object.  Instances of this parent class should not be referenced."<<endl;exit(1);};
protected:
  unsigned int dim;
public:
  virtual ~sampleable_probability_function(){};
  sampleable_probability_function(stateSpace *space):probability_function(space){};
  virtual state drawSample(Random &rng){fail();return state();}
  virtual double evaluate(state &s){fail();return -1;};
  virtual double evaluate_log(state &s){fail();return -INFINITY;};
  virtual int getDim(){return dim;};
  virtual string show(){return "UnspecifiedSampleableProb()";};
};

///Interface class for bayesian signal data. This is some kind of compound data.
///We begin with only what we need for ptmcmc, that we can write the signal
class bayes_signal {
public:
  virtual int size()=0;
  virtual void write(ostream &out,state &st, int nsamples=-1, double tstart=0, double tend=0)=0;
};

///Base class for defining a Bayesian sampler object
///
///To begin with the only option is for MCMC sampling, though we expect soon to add a multinest option.
class bayes_sampler : public Optioned {
public:
  bayes_sampler(){};
  virtual bayes_sampler * clone()=0;
  virtual int initialize()=0;
  virtual int run(const string & base, int ic=0)=0;
  ///This is too specific for a generic interface, but we're building on what we had before...
  virtual int analyze(const string & base, int ic, int Nsigma, int Nbest, bayes_signal &data, double tfinestart, double tfineend)=0;
};

///Bayes class for a likelihood function object
///
///The interface here is probably not everything we want, but is enough for what was already in the main function.
class bayes_likelihood : public probability_function {
public:
  bayes_likelihood(stateSpace *sp):probability_function(sp){};
  virtual void reset()=0;
  virtual double bestPost()=0;
  virtual state bestState()=0;
};

#endif

