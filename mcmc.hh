///States for MCMC
///
///state objects are tied to a stateSpace object which specifies to domain.
///boundary objects allow specification of the domain bounraies in each dimension.
///John G Baker - NASA-GSFC (2013-2014)

#ifndef PTMCMC_MCMC_HH
#define PTMCMC_MCMC_HH
#include <valarray>
#include <vector>
#include <sstream>
#include <cmath>
#include <iostream>
//#include <utility>
#include <memory>
#include "ProbabilityDist.h"
#include "newran.h"

using namespace std;

typedef unsigned int uint;

extern shared_ptr<Random>globalRNG;

class proposal_distribution;
class probability_function;
class chain;

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
      
///Class for holding and manipulating markov chain states
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


//A class for defining proposal distributions.
//Useful proposal distributions often reference a chain.
class proposal_distribution {
protected:
  double log_hastings;
  int last_type;
public:
  virtual ~proposal_distribution(){};
  proposal_distribution(){log_hastings=0;last_type=0;};
  virtual double log_hastings_ratio(){return log_hastings;};//(proposal part of Hasting ratio for most recent draw;
  virtual void set_chain(chain *c){};//Not always needed.
  //virtual state draw(state &s,Random &rng){return s;};//The base class won't be useful.
  virtual state draw(state &s,chain *caller){return s;};//The base class won't be useful.
  ///Some proposals require some set-up, such as a chain of sufficient length.
  virtual bool is_ready() {return true;};
  virtual proposal_distribution* clone()const{return new proposal_distribution(*this);};
  virtual string show(){return "UnspecifiedProposal()";};
  ///Return type of draw (where that is relevant, otherwise 0)
  virtual int type(){return last_type;}
  virtual bool support_mixing(){return false;}
};

// A generalized chain has results like a chain, but we specify
// anything about *how* those results are produced.  Standard analysis can be applied. 
class chain {
  //TODO?
  //autocorrelation
  //convergence,MAP,...
  static int idcount;
protected:
  int id;
  int Nsize,Ninit; //Maybe move this history stuff to a subclass
  int Nfrozen;
  int dim;
  shared_ptr<Random> rng; //Random number generator for this chain. Each chain has its own so that the results can be threading invariant
  //This function is defined for just for sorting below
  //static bool AgtB(const pair<double,double> & A,const pair<double,double>&B){return A.second>B.second;};
  double get_uniform(){
    double x=rng->Next(); //pick a number
    //rngout<<x<<" 1"<<endl;
    //#pragma omp critical
    //cout<<id<<":"<<x<<" 1 rng="<<rng<<endl;
    return x;
  };
public:
  virtual ~chain(){};
  chain():
    rng(new MotherOfAll(ProbabilityDist::getPRNG()->Next()))
    //rng(ProbabilityDist::getPRNG())//This should recover identical results to pre_omp version...?
    //rng(globalRNG)
  {
    id=idcount;idcount++;
    //cout<<"chain::chain():Set  id="<<id<<" one of "<<idcount<<" instance(s)."<<endl;//debug
    //cout<<"chain::chain():Set rng="<<rng.get()<<" one of "<<rng.use_count()<<" instance(s)."<<endl;//debug
    //Each chain has its own pseudo (newran) random number generator seeded with a seed drawn from the master PRNG
    //As long as the chains are created in a fixed order, this should allow threading independence of the result.
    Nfrozen=-1;
  };
  virtual shared_ptr<Random> getPRNG(){return rng;};
  virtual string show(){
    ostringstream s;
    s<<"chain(id="<<id<<"size="<<Nsize<<")\n";
    return s.str();
  };
  virtual string status(){
    ostringstream s;
    s<<"chain(id="<<id<<"N="<<Nsize<<"):"<<this->getState().get_string();
    return s.str();
  };
  virtual void reserve(int nmore){};//If you know how long you are going to run, it can be more efficient to reserve up front, rather than resizing ever step
  virtual int capacity(){return -1;};//If you know how long you are going to run, it can be more efficient to reserve up front, rather than resizing ever step
  virtual int size(){if(Nfrozen==-1)return Nsize;else return Nfrozen;};
  virtual state getState(int elem=-1,bool raw_indexing=false){cout<<"chain::getState: Warning. You've called a dummy function."<<endl;state s;return s;};//base class is not useful
  virtual double getLogPost(int elem=-1,bool raw_indexing=false){cout<<"chain::getLogPost: Warning. You've called a dummy function."<<endl;return 0;};//get log posterior from a chain sample
  virtual double getLogLike(int elem=-1,bool raw_indexing=false){cout<<"chain::getLogLike: Warning. You've called a dummy function."<<endl;return 0;};//get log prior from a chain sample
  //virtual double getLogPrior(int elem=-1,bool raw_indexing=false){cout<<"chain::getLogPrior: Warning. You've called a dummy function."<<endl;return 0;};//get log prior from a chain sample
  virtual int getDim(){return dim;};
  ///Test the expectation value for some test function.
  ///This could be, for example:
  ///1. An expected parameter value (test_func should return the parameter value)
  ///2. Probability of being in some bin region of param space (test_func returns 1 iff in that region)
  virtual double  Pcond(bool (*test_func)(state s)){return dim;};
  virtual int i_after_burn(int nburn=0){return nburn;}
  virtual void inNsigma(int Nsigma,vector<int> & indicies,int nburn=0){
    //cout<<" inNsigma:this="<<this->show()<<endl;
    int ncount=size()-this->i_after_burn(nburn);
    //cout<<"size="<<size()<<"   nburn="<<nburn<<"   i_after_burn="<<this->i_after_burn(nburn)<<endl;
    int inNstd_count=(int)(erf(Nsigma/sqrt(2.))*ncount);
    vector<pair<double,double> > lpmap;
    //double zero[2]={0,0};
    //lpmap.resize(ncount,zero);
    //cout<<"ncount="<<ncount<<"   inNstd_count="<<inNstd_count<<endl;
    lpmap.resize(ncount);
    for(uint i=0;i<ncount;i++){
      int idx=i+this->i_after_burn(nburn);
      //cout<<idx<<" state:"<<this->getState(idx,true).get_string()<<endl;
      lpmap[i]=make_pair(-this->getLogPost(idx,true),idx);
    }
    //sort(lpmap.begin(),lpmap.end(),chain::AgtB);
    sort(lpmap.begin(),lpmap.end());
    indicies.resize(inNstd_count);
    for(uint i=0;i<inNstd_count;i++){
      indicies[i]=lpmap[i].second;
      //cout<<"  :"<<lpmap[i].first<<" "<<lpmap[i].second<<endl;
    }
    return;
  };
  virtual void set_proposal(proposal_distribution &proposal){
    cout<<"chain::step: No base-class set_proposal operation defined!"<<endl;
    exit(1);
  };
  virtual void step(){
    cout<<"chain::step: No base-class step() operation defined!"<<endl;
    exit(1);
  };
  virtual void dumpChain(ostream &os,int Nburn=0,int ievery=1){
    cout<<"chain::step: No base-class dumpChain() operation defined!"<<endl;
    exit(1);
  };
  //Interface for optional features of derived clases.
  ///This function should be overloaded to indicate the number of subchains in a hierarchical chain;
  virtual int multiplicity(){return 1;};
  ///subchains should be indexed from 0 to multiplicity-1
  virtual chain* subchain(int index){return this;};
  ///return the inverse temperature for this chain
  virtual double invTemp(){return 1.0;};
  ///Temporarily freeze the length of the chain as reported by size()
  void history_freeze(){Nfrozen=Nsize;};
  void history_thaw(){Nfrozen=-1;};
  int get_id(){return id;};
};

#endif

