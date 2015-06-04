#ifndef PTMCMCS_HH
#define PTMCMCS_HH
#include <valarray>
#include <vector>
#include <sstream>
#include <cmath>
#include <iostream>
#include <utility>
#include <memory>
#include "include/ProbabilityDist.h"
//#include "include/newran.h"

using namespace std;

typedef unsigned int uint;

//extern ofstream rngout;
extern shared_ptr<Random>globalRNG;

//Code for parallel-tempering MCMC (John's study)
//
//Basics:  
// We will conduct Nc MCMC chains each operating at differnent temperatures.
// Some parallel tempering probability distribution will control the frequency
// with which the neighboring chains interchange.
// 
// We assume some likelihood function is defined and that is a function of Np
// parameters.
//
// We will provide a set of basic uninformed proposal distribution options.
//
// A good general reference:http://www.mcmchandbook.net
//
// This is version 2. which has added stateSpace support, 


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
  bool enforce(double &x){
    //cout<<"boundary::enforce: testing value "<<x<<" in range "<<show()<<endl;//debug
    //check wrapping first
    if(lowertype==wrap^uppertype==wrap){//can't have just one wrap 
      cout<<"boundary::enforce: Inconsistent wrap."<<endl;
      return false;
    } else if (lowertype==wrap) {
      //cout<<"boundary::enforce: testing value "<<x<<" in range "<<show()<<endl;//debug
      double width=xmax-xmin;
      if(width<=0){
	cout<<"  boundary::enforce: Wrap: Negative width."<<endl;//debug
	return false;
      }
      double xt=fmod(x-xmin,width);
      if(xt<0)xt+=width;//fmod is stupid and finds the remainder after rounding *toward* zero.
      x=xmin+xt;//Note label range is open near xmax, closed to xmin
      //cout<<"  Wrap: OK."<<endl;//debug
      return true;
    } 
    //next check reflection
    if (lowertype==reflect&&uppertype==reflect){
      //double reflection is like wrapping, then folding
      double halfwidth=xmax-xmin;
      if(halfwidth<=0){
	return false;
	cout<<"boundary::enforce:  DoubleReflect: Negative width."<<endl;//debug
      }
      double width=2*halfwidth;
      double xt=fmod(x-xmin,width);
      if(xt<0)xt+=width;
      if(xt>=halfwidth)xt=halfwidth-xt;
      x=xmin+xt;
      cout<<"  DoubleReflect hasn't been tested...."<<endl;//debug
      return true;
    }
    if(lowertype==reflect&&x<xmin)x=xmin+(xmin-x);
    else if(uppertype==reflect&&x>xmax)x=xmax-(x-xmax);
    if(lowertype==limit&&x<xmin){
      //cout<<"  Lower limit failure."<<endl;//debug
      return false;
    }
    if(uppertype==limit&&x<xmax){
      //cout<<"  Upper limit failure."<<endl;//debug
      return false;
    };
    //cout<<"  All good!."<<endl;//debug
    return true;
  };
  ///Show structural info
  string show(){
    ostringstream s;
    if(lowertype==wrap)s<<"w["<<xmin<<","<<xmax<<")w";
    else {
      if(lowertype==reflect)s<<"R[";
      else if(lowertype==limit)s<<"[";
      else s<<"(";
      s<<xmin<<","<<xmax;
      if(uppertype==reflect)s<<"]R";
      else if(uppertype==limit)s<<"]";
      else s<<")";
    }
    return s.str();
  };
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
    //cout<<"new stateSpace with dim="<<dim<<endl;
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
  bool enforce(valarray<double> &params){
    if(params.size()!=dim){
      cout<<"stateSpace::enforce:  Dimension error.  Expected "<<dim<<" params, but given "<<params.size()<<"."<<endl;
      exit(1);
    }
    for(uint i=0;i<dim;i++){
      //cout<<"stateSpace::enforce: testing parameter "<<i<<endl;//debug
      if(!bounds[i].enforce(params[i])){
	//cout<<"        FAILED."<<endl;
	return false;
      }
    }
    //cout<<"        PASSED."<<endl;
    return true;
  };
  ///Show structural info
  string show(){
    ostringstream s;
    s<<"StateSpace:(dim="<<dim<<")\n";
    for(uint i=0;i<dim;i++){
      s<<"  "<<get_name(i)<<" in "<<bounds[i].show()<<"\n";
    }
    return s.str();
  };
};
      
///Class for holding and manipulating markov chain states
class state {
  bool valid;
  stateSpace *space;
  void enforce(){
    if(!space)valid=false;
    if(!valid)return;
    valid=space->enforce(params);
    //cout<<"state::enforce:State was "<<(valid?"":"not ")<<"valid."<<endl;//debug
  };
  valarray<double> params;
public:
  //Need assignment operator since default valarray assignment is problematic
  const state& operator=(const state model){space=model.space;valid=model.valid,params.resize(model.size(),0);params=model.params;return *this;};
  state(stateSpace *space=nullptr,int n=0):space(space){
    params.resize(n,0);
    valid=false;
    //cout<<"state::state():space="<<this->space<<endl;    
    if(space){
      valid=true;
      enforce();
    }
  };
  state(stateSpace *sp, const valarray<double>&array):space(sp),params(array){
    //cout<<"state::state(stuff):space="<<this->space<<endl;
    valid=false;
    if(space)valid=true;
    enforce();
  };
  int size()const{return params.size();}
  //some algorithms rely on using the states as a vector space
  virtual state add(const state &other)const{
    //here we only require that the result is valid.
    state result(space,size());
    if(other.size()!=size()){
      cout<<"state::add: Sizes mismatch. ("<<size()<<"!="<<other.size()<<")\n";
      exit(1);
    }
    for(uint i=0;i<size();i++)result.params[i]=params[i]+other.params[i];
    result.enforce();
    return result;
  };
  virtual state scalar_mult(double x)const{
    //we only require that the result is valid
    state result(space,size());
    for(uint i=0;i<size();i++)result.params[i]=params[i]*x;
    result.enforce();
    return result;
  };
  ///For some applications it is necessary to have an inner product on the state space. Probably should move this out to stateSpace.
  virtual double innerprod(state other)const{
    if(!(valid&&other.valid))return NAN;
    double result=0;
    if(other.size()!=size()){
      cout<<"state::innerprod: sizes mismatch.\n";
      exit(1);
    }
    for(uint i=0;i<size();i++)result+=params[i]*other.params[i];
    return result;
  };
  virtual string get_string()const{
    ostringstream s;
    int n=params.size();
    //cout<<"n="<<n<<endl;//debug
    for(int i=0;i<n-1;i++){
      //cout<<"i="<<i<<endl;
      s<<params[i]<<", ";
    }
    if(n>0)s<<params[n-1];
    else s<<"<empty>";
    if(!valid)s<<" [INVALID]";
    return s.str();
  };
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
  string show(){
    ostringstream s;
    s<<"(\n";
    for(uint i=0;i<params.size();i++)s<<"  "<<(space?space->get_name(i):"[???]")<<" = "<<params[i]<<"\n";
    s<<")\n";
    return s.str();
  };
  bool invalid()const{return !valid;};
};

class proposal_distribution;

// A generalized chain has results like a chain, but we specify
// anything about *how* those results are produced.  Standard analysis can be applied. 
class chain {
  //autocorrelation
  //MAP,...
  //P(f(state)>x)
  static int idcount;
protected:
  int id;
  int Nsize; //Maybe move this history stuff to a subclass
  int dim;
  shared_ptr<Random> rng; //Random number generator for this chain. Each chain has its own so that the results can be threading invariant
  //This function is defined for just for sorting below
  //static bool AgtB(const pair<double,double> & A,const pair<double,double>&B){return A.second>B.second;};
  double get_uniform(){
    double x=rng->Next(); //pick a number
    //rngout<<x<<" 1"<<endl;
    return x;
  };
public:
  chain():
    //rng(new MotherOfAll(ProbabilityDist::getPRNG()->Next()))
    //rng(ProbabilityDist::getPRNG())//This should recover identical results to pre_omp version...?
    rng(globalRNG)
  {
    id=idcount;idcount++;
    //cout<<"chain::chain():Set rng="<<rng.get()<<" one of "<<rng.use_count()<<" instance(s)."<<endl;//debug
    //Each chain has its own pseudo (newran) random number generator seeded with a seed drawn from the master PRNG
    //As long as the chains are created in a fixed order, this should allow threading independence of the result.
  };
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
  //virtual state getState(int elem=-1){return state();};//Base class is not useful;
  virtual void reserve(int nmore){};//If you know how long you are going to run, it can be more efficient to reserve up front, rather than resizing ever step
  virtual int capacity(){return -1;};//If you know how long you are going to run, it can be more efficient to reserve up front, rather than resizing ever step
  virtual int size(){return Nsize;};
  virtual state getState(int elem=-1){state s;return s;};//grab a state, or current state;
  virtual double getLogPost(int elem=-1){return 0;};//get log posterior from a chain sample
  virtual int getDim(){return dim;};
  ///Test the expectation value for some test function.
  ///This could be, for example:
  ///1. An expected parameter value (test_func should return the parameter value)
  ///2. Probability of being in some bin region of param space (test_func returns 1 iff in that region)
  virtual double  Pcond(bool (*test_func)(state s)){return dim;};
  virtual int i_after_burn(int nburn=0){return nburn;}
  virtual void inNsigma(int Nsigma,vector<int> & indicies,int nburn=0){
    //cout<<" inNsigma:this="<<this<<endl;
    int ncount=size()-i_after_burn(nburn);
    //cout<<"size="<<size()<<"   nburn="<<nburn<<"   i_after_burn="<<i_after_burn(nburn)<<endl;
    int inNstd_count=(int)(erf(Nsigma/sqrt(2.))*ncount);
    vector<pair<double,double> > lpmap;
    //double zero[2]={0,0};
    //lpmap.resize(ncount,zero);
    //cout<<"ncount="<<ncount<<"   inNstd_count="<<inNstd_count<<endl;
    lpmap.resize(ncount);
    for(uint i=0;i<ncount;i++){
      int idx=i+i_after_burn(nburn);
      lpmap[i]=make_pair(-getLogPost(idx),idx);
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
};
int chain::idcount=0;//this will fail if the header is used in two files...

// A general class for defining likelihoods/priors/etc (nonnormalized)
// Default version is flat.
class probability_function {
protected:
  stateSpace *space;
public:
  probability_function(stateSpace *space):space(space){};
  virtual double evaluate(state &s){return exp(evaluate_log(s));};
  virtual double evaluate_log(state &s){return 0;};
  virtual string show(){return "UnspecifiedProb()";};
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
  virtual state draw(state &s,Random &rng){return s;};//The base class won't be useful.
  ///Some proposals require some set-up, such as a chain of sufficient length.
  virtual bool is_ready() {return true;};
  virtual proposal_distribution* clone()const{return new proposal_distribution(*this);};
  virtual string show(){return "UnspecifiedProposal()";};
  ///Return type of draw (where that is relevant, otherwise 0)
  virtual int type(){return last_type;}
};

//********* DERIVED CLASSES *************

//** PROBABILITY FUNCTIONs *******

// A general (abstract) class for defining eg priors/etc 
// from which we can draw samples.
class sampleable_probability_function: public probability_function{
  ///Sometimes we need to know the largest relevant dimension
  void fail(){cout<<"sampleable_probability_function: This should be used strictly as a parent class, and it's virtual functions should be overridden in a base clas object.  Instances of this parent class should not be referenced."<<endl;exit(1);};
protected:
  unsigned int dim;
public:
  sampleable_probability_function(stateSpace *space):probability_function(space){};
  virtual state drawSample(Random &rng){fail();return state();}
  virtual double evaluate(state &s){fail();return -1;};
  virtual double evaluate_log(state &s){fail();return -INFINITY;};
  virtual int getDim(){return dim;};
  virtual string show(){return "UnspecifiedSampleableProb()";};
};


// A class for defining gaussian likelihoods/priors/etc
// from which we can draw samples based on the ProbabilityDist class.
// unit normal range on each parameter.
class gaussian_dist_product: public sampleable_probability_function{
  valarray<double> x0s;
  valarray<double> sigmas;
  vector<ProbabilityDist*> dists;
public:
  gaussian_dist_product(stateSpace *space,unsigned int N=1):sampleable_probability_function(space){
    dim=N;
    x0s.resize(N,0);
    sigmas.resize(N,1);
    dists.resize(dim);
    for(size_t i=0;i<dim;i++)dists[i]=new GaussianDist(x0s[i],sigmas[i]);
  };
  gaussian_dist_product(stateSpace *space, valarray<double>&x0s,valarray<double>&sigmas):x0s(x0s),sigmas(sigmas),sampleable_probability_function(space){
    dim=x0s.size();
    if(dim!=sigmas.size()){
      cout<<"gaussian_dist_product(constructor): Array sizes mismatch.\n";
      exit(1);
    }
    dists.resize(dim);
    for(size_t i=0;i<dim;i++){
      dists[i]=new GaussianDist(x0s[i],sigmas[i]);
    }
  };
  virtual ~gaussian_dist_product(){
    while(dists.size()>0){
      delete dists.back();
      dists.pop_back();
    }
  };
  state drawSample(Random &rng){
    valarray<double> v(dim);    
    //cout<<"gaussian_dist_product::drawSample: drawing vector of len "<<dim<<"for space at"<<space<<endl;//debug
    for(uint i=0;i<dim;i++){
      double number=dists[i]->draw(&rng);
      v[i]=number;
      //cout<<"  "<<number;//debug
    }
    //cout<<endl;//debug
    return state(space,v);
  };
  double evaluate(state &s){
    if(s.invalid())return 0;
    if(dim!=s.size()){
      cout<<"gaussian_dist_product:evaluate: State size mismatch.\n";
      exit(1);
    }
    double result=1;
    vector<double> pars=s.get_params_vector();
    for(uint i=0;i<dim;i++)result*=dists[i]->pdf(pars[i]);
    return result;
  };
  double evaluate_log(state &s){return log(evaluate(s));};
  string show(){
    ostringstream s;
    s<<"GaussianSampleableProb(\n";
    for( int i=0;i<dim;i++)
      s<<"  "<<(space?space->get_name(i):"[???]")<<" from "<<dists[i]->show()<<"\n";
    s<<")\n";
    return s.str();    
  };
};

// An example class for defining likelihoods/priors/etc
// from which we can draw samples based on the ProbabilityDist class.
// Default version is flat within unit range on each parameter.
class uniform_dist_product: public sampleable_probability_function{
  valarray<double> min;
  valarray<double> max;
  vector<ProbabilityDist*> dists;
public:
  uniform_dist_product(stateSpace *space , int N=1):sampleable_probability_function(space){
    dim=N;
    min.resize(N,0);
    max.resize(N,1);
    dists.resize(dim);
    for(uint i=0;i<dim;i++)dists[i]=new UniformIntervalDist(min[i],max[i]);
  };
  uniform_dist_product(stateSpace *space,valarray<double>&min_corner,valarray<double>&max_corner):min(min_corner),max(max_corner),sampleable_probability_function(space){
    dim=min.size();
    if(dim!=max.size()){
      cout<<"prob_dist_product(constructor): Array sizes mismatch.\n";
      exit(1);
    }
    dists.resize(dim);
    for(uint i=0;i<dim;i++)dists[i]=new UniformIntervalDist(min[i],max[i]);
  };
  virtual ~uniform_dist_product(){
    while(dists.size()>0){
      delete dists.back();
      dists.pop_back();
    }
  };
  state drawSample(Random &rng){
    valarray<double> v(dim);
    for(uint i=0;i<dim;i++){
      double number=dists[i]->draw(&rng);
      v[i]=number;
    }
    return state(space,v);
  };
  double evaluate(state &s){
    if(s.invalid())return 0;
    if(dim!=s.size()){
      cout<<"uniform_dist_product:evaluate: State size mismatch.\n";
      exit(1);
    }
    double result=1;
    vector<double> pars=s.get_params_vector();
    for(uint i=0;i<dim;i++)result*=dists[i]->pdf(pars[i]);
    return result;
  };
  double evaluate_log(state &s){return log(evaluate(s));};
  string show(){
    ostringstream s;
    s<<"UniformSampleableProb(\n";
    for( int i=0;i<dim;i++)
      s<<"  "<<(space?space->get_name(i):"[???]")<<" from "<<dists[i]->show()<<"\n";
    s<<")\n";
    return s.str();    
  };
};


// A class for defining likelihoods/priors/etc from an independent mix of gaussian and flat priors
// from which we can draw samples based on the ProbabilityDist class.
// unit normal range on each parameter.
class mixed_dist_product: public sampleable_probability_function{
  valarray<double> centers;  //like x0s of gaussian, or (min+max)/2 of uniform.
  valarray<double> halfwidths; //like sigmas of gaussian, or (max-min)/2 of uniform.
  valarray<int> types; //uniform or gaussian

  vector<ProbabilityDist*> dists;
public:
  //static const int mixed_dist_product::uniform=1;
  //static const int mixed_dist_product::gaussian=1;
  static const int uniform=1;
  static const int gaussian=2;
  static const int polar=2;
  mixed_dist_product(stateSpace *space,unsigned int N=1):sampleable_probability_function(space){
    //cout<<"mixed_dist_product::mixed_dist_product("<<space<<","<<N<<"): Constructing this="<<this<<endl;//debug;
    //cout<<"mixed_dist_product:Creating with space="<<space<<endl;
    dim=N;
    centers.resize(N,0);
    types.resize(N,0);
    halfwidths.resize(N,1);
    dists.resize(dim);
    for(size_t i=0;i<dim;i++){
      dists[i]=new GaussianDist(centers[i],halfwidths[i]);
      types[i]=gaussian;
    }
  };
  mixed_dist_product(stateSpace *space,valarray<int> &types,valarray<double>&centers,valarray<double>&halfwidths):types(types),centers(centers),halfwidths(halfwidths),sampleable_probability_function(space){
    //cout<<"mixed_dist_product::mixed_dist_product("<<space<<",types,centers,halfwidths): Constructing this="<<this<<endl;//debug;
    dim=centers.size();
    if(dim!=halfwidths.size()||dim!=types.size()){
      cout<<"mixed_dist_product(constructor): Array sizes mismatch.\n";
      exit(1);
    }
    dists.resize(dim);
    for(size_t i=0;i<dim;i++){
      if(types[i]==uniform)
	dists[i]=new UniformIntervalDist(centers[i]-halfwidths[i],centers[i]+halfwidths[i]);
      else if(types[i]==gaussian)
	dists[i]=new GaussianDist(centers[i],halfwidths[i]);
      else if(types[i]==polar)
	dists[i]=new UniformPolarDist();//note centers,halfwidths are ignored
      else {
	cout<<"mixed_dist_product(constructor): Unrecognized type. types["<<i<<"]="<<types[i]<<endl;
	exit(1);
      }
    }
  };
  virtual ~mixed_dist_product(){
    while(dists.size()>0){
      delete dists.back();
      dists.pop_back();
    }
  };
  state drawSample(Random &rng){
    valarray<double> v(dim);
    for(uint i=0;i<dim;i++){
      double number=dists[i]->draw(&rng);
      v[i]=number;
    }
    state s(space,v);
    //cout<<"mixed_dist_product::drawSample(): drew state:"<<s.show()<<endl;//debug
    return s; 
  };
  double evaluate(state &s){
    if(s.invalid())return 0;
    if(dim!=s.size()){
      cout<<"mixed_dist_product:evaluate: State size mismatch.\n";
      exit(1);
    }
    double result=1;
    vector<double> pars=s.get_params_vector();
    for(uint i=0;i<dim;i++){
      double pdfi=dists[i]->pdf(pars[i]);
      //cout<<"subspace prior for "<<s.getSpace()->get_name(i)<<" -> "<<pdfi<<endl;
      result*=pdfi;
    }
    return result;
  };
  double evaluate_log(state &s){return log(evaluate(s));};
  string show(){
    ostringstream s;
    s<<"SampleableProb(\n";
    for( int i=0;i<dim;i++)
      s<<"  "<<(space?space->get_name(i):"[???]")<<" from "<<dists[i]->show()<<"\n";
    s<<")\n";
    return s.str();    
  };
};

//Draw samples from a chain
//This base version provides support only on the chain points themselves (no interpolation) interpolative variants (eg using approximate nearest neighbor approach) could be developed if needed.  With chain support this can be used as prior as long as the proposal distribution is strictly compatible (in particular only proposing points in the reference chain).
class chain_distribution: public sampleable_probability_function{
  chain & c;
  int istart;
  int last_sample;
public:
  chain_distribution(chain &c, int istart=0):c(c),istart(istart),sampleable_probability_function(c.getState().getSpace()){
    last_sample=-1;
  };
  state drawSample(Random &rng){
    double xrand=rng.Next();
    //rngout<<xrand<<" 2"<<endl;

    int i=(int)(xrand*(c.size()-istart))+istart; //pick a chain element
    //cout<<"chain_distribution:drew sample "<<i<<" = "<<c.getState(i).get_string()<<endl;
    //cout<<" &c="<<&c<<endl;
    last_sample=i;
    return c.getState(i);
  };
  double evaluate_log(state &s){
    if(s.invalid())return -INFINITY;
    state last_state;
    double diff=0;
    //cout<<"chain_distribution: last_sample="<<last_sample<<", evaluating log for state "<<s.get_string()<<endl;
    if(last_sample>=0 and last_sample<c.size()){//verify state equivalence
      last_state=c.getState(last_sample);
      state sdiff=s.scalar_mult(-1).add(last_state);
      diff=sdiff.innerprod(sdiff);
    } 
    if(last_sample<0||last_sample>=c.size()||diff>0){
      cout<<"chain_distribution:evaluate_log: Cannot evaluate on arbitrary state point. Expected sample("<<last_sample<<")="<<last_state.get_string()<<"!="<<s.get_string()<<".\n";
      exit(1);
    }
    return c.getLogPost(last_sample);
  };
  double evaluate(state &s){return exp(evaluate_log(s));};
};

//** CHAIN Classes *******

// A markov (or non-Markovian) chain based on some variant of the Metropolis-Hastings algorithm
// May add "burn-in" distinction later.
class MH_chain: public chain{
  int Ninit;
  int Nhist;
  int Ntries;
  int Naccept;
  int last_type;
  vector<int> types;
  vector<state> states;
  vector<double> lposts;
  vector<double> llikes;
  //  vector<double> chainsteps; 
  vector<double> acceptance_ratio; 
  vector<double> invtemps; 
  //may be more efficient (esp in parallel proc) to change temp, than move chain, so save temp history
  //some algorithms use adaptive temperature
  probability_function *llikelihood;
  sampleable_probability_function *lprior;
  double minPrior;
  proposal_distribution *default_prop;
  bool default_prop_set;
public:
  double invtemp;
  //virtual ~MH_chain(){};//Assure correct deletion of any child via base pointer
  MH_chain(probability_function * log_likelihood, sampleable_probability_function *log_prior,double minPrior=-30):
    llikelihood(log_likelihood),lprior(log_prior),minPrior(minPrior){
    Nsize=0;Nhist=0;invtemp=1;Ntries=1;Naccept=1;
    dim=log_prior->getDim();
    default_prop_set=false;
    //cout<<"creating chain at this="<<this<<" with lprior="<<lprior<<endl;//debug
  };
  virtual void reserve(int nmore){//If you know how long you are going to run, it can be more efficient to reserve up front, rather than resizing ever step
    states.reserve(size()+nmore);
    lposts.reserve(size()+nmore);
    llikes.reserve(size()+nmore);
    acceptance_ratio.reserve(size()+nmore);
    invtemps.reserve(size()+nmore);
  };
  virtual int capacity(){return states.capacity();};
  ///Initialize the chain with one or more states.  Technically we do consider these as part of the chain, for output/analysis purposes as they are not selected based on the MH criterion
  void initialize(uint n=1){
    //cout<<"initializing: this="<<this<<": n="<<n<<"  lprior="<<lprior<<endl;
    Ninit=n;
    for(uint i=0;i<n;i++){
      state s=lprior->drawSample(*rng);
      int icnt=0,icntmax=100;
      while(s.invalid()&&icnt<100){
	icnt++;
	if(icnt>=icntmax)
	  cout<<"MH_chain::initialize: Having trouble drawing a valid state.  Latest state:"<<s.show()<<"...was invalid in space:"<<s.getSpace()->show()<<endl;
	s=lprior->drawSample(*rng);
      }
      //cout <<"starting with Nsize="<<Nsize<<endl;//debug
      add_state(s);}
    //cout <<"starting with Nsize="<<Nsize<<endl;
    };
  void add_state(state newstate,double log_like=999,double log_post=999){
    //cout<<"this="<<0<<",adding state: like="<<log_like<<",post="<<log_post<<endl;
    //if log_like or log_post can be passed in to save computation.  Values passed in are assumed to equal the evaluation results.
    //Value 999 signals need to reevaluate.  If true evaluated value was 999, then reevaluating should yield 999 anyway.
    double newllike=log_like;
    if(newllike==999)newllike=llikelihood->evaluate_log(newstate);
    double newlpost=log_post;
    if(newlpost==999)newlpost=lprior->evaluate_log(newstate)+invtemp*newllike;
    states.push_back(newstate);
    lposts.push_back(newlpost);
    llikes.push_back(newllike);
    acceptance_ratio.push_back(Naccept/(double)Ntries);
    Nsize++;
    //cout<<"Added state "<<Nsize-1<<" = ("<<newstate.get_string()<<") lposts,llikes->"<<newlpost<<","<<newllike<<" = "<<lposts[Nsize-1]<<","<<llikes[Nsize-1]<<endl;
    //cout<<"Nsize="<<Nsize<<endl;//debug
    //process records
    invtemps.push_back(invtemp);
    types.push_back(last_type);
  };
  void set_proposal(proposal_distribution &proposal){
    default_prop=&proposal;
    default_prop->set_chain(this);
    default_prop_set=true;  
    //Other possible actions, such as a burn-in chain, or a combination of chains would be possible here.
  };
  void step(){
    if(!default_prop_set){
      cout<<"MH_chain::step() cannot step without setting a proposal.  Either use set_propsal or specify in  the step call."<<endl;
      exit(1);
    }
    step(*default_prop);
  };
  void step(proposal_distribution &prop,void *data=nullptr){
    if(Nsize==0){
      cout<<"MH_chain:step: Can't step before initializing chain.\n"<<endl;
      //cout<<"this="<<this<<endl;
      exit(1);
    }
    //cout<<"in MH_chain::step(prop="<<&prop<<")"<<endl;
    state &oldstate=states[Nsize-1];
    double oldlpost=lposts[Nsize-1];
    double oldlprior=oldlpost-invtemps[Nsize-1]*llikes[Nsize-1];//maybe set this back to zero?
    state newstate=prop.draw(oldstate,*rng);
    //cout<<"chain::step: testing new state:"<<newstate.show();
    //double lcondratio;//log ratio of conditional proposal probabilities
    double newlike,newlpost,newlprior=lprior->evaluate_log(newstate);
    //cout<<"MH_chain::step: newlprior="<<newlprior<<endl;
    if(newlprior-oldlprior>minPrior){//Avoid spurious likelihood calls where prior effectively vanishes.  
      newlike=llikelihood->evaluate_log(newstate);
      newlpost=newlike*invtemp+newlprior;
    } else {
      //cout<<"for state:"<<newstate.show()<<endl;//debug
      //cout<<"lprior="<<newlprior<<"-"<<oldlprior<<"<"<<minPrior<<" -> outside prior range:"<<lprior->show()<<endl;//debug
      newlike=newlpost=-INFINITY;
    }
    //Now the test: 
    double log_hastings_ratio=prop.log_hastings_ratio();
    log_hastings_ratio+=newlpost-oldlpost;
    bool accept=true;
    //cout<<Nhist<<"("<<invtemp<<"): ("<<newlike<<","<<newlpost<<")vs.("<<oldlpost<<")->"<<log_hastings_ratio<<endl;//debug
    if(log_hastings_ratio<0){
      double x=get_uniform(); //pick a number
      accept=(log(x)<log_hastings_ratio);
      //cout<<"     log(x)="<<log(x)<<" -> "<<(accept?"accept":"reject")<<endl;//debug
    }
    Ntries++;
    if(accept){
      Naccept++;
      //cout<<"        accepting"<<endl;//debug
      last_type=prop.type();
      add_state(newstate,newlike,newlpost);
      //chainsteps.push_back(Nhist-Ninit);
    }
    else {
      //cout<<"        rejecting"<<endl;//debug
      add_state(oldstate,llikes[Nsize-1],oldlpost);
      //cout<<"Nhist="<<Nhist<<"        rejected"<<endl;
    }
    Nhist++;
    // cout<<"incremented NHist="<<Nhist<<endl;
  };
  double expectation(double (*test_func)(state s),int Nburn=0){
    double sum=0;
    for(int i=Ninit+Nburn;i<Nsize;i++)sum+=test_func(states[i]);
    return sum/(Nsize-Ninit-Nburn);
  };
  double variance(double (*test_func)(state s),double fmean,int Nburn=0){
    double sum=0;
    for(int i=Ninit+Nburn;i<Nsize;i++){
      double diff=test_func(states[i])-fmean;
      sum+=diff*diff;
    }
    return sum/(Nsize-Ninit-Nburn);
  };
  int size(){return Nsize;};
  state getState(int elem=-1){
    if(elem<0||elem>=Nsize)
      return states[Nsize-1];//Current state by default
    else
      return states[elem];
  };
  double getLogPost(int elem=-1){
    if(elem<0||elem>=Nsize)
      return lposts[Nsize-1];//Current state by default
    else
      return lposts[elem];
  };
  double getLogLike(int elem=-1){
    if(elem<0||elem>=Nsize)
      return llikes[Nsize-1];//Current state by default
    else
      return llikes[elem];
  };
  string getStateStringLabels(){
    ostringstream s;
    int np=getState().size();
    s<<"#Nsize: log(posterior) acceptance_ratio: ";
    for(int i=0;i<np-1;i++)s<<"param("<<i<<") ";
    s<<"param("<<np-1<<")";
    return s.str();
  };
  string getStateString(){
    ostringstream s;
    int np=getState().size();
    s<<Nsize<<": "<<lposts[Nsize-1]<<" "<<acceptance_ratio[Nsize-1]<<": ";
    vector<double> pars=getState().get_params_vector();
    for(int i=0;i<np-1;i++)s<<pars[i]<<" ";
    s<<pars[np-1];
    return s.str();
  };
  void dumpChain(ostream &os,int Nburn=0,int ievery=1){
    if(Nsize==0)return;
    int np=states[0].size();
    os<<"#Ninit="<<Ninit<<", Nburn="<<Nburn<<"\n";
    os<<"#eval: log(posterior) log(likelihood) acceptance_ratio prop_type: ";
    for(int i=0;i<np-1;i++)os<<"param("<<i<<") ";
    os<<"param("<<np-1<<")"<<endl;
    if(Nburn+Ninit<0)Nburn=-Ninit;
    for(int i=Ninit+Nburn;i<Nsize;i+=ievery){
      os<<i<<" "<<lposts[i]<<" "<<llikes[i]<<" "<<acceptance_ratio[i]<<" "<<types[i]<<": ";
      vector<double>pars=states[i].get_params_vector();
      //cout<<i<<"=i<states.size()="<<states.size()<<"?"<<endl;//debug
      //cout<<"state:"<<states[i].show()<<endl;
      for(int j=0;j<np-1;j++)os<<pars[j]<<" ";
      os<<pars[np-1]<<endl;
    }
  };
  virtual int i_after_burn(int nburn=0){return nburn+Ninit;}
  virtual string show(){
    ostringstream s;
    s<<"MH_chain(id="<<id<<"invtemp="<<invtemp<<"size="<<Nsize<<")\n";
    return s.str();
  };
  virtual string status(){
    ostringstream s;
    s<<"chain(id="<<id<<", N="<<Nsize<<", T="<<1.0/invtemp<<"):"<<lposts[Nsize-1]<<", "<<llikes[Nsize-1]<<" : "<<this->getState().get_string();
    return s.str();
  };

};

// A parallel tempering set of markov (or non-Markovian) chain
// May add "burn-in" distinction later.
class parallel_tempering_chains: public chain{
  //in this implementation, temps are kept with chains
  //states are swapped in exchange steps.
  //General reference: Littenberg-Cornish arxiv:1008.1577
  //Mainly based on KatzenbergerEA06, but for now we don't implement any temperature
  //tuning.  For now we use aspects of Katzenberger's tuning algorithm for diagnostics.
  //Thus we say "replicas" move between chains/temps.
  int Ntemps;
  ///swap rate per chain.  Effective max of 1/(Ntemps-1), ie 1 swap per step total.
  double swap_rate;
  vector<MH_chain> chains;//The actual chains.
  //vector<int> ichains;//Which chain is each replica attached to now
  //Next 3 are diagnostic, based on the auto-tuning alogorithm of KatzenbergerEA06
  //(arXiv:cond-mat/0602085) which has this replica last visited?
  //direction= -1 for 0-temp,+1 for max temp,0 for neither yet,
  vector<int> directions;
  vector<int> ups;//on how many steps has this temp had "up" history
  vector<int> downs;
  vector<proposal_distribution*> props;//probably leaking now from props member, maybe should change to unique_ptr<proposal_distribution>
  vector<int> swap_accept_count;
  vector<int> swap_count;
  vector<double> temps;
 public:
  virtual ~parallel_tempering_chains(){  };//assure correct deletion of any child
  parallel_tempering_chains(int Ntemps,int Tmax,double swap_rate=0.01):Ntemps(Ntemps),swap_rate(swap_rate){
    props.resize(Ntemps);
    directions.resize(Ntemps);
    ups.resize(Ntemps);
    downs.resize(Ntemps);
    swap_accept_count.assign(Ntemps-1,0);
    swap_count.assign(Ntemps-1,0);
    temps.assign(Ntemps,0);
    Nsize=0;
    //make geometrically spaced temperature set.
    double tratio=exp(log(Tmax)/(Ntemps-1));
    temps[0]=1;
    for(int i=1;i<Ntemps;i++)temps[i]=temps[i-1]*tratio;
  };
  void initialize( probability_function *log_likelihood, sampleable_probability_function *log_prior,int n=1){
    dim=log_prior->getDim();
    for(int i=0;i<Ntemps;i++){
      cout<<"PTchain: initializing chain "<<i<<endl;
      MH_chain c(log_likelihood, log_prior);
      chains.push_back(c);
      //cout<<"initializing chain "<<i<<", Ninit="<<n<<endl;
      chains[i].initialize(n);
      chains[i].invtemp=1/temps[i];
      //cout<<"initialized chain "<<i<<" at "<<&chains[i]<<endl;
      //ichains[i]=i;
      if(i==0)directions[i]=-1;
      else if(i==Ntemps-1)directions[i]=1;
      else directions[i]=0;
      ups[i]=0;
      downs[i]=0;
      Nsize=c0().size();
    };
  };
  void set_proposal(proposal_distribution &proposal){
    //cout<<"("<<this<<")ptchain::set_proposal(proposal="<<&proposal<<")"<<endl;
    for(int i=0;i<Ntemps;i++){
      props[i]=proposal.clone();
      //cout<<"cloning proposal: props["<<i<<"]="<<props[i]<<endl;
      //cout<<"                    chain=("<<&chains[i]<<")"<<chains[i].show()<<endl;
      props[i]->set_chain(&chains[i]); //*** This seems to set the last chain to all props...***
      //Other possible actions, such as a burn-in chain, or a combination of chains would be possible here.
    }
  };
  void step(){
    for(int i=0;i<Ntemps;i++){
      //cout<<"Calling ("<<this<<")ptchain::step() for chain="<<&chains[i]<<" with prop="<<props[i]<<endl;
      chains[i].step(*props[i]);
    }
    double x;
    if(Ntemps>1&&get_uniform()<(Ntemps-1)*swap_rate){//do swap 
      //(above, if Ntemps==1,we avoid calling RNG for equivalent behavior to single chain)
      //pick a chain
      int i=int(get_uniform()*(Ntemps-1));
      //diagnostic records first
      if(i>0){
	if(directions[i]>0)ups[i]++;
	if(directions[i]<0)downs[i]++;
      }
      //cout<<"LogLikes:"<<chains[i].getLogLike()<<" "<<chains[i+1].getLogLike()<<endl;
      //cout<<"invtemps:"<<chains[i].invtemp<<" "<<chains[i+1].invtemp<<endl;
      double log_hastings_ratio=-(chains[i+1].invtemp-chains[i].invtemp)*(chains[i+1].getLogLike()-chains[i].getLogLike());//Follows from (21) of LittenbergEA09.
      bool accept=true;
      if(log_hastings_ratio<0){
	x=get_uniform(); //pick a number
	accept=(log(x)<log_hastings_ratio);
      }
      //cout<<i<<" "<<log_hastings_ratio<<" -> "<<(accept?"Swap":"----")<<endl;
      if(accept){
	//we swap states and leave temp fixed.
	state sA=chains[i].getState();
	double llikeA=chains[i].getLogLike();
	//double lpostA=chains[i].getLogPost();  //This doesn't work since posterior depends on temperature.
	state sB=chains[i+1].getState();
	//double lpostB=chains[i+1].getLogPost();
	double llikeB=chains[i+1].getLogLike();
	double dirB=directions[i+1];
	//chains[i+1].add_state(sA,llikeA,lpostA);
	chains[i+1].add_state(sA,llikeA);  //Allow the chain to compute posterior itself.
	//chains[i].add_state(sB,llikeB,lpostB);
	chains[i].add_state(sB,llikeB);
	directions[i+1]=directions[i];
	directions[i]=dirB;
	if(i==0)directions[i]=1;
	if(i+1==Ntemps-1)directions[i+1]=-1;
	swap_accept_count[i]++;
      }
      swap_count[i]++;
    }
    Nsize++;
  };
  ///reference to zero-temerature chain.
  MH_chain & c0(){return chains[0];};
  state getState(int elem=-1){return c0().getState(elem);};
  double getLogPost(int elem=-1){return c0().getLogPost(elem);};
  void dumpChain(ostream &os,int Nburn=0,int ievery=1){  dumpChain(0,os,Nburn,ievery);}
  void dumpChain(int ichain,ostream &os,int Nburn=0,int ievery=1){
    chains[ichain].dumpChain(os,Nburn,ievery);
  };
  void dumpTempStats(ostream &os){
    if(Nsize==0)return;
    os<<"#T0 up_frac0 up-swap_ratio0-1"<<endl;
    for(int i=0;i<Ntemps;i++){
      double up_frac;
      if(i==0)up_frac=1;      
      else if(i==Ntemps-1)up_frac=0;      
      else up_frac=ups[i]/(double)(ups[i]+downs[i]);
      os<<temps[i]<<" "<<up_frac<<" ";
      //cout<<"i="<<i<<" ups="<<ups[i]<<" downs="<<downs[i]<<endl;
      if(i<Ntemps-1)os<<swap_accept_count[i]/(double)swap_count[i]<<": ";
      //else os<<0;
      os<<endl;	     
    }
    os<<"\n"<<endl;
  };
  virtual int i_after_burn(int nburn=0){return c0().i_after_burn(nburn);}
  virtual int capacity(){return c0().capacity();};
  virtual string show(){
    ostringstream s;
    s<<"parallel_tempering_chains(id="<<id<<"Ntemps="<<Ntemps<<"size="<<Nsize<<")\n";
    return s.str();
  };
  virtual string status(){
    ostringstream s;
    s<<"chain(id="<<id<<", Ntemps="<<Ntemps<<"):\n";
    for(int i=0;i<Ntemps;i++)s<<chains[i].status()<<"\n";
    return s.str();
  };
 
  //thermal integration evidence???
};
  
    
//************** PROPOSAL DISTRIBUTION classes ******************************


//Draw from a distribution

//A trivial wrapper
class draw_from_dist: public proposal_distribution{
  sampleable_probability_function &dist;
public:
  draw_from_dist(sampleable_probability_function &dist):dist(dist){};
  state draw(state &s,Random &rng){return dist.drawSample(rng);};
  draw_from_dist* clone()const{return new draw_from_dist(*this);};
  string show(){return "DrawFrom["+dist.show()+"]()";};
};

class gaussian_prop: public proposal_distribution{
  valarray<double>sigmas;
  gaussian_dist_product *dist;
public:
  gaussian_prop(valarray<double> sigmas):sigmas(sigmas){
    valarray<double> zeros(0.0,sigmas.size());
    dist = new gaussian_dist_product(nullptr,zeros, sigmas);
  };
  virtual ~gaussian_prop(){delete dist;};
  state draw(state &s,Random &rng){
    //Umm is this Markovian. Do we need to set log_hastings for...
    state offset=dist->drawSample(rng);
    return s.add(offset);
  };
  gaussian_prop* clone()const{return new gaussian_prop(*this);};
  string show(){return "StepBy["+dist->show()+"]()";};
};


//Typically you want to draw from a set of various proposal distributions.
class proposal_distribution_set: public proposal_distribution{
  int Nsize;
  vector<proposal_distribution*>proposals;
  vector<double>bin_max;
public:
  virtual ~proposal_distribution_set(){
    for(auto prop:proposals)
      if(prop)delete prop;//delete proposals
  };
  virtual proposal_distribution_set* clone()const{//need a deep clone...
    proposal_distribution_set* result;
    result=new proposal_distribution_set(*this);
    for(int i=0;i<proposals.size();i++){
      result->proposals[i]=proposals[i]->clone();
      //cout<<"proposal_distribution_set::clone(): cloning->new prop="<<result->proposals[i]<<endl;
    }
    return result;
  };
  proposal_distribution_set(vector<proposal_distribution*> &props,vector<double> &shares){
    //First some checks
    if(props.size()!=shares.size()){
      cout<<"proposal_distribution_set(constructor): Array sizes mismatched.\n";
      exit(1);
    }
    Nsize=shares.size();
    double sum=0;
    for(int i=0;i<Nsize;i++)sum+=shares[i];//make sure the portions total one
    for(int i=0;i<Nsize;i++){
      proposals.push_back(props[i]);//no copy here.
      if(i==0)bin_max.push_back(shares[0]/sum);
      else bin_max.push_back((bin_max[i-1]+shares[i])/sum);
    }
    last_type=0;
  };
  ///For proposals which draw from a chain, we need to know which chain
  void set_chain(chain *c){for(int i=0;i<Nsize;i++)proposals[i]->set_chain(c);};
  ///Randomly select from proposals i in 0..n and draw.
  ///Sets type() value at i+10*proposals[i].type() 
  state draw(state &s,Random &rng){
    //cout<<"in prop_dist_set::draw(this="<<this<<")"<<endl;
    int count=0;
    while(true){
      double x;
      if(Nsize>1)x=rng.Next();
      else x=0;//this should make a trival (set of 1) proposal_dist the same as the underlying dist even up to RNG draws.
      //rngout<<x<<" 3"<<endl;
      for(int i=0;i<Nsize;i++){
	if(proposals[i]->is_ready()&&x<bin_max[i]){
	  state out= proposals[i]->draw(s,rng);
	  log_hastings=proposals[i]->log_hastings_ratio();
	  //cout<<"Applying prop "<<proposals[i]->show();
	  last_type=i+10*proposals[i]->type();//We assume no need for >10 prop classes in the set
	  return out;
	}
      }
      cout<<"proposal_distribution_set::draw:Proposals all not ready."<<endl;
      count++;
      if(count>100){//something is wrong.
	cout<<"propsal_distribution_set::draw: Hmmm... Seems that (nearly?) none of the proposals are ready;\n";
	exit(1);
      }
    }
  };
  string show(){
    ostringstream s;
    s<<"ChooseFrom(";
    double last=0;
    for(uint i=0;i<Nsize;i++){
      s<<"  "<<(bin_max[i]-last)*100.<<"% : "<<proposals[i]->show()<<"\n";
      last=bin_max[i];
    }
    s<<")\n";
    return s.str();
  };
};  
    
//DifferentialEvolution
//Based mainly on (ter Braak and Vrugt 08, Stat Comput (2008) 18: 435–446)
//Also SampsonEA2011 ArXiV:1105.2088
//The basic algorithm involves choosing a random state from the past history and proposing a jump in that
// direction. terBraakEA also add a "small" random jump 
//Comments: SampsonEA comment that "It can be shown that this approach is asymptotically Markovian in the limit as one uses the full past history of the chain."  terBraak don't show this, but state "we conjecture that DE- MCZ is ergodic and converges to a stationary distribution with [the right] pdf."  The basis for this is by general reference to the Adaptive Metropolis-Hastings paper (HaarioEA 2001).  
class differential_evolution: public proposal_distribution {
  bool have_chain;
  int dim;
  //The chain will manage the history.
  //int Mcount;
  //int save_every;//to save memory it is not necessary to save every element of the history, nearby samples are not independent anyway.
  //int Scount;
  //vector<state> history;
  chain *ch;
  ///recommended to sometimes try unit scaling, which promotes mode-hopping.
  double gamma_one_frac;
  double reduce_gamma_fac;
  ///size of small random jumps (terBraak)
  double b_small; 
  ///We ignore the first Mcount*ignore_frac entries in the history;
  double ignore_frac;
  //Comment:Citing that work terBraak also comments that (retaining)  the "full [sampled] past is required to guarantee ergodicity of the chain."  In HaaioEA, however, it seems clear that retaining a fixed recent fraction of the past (they give 1/2 as an example) would also be sufficient for their theorem.  I haven't carefully studied it, but this also seems to be supported by Roberts+Rosenthal2007, perhaps satisfying their "diminishing adaptation" condition.
  double snooker;
  //SampsonEA don't use it, but terBraakEA seem to prefer a variant on DE where the scaling is set by a separate drawing of states;  I guess this propotes variety.
  
  //SOME INTERNAL FUNCTIONS
  state draw_standard(state &s, Random &rng){
    //get gamma
    ///ie gamma weights the size of the jump. It is suggested 2.38/sqrt(2*d)
    double gamma=1.68/sqrt(dim)/reduce_gamma_fac;//Precalc sqrt(d) for speeed?
    double xgamma=rng.Next();
    //rngout<<xgamma<<" 4"<<endl;
    if(xgamma<gamma_one_frac)gamma=1;
    //get pieces of proposal-state expression terBraak08 eq 2
    state s1=draw_from_chain(rng);
    state s2=draw_from_chain(rng);
    //cout<<"s1="<<s1.show()<<endl;
    //cout<<"s2="<<s2.show()<<endl;
    gaussian_dist_product edist(nullptr,dim);
    //cout<<"edist:"<<edist.show();
    state e=edist.drawSample(rng);
    //cout<<"e:"<<e.show();
    /*state prop=e.scalar_mult(b_small);
    //cout<<"x=("<<s.get_string()<<")"<<endl;
    //cout<<"e=("<<prop.get_string()<<")"<<endl;
    prop=prop.add(s1.scalar_mult(gamma));
    //cout<<"s1=("<<s1.get_string()<<")"<<endl;
    //cout<<"s2=("<<s2.get_string()<<")"<<endl;
    prop=prop.add(s2.scalar_mult(-gamma));
    //cout<<"e+(s1-s2)*(gamma="<<gamma<<")=("<<prop.get_string()<<")"<<endl;
    prop=prop.add(s);
    //cout<<"x*=("<<prop.get_string()<<")"<<endl;
    */    
    state prop=s;
    //cout<<"prop:"<<prop.show();
    //cout<<"x=("<<s.get_string()<<")"<<endl;
    //cout<<"e=("<<prop.get_string()<<")"<<endl;
    prop.add(e.scalar_mult(b_small));
    prop=prop.add(s1.scalar_mult(gamma));
    prop=prop.add(s2.scalar_mult(-gamma));
    //cout<<"x*=("<<prop.get_string()<<")"<<endl;
    
    //may be faster to do this with vectors instead of "states"
    log_hastings=0;
    last_type=0;
    return prop;
  };
state draw_snooker(state &s, Random & rng){
    //get gamma
    ///i.e. gamma weights the size of the jump, draw from U[1.2,2.2] as in terBraak08
    double xgamma=rng.Next();
    //rngout<<xgamma<<" 5"<<endl;
    double gamma=(1.2+xgamma)/reduce_gamma_fac;
    ///get pieces of proposal-state expression terBraak08 eq 3
    //cout<<"Snooker step from s=("<<s.get_string()<<")"<<endl;
    double smznorm2=0;
    state minusz=s,smz=s;
    int isafe=0;
    while(smznorm2==0){//we assure that s!=z to realize perform the required projection meaningfully; wasn't specified in terBraak08. This is a real possibility since there are many repeated states in the chain.  Alternative would be to repropose the current state, which is silly, or we could add a small random piece as in the "standard" draw (That possibility is mentioned in terBraak).
      //if(isafe>1)cout<<"*********************** isafe="<<isafe<<endl; 
      state z=draw_from_chain(rng);
      //cout<<"z=("<<z.get_string()<<")"<<endl;
      minusz=z.scalar_mult(-1);
      //cout<<"-z=("<<minusz.get_string()<<")"<<endl;
      smz=s.add(minusz);
      //cout<<"s-z=("<<smz.get_string()<<")"<<endl;
      smznorm2=smz.innerprod(smz);
      //cout<<"|s-z|^2="<<smznorm2<<endl;
      isafe++;
      if(isafe>1000){
	cout<<"differential_evolution::draw_snooker: We seem to be stuck in an infinite loop.  Bailing out!"<<endl;
	exit(1);
      }
    }
    state s1=draw_from_chain(rng);
    state s2=draw_from_chain(rng);
    //cout<<"s1=("<<s1.get_string()<<")"<<endl;
    //cout<<"s2=("<<s2.get_string()<<")"<<endl;
    state ds12=s1.scalar_mult(gamma);
    ds12=ds12.add(s2.scalar_mult(-gamma));
    // cout<<"(s1-s2)*(gamma="<<gamma<<")=("<<ds12.get_string()<<")"<<endl;
    state prop=s;
    prop=prop.add(smz.scalar_mult(ds12.innerprod(smz)/smznorm2));
    state pmz=prop.add(minusz);
    //terBraak08 Eq 4.
    log_hastings=(log(pmz.innerprod(pmz))-log(smznorm2))*(dim-1)/2.0;
    last_type=1;
    return prop;
  };
  state  draw_from_chain(Random &rng){
    if(!is_ready()){
      if(have_chain)cout<<"size="<<ch->size()<<" < "<<get_min_start_size()<<"=minsize"<<endl;
      else cout<<"No chain set!"<<endl;
      cout<<"differential_evolution:draw_from_chain: Chain is not ready. Verify readiness with is_ready() before drawing.\n";
      exit(1);
    }
    //cout<<"drawing from chain: (this="<<this<<",ch="<<ch<<") "<<ch->show()<<endl;
    int size=ch->size();
    int start=0;
    int mins=get_min_start_size();
    int minc=get_min_cut_size();
    if((size-minc)*(1-ignore_frac)>mins)start=(size-minc)*ignore_frac;
    double xrnd=rng.Next();
    //rngout<<xrnd<<" 6"<<endl;
    int index = start+(size-start)*xrnd;
    //cout<<"drew "<<index<<" from ["<<start<<","<<start+size<<")"<<endl;
    return ch->getState(index);
  };
  int get_min_start_size(){return dim*10;};//Don't run below this chain length
  int get_min_cut_size(){return dim*100;};  //Don't ignore early part below this length
public:
  void reduce_gamma(double factor){reduce_gamma_fac=factor;};
  differential_evolution(double snooker=0.0, double gamma_one_frac=0.1,double b_small=0.0001,double ignore_frac=0.3):gamma_one_frac(gamma_one_frac),b_small(b_small),ignore_frac(ignore_frac),snooker(snooker){
    reduce_gamma_fac=1;
    have_chain=false;
  };
void set_chain(chain *c){ch=c;have_chain=true;dim=ch->getDim();};
  ///Verify that the chain is set and that the chain has some minimum length of 10*dim.
  bool is_ready(){
    return(have_chain&&ch->size()>=get_min_start_size());};
  state draw(state &s,Random &rng){
    //cout<<"in DE::draw(this="<<this<<")"<<endl;
    double x=rng.Next(); //pick a number
    //rngout<<x<<" 7"<<endl;
    if(snooker>x)return draw_snooker(s,rng);
    return draw_standard(s,rng);
  };
  virtual differential_evolution* clone()const{return new differential_evolution(*this);};
  string show(){
    ostringstream s;
    s<<"DifferentialEvolution(";
    s<<"snooker="<<snooker<<", gamma_one_frac="<<gamma_one_frac<<", b_small="<<b_small<<", ignore_frac="<<ignore_frac<<")\n";
    return s.str();
  };
}; 
#endif
