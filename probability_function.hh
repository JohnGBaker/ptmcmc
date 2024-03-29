///Probability funtions for MCMC.
///
///state objects are tied to a stateSpace object which specifies to domain.
///boundary objects allow specification of the domain bounraies in each dimension.
///Written by John G Baker - NASA-GSFC (2013-2014)

#ifndef PTMCMC_PROBABILITY_HH
#define PTMCMC_PROBABILITY_HH
#include "states.hh"
//#include <valarray>
//#include <vector>
//#include <sstream>
//#include <cmath>
//#include <iostream>
//#include <utility>
#include <memory>
#include "ProbabilityDist.h"
#include "newran.h"

using namespace std;

extern shared_ptr<Random>globalRNG;

class probability_function;


//** Base probability function classes

/// Base class for defining likelihoods/priors/etc (nonnormalized)
/// Default version is flat.
class probability_function {
protected:
  const stateSpace space_copy;//maybe we don't need const here
  const stateSpace *space; 
public:
  virtual ~probability_function(){};
  //probability_function(const stateSpace *space):space(&space_copy),space_copy(space?*space:stateSpace()){};//May want argument a copy rather than pointer so that it can be initialized from a temp. obj.
  probability_function(const stateSpace *space):space(space){};
  //should we change the state argument to const?
  virtual double evaluate(state &s){return exp(evaluate_log(s));};
  virtual double evaluate_log(state &s){return 0;};
  virtual string show(int i=-1){return "UnspecifiedProb()";};
  const stateSpace* get_space()const{return space;};
};

// A general (abstract) class for defining eg priors/etc 
// from which we can draw samples.
class sampleable_probability_function: public probability_function{
  ///Sometimes we need to know the largest relevant dimension
  void fail(string context)const{cout<<"sampleable_probability_function::"<<context<<": This should be used strictly as a parent class, and it's virtual functions should be overridden in a base class object.  Instances of this parent class should not be referenced."<<endl;exit(1);};
protected:
  unsigned int dim;
public:
  virtual ~sampleable_probability_function(){};
  sampleable_probability_function(const stateSpace *space):dim(0),probability_function(space){};
  virtual state drawSample(Random &rng)const{fail("drawSample");return state();}
  virtual double evaluate(state &s)const{fail("evaluate");return -1;};
  virtual double verbose_evaluate(state &s)const{return exp(evaluate_log(s));};
  virtual double evaluate_log(state &s)const{return log(evaluate(s));};
  virtual int getDim()const{return dim;};
  ///In one dimension invcdf realizes the inverse cumulative probability distribution,
  ///but it is worth clarifying the meaning of the multidimensional analog of this.
  ///Toward that, we can think of the 1-D CDF as a diffeomorphism from the nominal
  ///state space onto the unit interval with the property that the Jacobian J of the
  ///diffeomorphism is equal to the intended probability function p. If x is a point
  ///in the original space and y is its image in the unit interval, then J(x)=dy(x)/dx=p(x)
  ///so we see that y is the integral of p, the cdf.  The inverse CDF is x(y), which is
  ///just what we need to find p(x) distributed values x from y drawn uniformly from
  ///the unit interval.  The relevant generalization of the invcdf is to provide the
  ///inverse of the diffeomorphism which maps the state space to the unit hypercube
  ///and which has Jacobian J(\vec x)=p(\vec x). For products of independent variables
  ///the Jacobian matrix of the required diffeomorphism will be diagonal and the result
  ///is simply the invcdf in each dimension independently, but more complicated maps
  ///are also possible.
  virtual state invcdf(const state &s)const{cout<<"probability_function::invcdf: No invcdf is defined for this probability function: "<<show()<<endl;exit(1);};
  virtual string show(int i=-1)const{return "UnspecifiedSampleableProb()";};
  virtual void getScales(valarray<double> &outarray)const{};
  void getScales(vector<double> &outvector)const{
    valarray<double> outarray;
    getScales(outarray);
    outvector.assign(begin(outarray), end(outarray));
  };
};


//********* DERIVED CLASSES *************

//** PROBABILITY FUNCTIONs *******

// A class for defining gaussian likelihoods/priors/etc
// from which we can draw samples based on the ProbabilityDist class.
// unit normal range on each parameter.
class gaussian_dist_product: public sampleable_probability_function{
  valarray<double> x0s;
  valarray<double> sigmas;
  vector<ProbabilityDist*> dists;
  bool wrap_probability;
public:
  gaussian_dist_product(const stateSpace *space,unsigned int N=1);
  gaussian_dist_product(const stateSpace *space, const valarray<double>&x0s,const valarray<double>&sigmas,bool wrap_probability=false);
  gaussian_dist_product(const stateSpace *space, const vector<double>&x0s,const vector<double>&sigmas,bool wrap_probability=false):
    gaussian_dist_product(space,valarray<double>(x0s.data(), x0s.size()),valarray<double>(sigmas.data(), sigmas.size()), wrap_probability)
  {};
  virtual ~gaussian_dist_product();
  state drawSample(Random &rng)const;
  double evaluate(state &s)const;
  //double evaluate_log(state &s){return log(evaluate(s));};
  state invcdf(const state &s)const;    
  string show(int i=-1)const;
  virtual void getScales(valarray<double> &outarray)const{outarray=std::move(sigmas);};
};

// An class for defining likelihoods/priors/etc
// from which we can draw samples based on the ProbabilityDist class.
// Default version is flat within unit range on each parameter.
class uniform_dist_product: public sampleable_probability_function{
  valarray<double> min;
  valarray<double> max;
  vector<ProbabilityDist*> dists;
public:
  uniform_dist_product(const stateSpace *space , int N=1);
  uniform_dist_product(const stateSpace *space,const valarray<double>&min_corner,const valarray<double>&max_corner);
  uniform_dist_product(const stateSpace *space, const vector<double>&min_corner,const vector<double>&max_corner):
    uniform_dist_product(space,valarray<double>(min_corner.data(), min_corner.size()),valarray<double>(max_corner.data(), max_corner.size()))
  {};
  virtual ~uniform_dist_product();
  state drawSample(Random &rng)const;
  double evaluate(state &s)const;
  //double evaluate_log(state &s){return log(evaluate(s));};
  state invcdf(const state &s)const;    
  string show(int i=-1)const;
  virtual void getScales(valarray<double> &outarray)const{
    outarray=std::move(min);
    for(int i=0;i<dim;i++)outarray[i]=(max[i]-min[i])/2.0;
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
  bool verbose;
public:
  //static const int mixed_dist_product::uniform=1;
  //static const int mixed_dist_product::gaussian=1;
  static const int uniform=1;
  static const int gaussian=2;
  static const int polar=3;
  static const int copolar=4;
  static const int log=5;
  mixed_dist_product(const stateSpace *space,unsigned int N=1);
  mixed_dist_product(const stateSpace *space,const valarray<int> &types,const valarray<double>&centers,const valarray<double>&halfwidths,bool verbose=false);
  mixed_dist_product(const stateSpace *space, const vector<int> &types,const vector<double>&centers,const vector<double>&halfwidths,bool verbose=false): 
    mixed_dist_product(space,valarray<int>(types.data(),types.size()),valarray<double>(centers.data(), centers.size()),valarray<double>(halfwidths.data(), halfwidths.size()),verbose)
  {};

  virtual ~mixed_dist_product();
  state drawSample(Random &rng)const;
  double evaluate(state &s)const override{return evaluate_vb(s,false);};
  double evaluate_vb(state &s,bool verbose)const;
  double verbose_evaluate(state&s)const override{return evaluate_vb(s,true);};
  //double evaluate_log(state &s){return log(evaluate(s));};
  state invcdf(const state &s)const;    
  string show(int i=-1)const;
  virtual void getScales(valarray<double> &outarray)const{
    outarray=halfwidths;
    for(int i=0;i<outarray.size();i++)
      if(types[i]==log){//halfwidths is taken as multiplicative in this case
	double fac=(halfwidths[i]-1/halfwidths[i])/2; //fac*center=(max-min)/2
	if(fac>1)fac=1; //Revert to use center value as scale if that is larger
	outarray[i]=centers[i]*fac;
      }
    //outarray=std::move(halfwidths);
  };
};

///Generic class for defining sampleable probability distribution from a direct product of independent state spaces.
///Not yet implemented.  Need for generalizing prior definitons.
class independent_dist_product: public sampleable_probability_function{
  int Nss; //number of subspaces
  vector<const sampleable_probability_function*> ss_dists;
  vector<const stateSpace*> ss;
  vector<int>index_ss; //holds the subspace to which the ith element belongs
  vector<int>index_ss_index; //holds the index within the subspace where the ith element maps
  vector< vector<int> > ss_indices; //holds the product space index corresponding to each subspace index
public:
  independent_dist_product(const stateSpace *product_space,  const sampleable_probability_function *subspace1_dist, const sampleable_probability_function *subspace2_dist);
  independent_dist_product(const stateSpace *product_space,  const sampleable_probability_function *subspace1_dist, const sampleable_probability_function *subspace2_dist, const sampleable_probability_function *subspace3_dist);
  independent_dist_product(const stateSpace *product_space,  const sampleable_probability_function *subspace1_dist, const sampleable_probability_function *subspace2_dist, const sampleable_probability_function *subspace3_dist, const sampleable_probability_function *subspace4_dist);
  independent_dist_product(const stateSpace *product_space, const vector<const sampleable_probability_function*> &subspace_dists);
  virtual ~independent_dist_product(){};
  state drawSample(Random &rng)const;//Take the direct product state of subspace samples
  double evaluate(state &s)const;//Take product state of subspace evaluate()s
  //double evaluate_log(state &s){return log(evaluate(s));};//Or could be sum of subspace log_evaluate()s
  state invcdf(const state &s)const;//image is the direct product state of subspace invcdf images
  string show(int i=-1)const;
  virtual void getScales(valarray<double> &outarray)const{
    outarray.resize(dim);
    int count=0;
    for(int i=0;i<Nss;i++){
      int ni=ss_dists[i]->getDim();
      valarray<double>ss_scales;
      ss_dists[i]->getScales(ss_scales);
      outarray[slice(count,ni,1)]=ss_scales;
      count+=ni;
    }
  };
};

///Class for realizing the derived probability_function under a transformation of the stateSpace.
///Here we are defining the distribution on the pre-image space assuming the transformation nominally
///Goes from the pre-image to image, and that the distribution is known on the image space.
///This is still notional...
class transformed_dist: public sampleable_probability_function{
  stateSpaceTransform *sst;
public:
  transformed_dist(stateSpaceTransform *sst);
  virtual ~transformed_dist(){};
  state drawSample(Random &rng)const;
  double evaluate(state &s)const;
  //defaults to double evaluate_log(state &s){return log(evaluate(s));};
  state invcdf(const state &s)const;//See comment in base class description of this function.
  string show(int i=-1)const;
};

class chain;

//Draw samples from a chain
//This base version provides support only on the chain points themselves (no interpolation) interpolative variants (eg using approximate nearest neighbor approach) could be developed if needed.  With chain support this can be used as prior as long as the proposal distribution is strictly compatible (in particular only proposing points in the reference chain).
class chain_distribution: public sampleable_probability_function{
  chain & c;
  int istart;
  int last_sample;
public:
  chain_distribution(chain &c, int istart=0);
  state drawSample(Random &rng);
  double evaluate_log(state &s);
  //double evaluate(state &s){return exp(evaluate_log(s));};
};

#endif
