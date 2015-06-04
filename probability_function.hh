///Probability funtions for MCMC.
///
///state objects are tied to a stateSpace object which specifies to domain.
///boundary objects allow specification of the domain bounraies in each dimension.
///John G Baker - NASA-GSFC (2013-2014)

#ifndef PTMCMC_PROBABILITY_HH
#define PTMCMC_PROBABILITY_HH
#include "mcmc.hh"
//#include <valarray>
//#include <vector>
//#include <sstream>
//#include <cmath>
//#include <iostream>
//#include <utility>
//#include <memory>
//#include "include/ProbabilityDist.h"
//#include "include/newran.h"

using namespace std;

typedef unsigned int uint;

class chain;

//********* DERIVED CLASSES *************

//** PROBABILITY FUNCTIONs *******

// A class for defining gaussian likelihoods/priors/etc
// from which we can draw samples based on the ProbabilityDist class.
// unit normal range on each parameter.
class gaussian_dist_product: public sampleable_probability_function{
  valarray<double> x0s;
  valarray<double> sigmas;
  vector<ProbabilityDist*> dists;
public:
  gaussian_dist_product(stateSpace *space,unsigned int N=1);
  gaussian_dist_product(stateSpace *space, valarray<double>&x0s,valarray<double>&sigmas);
  virtual ~gaussian_dist_product();
  state drawSample(Random &rng);
  double evaluate(state &s);
  double evaluate_log(state &s){return log(evaluate(s));};
  string show();
};

// An example class for defining likelihoods/priors/etc
// from which we can draw samples based on the ProbabilityDist class.
// Default version is flat within unit range on each parameter.
class uniform_dist_product: public sampleable_probability_function{
  valarray<double> min;
  valarray<double> max;
  vector<ProbabilityDist*> dists;
public:
  uniform_dist_product(stateSpace *space , int N=1);
  uniform_dist_product(stateSpace *space,valarray<double>&min_corner,valarray<double>&max_corner);
  virtual ~uniform_dist_product();
  state drawSample(Random &rng);
  double evaluate(state &s);
  double evaluate_log(state &s){return log(evaluate(s));};
  string show();
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
  mixed_dist_product(stateSpace *space,unsigned int N=1);
  mixed_dist_product(stateSpace *space,valarray<int> &types,valarray<double>&centers,valarray<double>&halfwidths);
  virtual ~mixed_dist_product();
  state drawSample(Random &rng);
  double evaluate(state &s);
  double evaluate_log(state &s){return log(evaluate(s));};
  string show();
};

class chain;

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
  state drawSample(Random &rng);
  double evaluate_log(state &s);
  double evaluate(state &s){return exp(evaluate_log(s));};
};

#endif
