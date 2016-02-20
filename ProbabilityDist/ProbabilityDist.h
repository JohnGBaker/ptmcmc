#ifndef PROBABILITY_DIST_H
#define PROBABILITY_DIST_H
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <sstream>
#include <limits>
#include "newran.h"
#include <valarray> 

using namespace std;

// ----- ProbabilityDist:  A Probability Distribution class ----- 
//Written by John G Baker NASA-GSFC (2010-2014)
///\brief A probability distribution class.
///Here we are thinking of probability distributions
///e.g. for defining parameter distributions/priors.
///Here are some things one usually wants to do with 
///distributions:  
///  1. Randomly draw a value from the distribution.
///  2. Given a value, return a probability density.
/// Beyond that we might also want to
///  a. define mean/median/etc
///  b. normalize...
/// For now we neglect a and b, possibly to be added later.
/// For 1 and 2, we expect the distribution code to provide
//  cdf(x)     Cumulative probability distriubution
///  pdf(x)     Prob. dens. function.
///  invcdf(p)  The inverse of the cdf. (may only be needed internally.
///  draw()     Return a value, x, drawn from the distribution.
/// 
/// Random number generation from NewRan
/// May consider a lighter-weight alternative.
/// See: https://groups.google.com/forum/#!original/comp.lang.c/qZFQgKRCQGg/rmPkaRHqxOMJ
///
class ProbabilityDist {
protected:
  static MotherOfAll *rnd_num_gen;
  unsigned int numGivens;
  valarray<double> givenVals;
public:
  static Random *getPRNG(){return rnd_num_gen;};
  static void setSeed(double seed){//This must be called at begining of program
    try {ProbabilityDist::rnd_num_gen=new MotherOfAll(seed);}
    catch(BaseException) {cout<<"in setSeed (seed="<<seed<<")"<<endl<< BaseException::what() << endl;}};
  ProbabilityDist(){numGivens=0;};
  
  virtual ~ProbabilityDist(){};
  virtual ProbabilityDist* clone(){return new ProbabilityDist(*this);};
  virtual double nGivens()const{return numGivens;};
  virtual bool setGivens(const valarray<double>&givens){
    if(givens.size()!=numGivens)return false;
    givenVals=givens;
    return true;};
  ///Return cumulative probability 
  virtual double cdf(double x)const{return numeric_limits<double>::signaling_NaN();};
  ///Return probability density 
  virtual double pdf(double x)const{return numeric_limits<double>::signaling_NaN();};
  virtual double pdf(double x,valarray<double>&givens){
    setGivens(givens);
    return pdf(x);};
  virtual double invcdf(double x)const{return numeric_limits<double>::signaling_NaN();};
  ///Draw a value from the distribution. Use of volatile is an attempt to force the compiler not to optimize away this call, thereby forcing more predictable randomness (eg to get the same result with different threading).
  virtual double draw(volatile Random *vrng=0)const;
  virtual double draw(valarray<double>&givens,Random *rng=0){
    setGivens(givens);
    return draw(rng);};
  virtual string show()const{return string("<empty>");};
};

//We also implement several explicit versions of this

//Uniform interval
class UniformIntervalDist: public ProbabilityDist {
  double xmin,xmax;
public:
  UniformIntervalDist(double xmin,double xmax):xmin(xmin),xmax(xmax){};
  virtual ProbabilityDist* clone(){return new UniformIntervalDist(*this);};
  //~UniformIntervalDist();
  double cdf(double x)const{
    if(x<xmin)return 0;
    if(x>xmax)return 1;
    return (x-xmin)/(xmax-xmin);
  };
  double pdf(double x)const{
    //ostringstream ss;ss<<"UniformIntervalDist::pdf: checking "<<xmin<<" < "<<x<<" < "<<xmax<<"\n";cout<<ss.str();
    if(x<xmin)return 0;
    if(x>xmax)return 0;
    return 1/(xmax-xmin);
  }
  double invcdf(double p)const{
    if(p<0||p>1)return numeric_limits<double>::signaling_NaN();
    return(p*(xmax-xmin)+xmin);
  };
  string show()const{
    ostringstream ss;
    ss<<"["<<xmin<<","<<xmax<<"]";
    return ss.str();
  };
};

//Gaussian distribution
class GaussianDist: public ProbabilityDist {
protected:
  double x0,sigma;
public:
  GaussianDist(double x0,double sigma):x0(x0),sigma(sigma){};
  virtual ProbabilityDist* clone(){return new GaussianDist(*this);};
  //~GaussianDist();
  double cdf(double x)const{
    return (1+erf((x-x0)/sigma/sqrt(2)))/2;
  };
  double pdf(double x)const{
    double xnorm=(x-x0)/sigma;
    return exp(-xnorm*xnorm/2)/sqrt(2*M_PI)/sigma;
  }
  double invcdf(double p)const{
    //giving up
    cerr<<"Normal class does not support invcdf."<<endl;
    return numeric_limits<double>::signaling_NaN();

  };
  virtual double draw(volatile Random *vrng=0)const;
  string show()const{
    ostringstream ss;
    ss<<"Gaussian("<<x0<<"+/-"<<sigma<<")";
    return ss.str();
  };
};

//UniformPolarDist
//Uniform polar projection distribution
class UniformPolarDist: public ProbabilityDist {
  double xmin,xmax;
public:
  UniformPolarDist():xmin(0),xmax(M_PI){};
  virtual ProbabilityDist* clone(){return new UniformPolarDist(*this);};
  //~UniformPolarDist();
  double cdf(double x)const{
    if(x<xmin)return 0;
    if(x>xmax)return 1;
    return (1-cos(x))/2;
  };
  double pdf(double x)const{
    if(x<xmin)return 0;
    if(x>xmax)return 0;
    return sin(x)/2;
  }
  double invcdf(double p)const{
    if(p<0||p>1)return numeric_limits<double>::signaling_NaN();
    return acos(1-2*p);
  };
  string show()const{
    ostringstream ss;
    ss<<"UniformPolar( ("<<xmin<<","<<xmax<<") )";
    return ss.str();
  };
};

//UniformCoPolarDist
//Uniform polar projection distribution, with polar angle measured to +/- pi/2 from equator
class UniformCoPolarDist: public ProbabilityDist {
  double xmin,xmax;
public:
  UniformCoPolarDist():xmin(-M_PI/2),xmax(M_PI/2){};
  virtual ProbabilityDist* clone(){return new UniformCoPolarDist(*this);};
  //~UniformPolarDist();
  double cdf(double x)const{
    if(x<xmin)return 0;
    if(x>xmax)return 1;
    return (1+sin(x))/2;
  };
  double pdf(double x)const{
    if(x<xmin)return 0;
    if(x>xmax)return 0;
    return cos(x)/2;
  }
  double invcdf(double p)const{
    if(p<0||p>1)return numeric_limits<double>::signaling_NaN();
    return asin(2*p-1);
  };
  string show()const{
    ostringstream ss;
    ss<<"UniformCoPolar( ("<<xmin<<","<<xmax<<") )";
    return ss.str();
  };
};

// GaussianFunctionDist
///Generized Gaussian distribution conditional on one other parameter
///The meana and width of this gaussian are allowed to depend on another
///parameter.  More general, functional distributions could be defined
///Following this example.
class GaussianFunctionDist: public GaussianDist {
protected:
  double (*fsigma)(const valarray<double> &params);
  double (*fx0)(const valarray<double> &params);
public:
  GaussianFunctionDist(  double (*fx0)(const valarray<double> &params),
			 double (*fsigma)(const valarray<double> &params),
			 int nGivens):GaussianDist(0,0),
    fsigma(fsigma),fx0(fx0){
    numGivens=nGivens;
  };
  
  virtual ProbabilityDist* clone(){return new GaussianFunctionDist(*this);};
  ~GaussianFunctionDist(){};
  bool setGivens(const valarray<double>&givens){
    bool ok;
    ok = ProbabilityDist::setGivens(givens);
    x0=fx0(givens);
    sigma=fsigma(givens);
    return ok;
  };
  string show()const{
    ostringstream ss;
    ss<<"Gaussian(x0(pars)+/-sigma(pars))="<<GaussianDist::show();
    return ss.str();
  };
};

// DirectFunctionDist
///This is not a probability distribution per se
///but a substitute indicating that this is a 
///derived variable dependent on some other variables
///and that its probability distribution will flow
//from that dependence. (Still tentative 3/31/10)
class DirectFunctionDist: public ProbabilityDist {
protected:
  double (*f)(const valarray<double> &params);
public:
  DirectFunctionDist(  double (*f)(const valarray<double> &params),int nGivens):f(f){
    numGivens=nGivens;
  };
  virtual ProbabilityDist* clone(){return new DirectFunctionDist(*this);};  
  ~DirectFunctionDist(){};
  bool setGivens(const valarray<double>&givens){
    bool ok;
    ok = ProbabilityDist::setGivens(givens);
    return ok;
  };
  string show()const{
    ostringstream ss;
    ss<<"DirectFunction(pars)";
    return ss.str();
  };
};


///Lastly a handy consistency testing function:
double testProbabilityDist(ProbabilityDist & dist,int Nbin,ostream &os);

#endif
  
