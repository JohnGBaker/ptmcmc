// ----- ProbabilityDist:  A Probability Distribution class ----- 
//Written by John G Baker NASA-GSFC (2010-2014)
//Here we are thinking of probability distributions
//e.g. for defining parameter distributions/priors.
//
//Here are some things one usually wants to do with 
//distributions:  
//  1. Randomly draw a value from the distribution.
//  2. Given a value, return a probability density.
// Beyond that we might also want to
//  a. define mean/median/etc
//  b. normalize...
// For now we neglect a and b, possibly to be added later.
// For 1 and 2, we expect the distribution code to provide
//  cdf(x)     Cumulative probability distriubution
//  pdf(x)     Prob. dens. function.
//  invcdf(p)  The inverse of the cdf. (may only be needed internally.
//  draw()     Return a value, x, drawn from the distribution.

#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <limits>
#include "newran.h"
#include "ProbabilityDist.h"
#include <valarray>
 
using namespace std;

//extern ofstream rngout;

double ProbabilityDist::draw(volatile Random *vrng)const{
  double x;
  bool good =false;
#pragma omp critical
  {
    try {
      ///If user specifies random number generator, use it.
      if(vrng){
	Random *rng=(Random *)vrng;
	Random::Set(*rng);
      }
      else //...Else use base class static generator.
	Random::Set(*rnd_num_gen);
      //prog might also use newran
      Uniform uniform;
      x= uniform.Next();
      //rngout<<x<<" 10 "<<this->show()<<endl;
      //cout<<" :"<<x<<" 10 rng="<<(Random *)vrng<<endl;
      good=true;
    } catch(BaseException) {
      cout << BaseException::what() << endl;
    }
  }
  if(good)return invcdf(x);
  cout<<" ProbabilityDist::draw: Failed!"<<endl;
  return numeric_limits<double>::signaling_NaN();
};

MotherOfAll *ProbabilityDist::rnd_num_gen=0;


double GaussianDist::draw(volatile Random *vrng)const{
  double x;
  bool good =false;
#pragma omp critical
  {
    try {
      ///If user specifies random number generator, use it.
      if(vrng){
	Random *rng=(Random *)vrng;
	Random::Set(*rng);
      }
      else//...Else use base class static generator.
	Random::Set(*rnd_num_gen);//I put this in in case another part of the 
      //prog might also use newran
      Normal normal;
      x=normal.Next()*sigma+x0;
      //cout<<" :"<<x<<" 11 rng="<<(Random *)vrng<<endl;
      good=true;
    } catch(BaseException) {
      cout << BaseException::what() << endl;
    }
  }
  if(good)return x;
  cout<<" GaussianDist::draw: Failed!"<<endl;
  return numeric_limits<double>::signaling_NaN();
};

//Lastly handy consistency testing functions:

//First we test draw() versus pdf()
double testProbabilityDist(ProbabilityDist & dist,int Nbin,ostream &os){
  double vmin=DBL_MAX,vmax=-DBL_MAX;
  double counts[Nbin];
  valarray<double> vals(Nbin*Nbin);
  double var=0;
  //draw values
  for(int i=0;i<Nbin*Nbin;i++){
    vals[i]=dist.draw();
    if(vals[i]<vmin)vmin=vals[i];
    if(vals[i]>vmax)vmax=vals[i];
  }
  //cout<<"range:"<<vmin<<"--"<<vmax<<endl;
  //make histogram
  for(int i=0;i<Nbin;i++)counts[i]=0;
  for(int i=0;i<Nbin*Nbin;i++){
    int ibin=(int)((vals[i]-vmin)/(vmax-vmin)*(Nbin-1e-10));
    //if(ibin<0)cout<<"ibin="<<ibin<<" check: "<<vmin<<" < "<<vals[i]<<" < "<<vmax<<" Nbin="<<Nbin<<endl;
    counts[ibin]++;
  }
  for(int i=0;i<Nbin;i++)counts[i]/=Nbin*(vmax-vmin);
  //compare with pdf
  for(int i=0;i<Nbin;i++){
    double x=(i+0.5)*(vmax-vmin)/Nbin+vmin;
    double pdf=dist.pdf(x);
    double diff=counts[i]-pdf;
    var+=diff*diff;
    os<<x<<" "<<counts[i]<<" "<<pdf<<" "<<diff<<endl;
  }
  return sqrt(var)/(Nbin-1);
}

//next we test the normalization of the pdf() versus the cdf()
double testProbabilityDistIntegral(ProbabilityDist & dist,int Nbin){
  double sum=0;
  double mse=0;
  double xmin,x0,xmax;
  int count=0;
  dist.getLimits(xmin,xmax);
  x0=dist.invcdf(0.5);

  //cout<<"xmin,x0,xmax:"<<xmin<<","<<x0<<","<<xmax<<endl;
  //First we do the left half
  if(isfinite(xmin)){
    //integrate on regularly spaced grid over the range
    double dx=(x0-xmin)/Nbin;
    mse+=dist.cdf(xmin)*dist.cdf(xmin);//should be zero, of course, but maybe it isn't
    for(int i=0;i<Nbin;i++){
      double xc = xmin + (i+0.5)*dx;
      double xr = xmin + (i+1.0)*dx;
      sum+=dist.pdf(xc)*dx;
      double err=dist.cdf(xr)-sum;
      mse+=err*err; 
    }
    count+=Nbin+1;
  } else {
    //integrate on semi-infinite logarithmic grid  use x=x0-dx*(exp(z*dy)-1) for the left half
    //with unit intervals over 0<=z<=N.  Let n=N^(1/3);
    //set dx=n/pdf(x0), supposing that the slope of the cdf at midpoint provides a representative length scale
    //set dy=n^(-2/3); thus:
    // 1. Near x0, the step size is ~ dx/n, shrinking with N
    // 2. The region of near-flat spacing (say 1>z*dy) extends to x0-x ~ n*dx, growing with N
    // 3. The outer limit is x0-x ~ n*dx*exp(n), growing exponentially with n
    // note that the derivative |dx/dz| is dy*(x0-x) 
    double nval=pow(Nbin,1/3.0);
    double dx=nval/dist.pdf(x0),dy=1/nval;
    for(int i=0;i<Nbin;i++){  
      double xc = x0 - dx*(exp((Nbin-i-0.5)*dy)-1);
      double xr = x0 - dx*(exp((Nbin-i)*dy)-1);
      sum+=dist.pdf(xc)*dy*(xc-x0);
      double err=dist.cdf(xr)-sum;
      mse+=err*err; 
    }
    count+=Nbin;
  }
  //cout<<"left mse "<<mse<<endl;
  //cout<<"sum = "<<sum<<endl;
  //Now the right half
  sum=0;
  if(isfinite(xmax)){
    //integrate on regularly spaced grid over the range
    double dx=(xmax-x0)/Nbin;
    mse+=(dist.cdf(xmax)-1)*(dist.cdf(xmax)-1);//should be zero, of course, but maybe it isn't
    for(int i=0;i<Nbin;i++){
      double xc = xmax - (i+0.5)*dx;
      double xl = xmax - (i+1.0)*dx;
      sum+=dist.pdf(xc)*dx;
      double err=1-dist.cdf(xl)-sum;
      mse+=err*err; 
    }
    count+=Nbin+1;
  } else {
    //integrate on semi-infinite logarithmic grid  use x=x0+dx*(exp(z*dy)-1) for the left half
    //with unit intervals over 0<=z<=N.  Let n=N^(1/3);
    //set dx=n/pdf(x0), supposing that the slope of the cdf at midpoint provides a representative length scale
    //set dy=n^(-2/3); thus:
    // 1. Near x0, the step size is ~ dx/n, shrinking with N
    // 2. The region of near-flat spacing (say 1>z*dy) extends to x-x0 ~ n*dx, growing with N
    // 3. The outer limit is x-x0 ~ n*dx*exp(n), growing exponentially with n
    // note that the derivative |dx/dz| is dy*(x-x0) 
    double nval=pow(Nbin,1/3.0);
    double dx=nval/dist.pdf(x0),dy=1/nval;
    for(int i=0;i<Nbin;i++){  
      double xc = x0 - dx*(exp((Nbin-i-0.5)*dy)-1);
      double xl = x0 - dx*(exp((Nbin-i)*dy)-1);
      sum+=dist.pdf(xc)*dy*(xc-x0);
      double err=1-dist.cdf(xl)-sum;
      mse+=err*err; 
    }
    count+=Nbin;
  }
  //cout<<"full mse "<<mse<<endl;
  //cout<<"sum = "<<sum<<endl;
  return sqrt(mse)/count;
}
