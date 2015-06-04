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

//Lastly a handy consistency testing function:
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
