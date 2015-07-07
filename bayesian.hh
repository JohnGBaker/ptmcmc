///General structures for Bayesian analysis
///
///This is a set of interfaces for objects needed in Bayesian analysis.
///The As of June2015, the interfaces are in early stages of development.
///John G Baker - NASA-GSFC (2015)

#ifndef PTMCMC_BAYESIAN_HH
#define PTMCMC_BAYESIAN_HH
//#include <valarray>
//#include <vector>
//#include <sstream>
//#include <cmath>
//#include <iostream>
//#include <memory>
#include "options.hh"
#include "states.hh"
#include "probability_function.hh"

using namespace std;

///Interface class for bayesian signal data. This is some kind of compound data.
///We begin with only what we need for ptmcmc, that we can write the signal
class bayes_signal : public stateSpaceInterface, public Optioned {
public:
  virtual vector<double> get_model_signal(state &st, vector<double> &labels)=0;
  ///Stochastic signals imply some variance, default is to return 0
  virtual double getVariance(double tlabel){return 0;};
};

///Interface class for bayesian signal data. This is some kind of compound data.
///We begin with only what we need for ptmcmc, that we can write the signal
class bayes_old_signal {
public:
  virtual int size()=0;
  virtual void write(ostream &out,state &st, int nsamples=-1, double tstart=0, double tend=0)=0;
};


///Interface class for bayesian signal data. This is some kind of compound data.
///
///The label space is treated as one-dimensional, though this may be 2-D as
///for image data, it could also be time or freq.
///dvalue is a scale for errors.
class bayes_data : public stateSpaceInterface, public Optioned {
public:
protected:
  vector<double>labels,values,dvalues;
  double label0;
  bool have_model;
  bool have_data;
public:
  bayes_data():label0(0),have_model(false),have_data(false){};
  vector<double>getLabels()const{return labels;};
  void getDomainLimits(double &start, double &end)const{
    if(labels.size()==0){
      cout<<"bayes_data::getDomainLimits: Cannot get limit on empty object."<<endl;
      exit(1);
    }
    start=labels.front();
    end=labels.back();
  };
  virtual int size()const=0;
  virtual double getFocusLabel(bool original=false)const=0;
  virtual double getValue(int i )const{return values[i];};
  virtual vector<double>getValues()const{return values;};
  virtual vector<double>getDeltaValues()const{return dvalues;};
  virtual double getVariance(int i)const=0;
  //Some data come with their own modeled parameters.
  virtual void set_model(state &st){have_model=true;};
  virtual void check_data()const{
    if(!have_data){
      cout<<"bayes_data::check_data: Cannot operate on data before it is loaded."<<endl;
      exit(1);
    }
  };
};


///Bayes class for a likelihood function object
///
///The interface here is probably not everything we want, but is enough for what was already in the main function.
class bayes_likelihood : public probability_function, public stateSpaceInterface, public Optioned {
  double like0;
protected:
  bayes_signal *signal;
  bayes_data *data;
  
public:
  bayes_likelihood(stateSpace *sp,bayes_data *data,bayes_signal *signal):probability_function(sp),data(data),signal(signal),like0(0){};
  //The latter is just a hack for testing should be removed.
  //bayes_likelihood(stateSpace *sp):probability_function(sp),like0(0){};
  ///Hard check that the signal and data are non-null
  virtual void check()const{
    if(!data||!signal){
      cout<<"bayes_likelihood::check(): Cannot operate on undefined pointers!"<<endl;
      exit(1);
    }
  };
  int size()const{return data->size();};
  virtual void reset()=0;
  virtual double bestPost()=0;
  virtual state bestState()=0;
  //double log_chi_squared(state &s);
  double log_chi_squared(state &s){
    check();
    //here we assume the model data is magnitude as function of time
    //and that dmags provides a 1-sigma error size.
    double sum=0;
    double nsum=0;
    vector<double>tlabels=data->getLabels();
    
    vector<double>modelData=signal->get_model_signal(s,tlabels);
    //This block is inside omp critical because set_model() sets class members of data, which is a shared object.
    //alternatively we could clone data, but that would mean actually copying the observation data unnecessarily.
    //A better solution might be separating the functions of instrument and instrument data, so that we can clone
    //instrument without copying the data.  All this is only an issue if there are parameters that vary which 
    //the data object needs to know about in getValue or getVariance.
#pragma omp critical
    {
      set_model(s);
      for(int i=0;i<tlabels.size();i++){
        double d=modelData[i]-data->getValue(i);
        double S=getVariance(i,tlabels[i]);
        sum+=d*d/S;
        nsum+=log(S);  //note trying to move log outside loop can lead to overflow error.
      }
    }
    sum+=nsum;
    sum/=-2;
    return sum-like0;
  };  //virtual bool setStateSpace(stateSpace &sp)=0;
  virtual double getVariance(int i, double label){check();return data->getVariance(i)+signal->getVariance(label);};
  ///from stateSpaceInterface
  virtual stateSpace getObjectStateSpace()const{return stateSpace();};
  virtual void defWorkingStateSpace(const stateSpace &sp){
    check();
    data->defWorkingStateSpace(sp);
    signal->defWorkingStateSpace(sp);
  };
  virtual void set_model(state &st){};
  //virtual void write(ostream &out,state &st, int nsamples=-1, double tstart=0, double tend=0)=0;
  virtual void write(ostream &out,state &st)=0;
  virtual void writeFine(ostream &out,state &st)=0;
  virtual void getFineGrid(double & nfine, double &tstart, double &tend)const=0;
  protected:
  ///A largely cosmetic adjustment to yield conventional likelihood level with noise_mag=0;
  void set_like0(){
    check();
    data->check_data();
    like0=0;
    for(int i=0; i<size(); i++){
      double S=data->getVariance(i);
      like0+=log(S);
    }
    like0/=-2;
    cout<<"setting like0="<<like0<<endl;
  };
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
  //virtual int analyze(const string & base, int ic, int Nsigma, int Nbest, bayes_old_signal &data, double tfinestart, double tfineend)=0;
  virtual int analyze(const string & base, int ic, int Nsigma, int Nbest, bayes_likelihood &like)=0;
};


#endif

