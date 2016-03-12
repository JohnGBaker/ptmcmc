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

///This class provides some functions/interface common to all bayesian components (data, signal, likelihood).
///
///In particular these all implement the stateSpaceInterface, and the Optioned interfaces.
///Derived classes generally provide:
///-addOptions(Options,String):[options.hh]Object defines its (de facto) command-line options. Begin with call to GLens::addOptions(Options,String).
///-setup():Object sets itself up after options have been parsed. Should set nativeSpace and nativePrior if used. End with call to haveSetup().
///-getObjectStateSpace():[states.hh,required] Object returns an appropriate stateSpace for its params. Begin with checkSetup();
///-getObjectPrior(): Object returns an appropriate for its params. Begin with checkSetup();
///-defWorkingStateSpace():[states,hh,required] Object finds location of its parameters in the state space. End by calling haveWorkingStateSpace().
///-setState(State): Object takes values from its parameters as needed for state-specific computations. Begin with haveWorkingStateSpace().
class bayes_component: public stateSpaceInterface,public Optioned{
  bool have_setup;
  bool have_prior;
protected:
  stateSpace nativeSpace;
  shared_ptr<const sampleable_probability_function> nativePrior;
  bayes_component(){have_setup=false;have_prior=false;};
  ~bayes_component(){};
  ///This declares that setup is complete.
  void haveSetup(){have_setup=true;};
  void setPrior(sampleable_probability_function* prior){nativePrior.reset(prior);have_prior=true;};
  ///This assert checks that the object is already set up.
  bool checkSetup(bool quiet=false)const{
    if((!quiet)&&!have_setup){
      cout<<"bayes_component::checkSetup: Cannot apply object before setup. Be sure to call haveSetup() when set-up is complete."<<endl;
      exit(1);
    }
    return have_setup;
  };
public:
  ///Return a pointer to an appropriate prior for this objects stateSpace
  virtual const sampleable_probability_function* getObjectPrior()const{
    if(have_prior)return nativePrior.get();
    else { cout<<"bayes_component::getObjectPrior: No prior is defined for this object!"<<endl;exit(1);}
  };
  virtual const stateSpace* getObjectStateSpace()const{checkSetup();return &nativeSpace; };
};
	
///Interface class for bayesian signal data. This is some kind of compound data.
///We begin with only what we need for ptmcmc, that we can write the signal
class bayes_signal : public bayes_component {
public:
  virtual vector<double> get_model_signal(const state &st, const vector<double> &labels)const=0;
  ///Stochastic signals imply some variance, default is to return 0
  //virtual double getVariance(double tlabel){return 0;};
  virtual vector<double> getVariances(const state &st,const vector<double> tlabel)const{return vector<double>(tlabel.size(),0);};
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
class bayes_data : public bayes_component {
public:
protected:
  vector<double>labels,values,dvalues;
  double label0;
  bool have_model;
  bool have_data;
  bool allow_fill;
public:
  bayes_data():label0(0),have_model(false),have_data(false),allow_fill(false){};
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
  //virtual double getVariance(int i)const=0;
  virtual vector<double> getVariances(const state &s)const{
    checkData();
    cout<<"bayes_data::getVariances"<<endl;
    vector<double>v=getDeltaValues();
    for(int i=0;i<v.size();i++){
      v[i]*=v[i];
    }
    return v;
  };
  //Some data come with their own modeled parameters.
  //virtual void set_model(state &st){have_model=true;};
  ///Check that data are ready.  Do we need this in addition to checkSetup?
  virtual void checkData()const{
    if(!have_data){
      cout<<"bayes_data::checkData: Cannot operate on data before it is loaded."<<endl;
      exit(1);
    }
  };
  void fill_data(vector<double> &newvalues,vector<double> &newdvalues){ 
    checkData();
    if(!allow_fill){
      cout<<"bayes_data::fill_data: Operation not permitted for this class object/instance."<<endl;
      exit(-1);
    }
    //test the vectors
    if(newvalues.size()!=labels.size()||newdvalues.size()!=labels.size()){
      cout<<"bayes_data::fill_data: Input arrays are of the wrong size."<<endl;
      exit(-1);
    }
    for(int i=0;i<labels.size();i++){
      values[i]=newvalues[i];
      dvalues[i]=newdvalues[i];
    }
  };  
};

///Bayes class for a likelihood function object
///
///The interface here is probably not everything we want, but is enough for what was already in the main function.
class bayes_likelihood : public probability_function, public bayes_component {
  double like0;
protected:
  bayes_signal *signal;
  bayes_data *data;
  
public:
  bayes_likelihood(stateSpace *sp,bayes_data *data,bayes_signal *signal):probability_function(sp),data(data),signal(signal),like0(0){};
  //The latter is just a hack for testing should be removed.
  //bayes_likelihood(stateSpace *sp):probability_function(sp),like0(0){};
  ///Hard check that the signal and data are non-null
  virtual void checkPointers()const{
    if(!data||!signal){
      cout<<"bayes_likelihood::checkPointers(): Cannot operate on undefined pointers!"<<endl;
      exit(1);
    }
  };
  int size()const{return data->size();};
  virtual void reset()=0;
  virtual double bestPost()=0;
  virtual state bestState()=0;
  double log_chi_squared(state &s)const{
    checkPointers();
    checkSetup();
    //here we assume the model data is magnitude as function of time
    //and that dmags provides a 1-sigma error size.
    double sum=0;
    double nsum=0;
    vector<double>tlabels=data->getLabels();
    vector<double>modelData=signal->get_model_signal(transformSignalState(s),tlabels);
    vector<double>S=getVariances(s);
#pragma omp critical
    {
      //cout<<"size="<<tlabels.size()<<endl;
      for(int i=0;i<tlabels.size();i++){
        double d=modelData[i]-data->getValue(i);
        //double S=getVariance(i,tlabels[i]);
        sum+=d*d/S[i];
        nsum+=log(S[i]);  //note trying to move log outside loop can lead to overflow error.
      }
    }
    //cout<<" sum="<<sum<<"  nsum="<<nsum<<endl;
    sum+=nsum;
    sum/=-2;
    //cout<<" sum="<<sum<<"  like0="<<like0<<endl;
    //cout<<"this="<<this<<endl;
    return sum-like0;
  };  //virtual bool setStateSpace(stateSpace &sp)=0;
  virtual vector<double> getVariances(const state &st)const{
    checkPointers();
    const vector<double>labels=data->getLabels();
    vector<double>var=data->getVariances(transformDataState(st));
    vector<double>svar=signal->getVariances(transformSignalState(st),labels);
    for(int i=0;i<data->size();i++)var[i]+=svar[i];
    return var;
  };
  ///from stateSpaceInterface
  virtual void defWorkingStateSpace(const stateSpace &sp){
    haveWorkingStateSpace();
    checkPointers();
    data->defWorkingStateSpace(sp);
    signal->defWorkingStateSpace(sp);
  };
  //virtual void set_model(state &st){};
  //virtual void write(ostream &out,state &st, int nsamples=-1, double tstart=0, double tend=0)=0;
  virtual void write(ostream &out,state &st)=0;
  virtual void writeFine(ostream &out,state &st,int ns=-1, double ts=0, double te=0)=0;
  virtual void getFineGrid(int & nfine, double &tstart, double &tend)const=0;
  protected:
  ///A largely cosmetic adjustment to yield conventional likelihood level with noise_mag=0;
  void set_like0(){
    checkPointers();
    data->checkData();
    like0=0;
    const vector<double>dv=data->getDeltaValues();
    for(int i=0; i<size(); i++){
      like0+=log(dv[i]*dv[i]);
    }
    like0/=-2;
    cout<<"bayesian_likelihood:Setting like0="<<like0<<endl;
    //cout<<"this="<<this<<endl;
  };
  //virtual state transformDataState(const state &s)const {return s;};
  virtual state transformDataState(const state &s)const=0;
  //virtual state transformSignalState(const state &s)const {return s;};
  virtual state transformSignalState(const state &s)const=0;
  ///This method generates mock data and assigns it to the associated bayes_data object.
public:
  void mock_data(state &s){
    checkPointers();
    checkSetup();
    GaussianDist normal(0.0,1.0);
    //here we assume the model data is magnitude as function of time
    //and that dmags provides a 1-sigma error size.
    vector<double>labels=data->getLabels();
    //First we set up instrumental-noise free data. This is so that the data object can estimate noise
    vector<double>modelData=signal->get_model_signal(transformSignalState(s),labels);
    vector<double>Sm=signal->getVariances(s,labels);
    vector<double>values(labels.size());
    vector<double>dvalues(labels.size());
    for(int i=0;i<labels.size();i++){
      values[i]=modelData[i]+sqrt(Sm[i])*normal.draw();
      dvalues[i]=0;
    }
    data->fill_data(values,dvalues);
    //Now, based on the noise free data object, we estimate the instrumental variance
    vector<double>Sd=data->getVariances(s);
    for(int i=0;i<labels.size();i++){
      double sigma=sqrt(Sd[i]);
      values[i]+=sigma*normal.draw();
      dvalues[i]=sigma;
    }
    data->fill_data(values,dvalues);
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

