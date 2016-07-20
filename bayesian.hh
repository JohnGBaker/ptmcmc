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

///Provides some functions/interface common to all bayesian components (data, signal, likelihood).
///
///In particular these all implement the stateSpaceInterface, and the Optioned interfaces.
///Derived classes generally provide:
///-addOptions(Options,String):[options.hh]Object defines its (de facto) command-line options. Begin with call to GLens::addOptions(Options,String).
///-setup():Object sets itself up after options have been parsed. Should must set nativeSpace and can define prior with setPrior(). End with call to haveSetup().
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
  virtual shared_ptr<const sampleable_probability_function> getObjectPrior()const{
    if(have_prior)return nativePrior;
    else { cout<<"bayes_component::getObjectPrior: No prior is defined for this object("<<typeid(*this).name()<<")!"<<endl;exit(1);}
  };
  virtual const stateSpace* getObjectStateSpace()const{checkSetup();return &nativeSpace; };
  virtual void setup(){};
  virtual string show_pointers(){
    ostringstream ss;
    if(have_prior){
      ss<<"nativePrior["<<nativePrior.get()<<","<<nativePrior.use_count()<<"]";
    } else ss<<"nativePrior[null]";
    return ss.str();
  };
};

///Interface class for bayesian signal data. This is some kind of compound data.
///We begin with only what we need for ptmcmc, that we can write the signal
class bayes_signal : public bayes_component {
public:
  virtual vector<double> get_model_signal(const state &st, const vector<double> &labels)const=0;
  ///Stochastic signals imply some variance, default is to return 0
  //virtual double getVariance(double tlabel){return 0;};
  virtual vector<double> getVariances(const state &st,const vector<double> tlabel)const{
    //cout<<"bayes_signal::getVariances"<<endl;
    return vector<double>(tlabel.size(),0);};
};

///Interface class for bayesian signal data. This is some kind of compound data.
///We begin with only what we need for ptmcmc, that we can write the signal
//class bayes_old_signal {
//public:
//  virtual int size()=0;
//  virtual void write(ostream &out,state &st, int nsamples=-1, double tstart=0, double tend=0)=0;
//};

///Interface class for bayesian signal data. This is some kind of compound data.
///
///The label space is treated as one-dimensional, though this may be 2-D as
///for image data, it could also be time or freq.
///dvalue is a scale for errors.
class bayes_data : public bayes_component {
  int have_data;
protected:
  vector<double>labels,values,dvalues;
  double label0;
  //bool have_model;
  bool allow_fill;
public:
  enum {LABELS=1,VALUES=2,DVALUES=4};
  //bayes_data():label0(0),have_model(false),have_data(0),allow_fill(false){};
  bayes_data():label0(0),have_data(0),allow_fill(false){};
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
  virtual vector<double>getValues()const{checkData(VALUES);return values;};
  virtual vector<double>getDeltaValues()const{checkData(DVALUES);return dvalues;};
  //virtual double getVariance(int i)const=0;
  virtual vector<double> getVariances(const state &s)const{
    checkData(DVALUES);
    //cout<<"bayes_data::getVariances"<<endl;
    vector<double>v=getDeltaValues();
    for(int i=0;i<v.size();i++){
      v[i]*=v[i];
    }
    return v;
  };
  ///Declare the aspects of the data that are available.
  ///Depending on the nature of the data, it may include labels, values, and/or dvalues
  ///Different operations may need different such elements.
  void haveData(int type=LABELS|VALUES|DVALUES){have_data = have_data | type;};
  //Some data come with their own modeled parameters.
  //virtual void set_model(state &st){have_model=true;};
  ///Check that required types of data are ready and fail otherwise.
  virtual void assertData(int test)const{
    if(0==(have_data & test)){
      cout<<"bayes_data::checkData: Cannot operate on data before it is loaded."<<endl;
      int missing = test & ~ have_data;
      if(missing & LABELS )cout<<"  *labels* are missing."<<endl;
      if(missing & VALUES )cout<<"  *values* are missing."<<endl;
      if(missing & DVALUES )cout<<"  *dvalues* are missing."<<endl;
      exit(1);
    }
  };
  ///Check that required types of data are ready and return result.
  virtual bool checkData(int test)const{
    if(0==(have_data & test))return false;
    return true;
  };
  void fill_data(vector<double> &newvalues,vector<double> &newdvalues){ 
    assertData(LABELS|VALUES|DVALUES);
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
  double best_post;
  state best;  
public:
  bayes_likelihood(stateSpace *sp,bayes_data *data,bayes_signal *signal):probability_function(sp),data(data),signal(signal),like0(0){
    if(sp)best=state(sp,sp->size());
    reset();
};
  //The latter is just a hack for testing should be removed.
  //bayes_likelihood(stateSpace *sp):probability_function(sp),like0(0){};
  ///Hard check that the signal and data are non-null
  virtual void checkPointers()const{
    if(!data||!signal){
      cout<<"bayes_likelihood::checkPointers(): Cannot operate on undefined pointers!"<<endl;
      exit(1);
    }
  };
  //Overloaded from bayes_component
  virtual void setup(){
    haveSetup();
    ///Set up the output stateSpace for this object
    checkPointers();
    nativeSpace=*data->getObjectStateSpace();
    nativeSpace.attach(*signal->getObjectStateSpace());
    //Set the prior...
    setPrior(new independent_dist_product(&nativeSpace,data->getObjectPrior().get(),signal->getObjectPrior().get()));
    space=&nativeSpace;
    best=state(space,space->size());
    //Unless otherwise externally specified, assume nativeSpace as the parameter space
    defWorkingStateSpace(nativeSpace);
  };
  int size()const{return data->size();};
  virtual void reset(){
    best_post=-INFINITY;
    best=best.scalar_mult(0);
  }
  virtual state bestState(){return best;};
  virtual double bestPost(){return best_post;};
  double log_chi_squared(state &s)const{
    checkPointers();
    checkSetup();
    data->assertData(data->LABELS|data->VALUES|data->DVALUES);
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
  };
  double log_poisson(state &s)const{
    checkPointers();
    checkSetup();
    data->assertData(bayes_data::LABELS);
    //here we assume the model data are rates at the data label value points
    //and the extra last entry in model array is the total expected rate
    double sum=0;
    vector<double>labels=data->getLabels();
    vector<double>modelData=signal->get_model_signal(transformSignalState(s),labels);
    int ndata=labels.size();
#pragma omp critical
    {
      for(int i=0;i<ndata;i++){
	sum += log(modelData[i]);
      }
      sum -= modelData[ndata];//For Poisson data the model array should end with the total number of events expected.
    }
    return sum-like0;
  };
  //virtual bool setStateSpace(stateSpace &sp)=0;
  virtual vector<double> getVariances(const state &st)const{
    checkPointers();
    cout<<"bayes_likelihood::getVariances"<<endl;
    const vector<double>labels=data->getLabels();
    vector<double>var=data->getVariances(transformDataState(st));
    vector<double>svar=signal->getVariances(transformSignalState(st),labels);
    //cout<<"bayes_like::getVariances: var.size="<<var.size()<<" svar.size="<<svar.size()<<endl;
    for(int i=0;i<data->size();i++)var[i]+=svar[i];
    return var;
  };
  ///from stateSpaceInterface
  virtual void defWorkingStateSpace(const stateSpace &sp){
    checkSetup();//Call this assert whenever we need options to have been processed.
    haveWorkingStateSpace();
    checkPointers();
    data->defWorkingStateSpace(sp);
    signal->defWorkingStateSpace(sp);
  };
  //virtual void set_model(state &st){};
  //virtual void write(ostream &out,state &st, int nsamples=-1, double tstart=0, double tend=0)=0;
  ///This is a function for writing out the data and model values and error bars.
  ///The dummy version here may work for many applications, or the function may be overloaded to suit a specific application
  virtual void write(ostream &out,state &st){
    checkWorkingStateSpace();//Call this assert whenever we need the parameter index mapping.
    checkPointers();
    data->assertData(data->LABELS);
    bool have_vals=data->checkData(data->VALUES);
    bool have_dvals=data->checkData(data->DVALUES);
    double xfoc=0,x0=0;  
    std::vector<double> dmags,dvar;
    std::vector<double>xs=data->getLabels();
    if(have_vals){
      xfoc=data->getFocusLabel();
      x0=data->getFocusLabel(true);
    }
    std::vector<double> model=signal->get_model_signal(transformSignalState(st),xs);
    if(have_dvals){
      dmags=data->getDeltaValues();
      dvar=getVariances(st);
    }
    
    for(int i=0;i<xs.size();i++){
      double S=0;
      if(have_dvals)S=dvar[i];
      if(i==0){
	out<<"#x"<<" "<<"x_vs_xfoc" 
	   <<" "<<"model_val";
	if(have_vals)  out<<" "<<"data_val";
	if(have_dvals) out<<" "<<"model_err"<<" "<<"data_err";
	out<<endl;
      }
      out<<xs[i]+x0<<" "<<xs[i]-xfoc
	 <<" "<<model[i];
      if(have_vals)  out<<" "<<data->getValue(i);
      if(have_dvals) out<<" "<<sqrt(S)<<" "<<dmags[i];
      out<<endl;
    }
   
  };
  ///This is a function for writing out the model values and error bars on a grid, generally imagined to be finer-grained than
  ///the data values. In particular, this allows examining the model where it fills in gaps in the data.
  ///The dummy version here may work for many applications, or the function may be overloaded to suit a specific application.
  virtual void writeFine(ostream &out,state &st,int samples=-1, double xstart=0, double xend=0){
    std::vector<double>xs=data->getLabels();
    if(samples<0)getFineGrid(samples,xstart,xend);
    double delta_x=(xend-xstart)/(samples-1.0);
    for(int i=0;i<samples;i++){
      double x=xstart+i*delta_x;
      xs.push_back(x);
    }
    double xfoc=data->getFocusLabel();
    double x0=data->getFocusLabel(true);
    
    vector<double> model=signal->get_model_signal(transformSignalState(st),xs);
    vector<double> dmags=data->getDeltaValues();
    vector<double> dvar=getVariances(st);

    for(int i=0;i<xs.size();i++){
      double x=xs[i];
      if(i==0)
	out<<"#x"<<" "<<"x_vs_xfoc" 
	   <<" "<<"model_mag"
	   <<endl;
      out<<x+x0<<" "<<x-xfoc
	 <<" "<<model[i]
	 <<endl;
    }
  };
  ///This function defines the grid used by writeFine (by default). 
  ///The dummy version here may work for many applications, or the function may be overloaded to suit a specific application.
  void getFineGrid(int & nfine, double &xfinestart, double &xfineend)const{
    checkPointers();
    nfine=data->size()*2;
    double x0,xstart,xend;
    data->getDomainLimits(xstart,xend);
    x0=data->getFocusLabel();
    double finewidthfac=1.5;
    xfinestart=x0-(-xstart+xend)*finewidthfac/2.0;
    xfineend=x0+(-xstart+xend)*finewidthfac/2.0;
  };
    
  protected:
  ///A largely cosmetic adjustment to yield conventional likelihood level with noise_mag=0;
  virtual void set_like0_chi_squared(){
    checkPointers();
    data->assertData(data->DVALUES);
    like0=0;
    const vector<double>dv=data->getDeltaValues();
    for(int i=0; i<size(); i++){
      like0+=log(dv[i]*dv[i]);
    }
    like0/=-2;
    cout<<"bayesian_likelihood:Setting like0="<<like0<<endl;
    //cout<<"this="<<this<<endl;
  };
  ///May apply transformations to the Data/Signal states
  virtual state transformDataState(const state &s)const {return s;};
  ///May apply transformations to the Data/Signal states
  virtual state transformSignalState(const state &s)const {return s;};
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

