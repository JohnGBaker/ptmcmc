//Linear curve with Gaussian noise example
//Written by John G Baker NASA-GSFC (2016)

//#include <valarray>
//#include <vector>
//#include <iostream>
//#include <iomanip>
//#include <fstream>
//#include <ctime>
#include "omp.h"
#include "options.hh"
//#include <mcheck.h>
#include "bayesian.hh"
#include "proposal_distribution.hh"
#include "ptmcmc.hh"


using namespace std;

/*
typedef initializer_list<double> dlist;
typedef initializer_list<int> ilist;
*/

shared_ptr<Random> globalRNG;//used for some debugging... 

///class for mock data
///It does little other than define a grid of points, and allow them to be populated...
///There is also a hook to fill the data, which mllike knows how to do.  In this case
///The "extra noise" parameter becomes the actual noise parameter.
class mock_data : public bayes_data {
  vector<double>&xs,&ys,&dys; //We just change from the abstract names labels,values,etc...
public:
  mock_data():xs(labels),ys(values),dys(dvalues){allow_fill=true;};
  ///The time samples are generated from a regular grid, or randomly...
  ///Note that cadence is the most probable size of timestep, with fractional variance scale set by log_dtvar
  void setup(){
    double xstart,xend,cadence,jitter;
    *optValue("mock_xstart")>>xstart;
    *optValue("mock_xend")>>xend;
    *optValue("mock_cadence")>>cadence;
    *optValue("mock_jitter")>>jitter;
    cout<<"Preparing mock data."<<endl;
    GaussianDist gauss(0.0,jitter);
    double dx=cadence*exp(gauss.draw());
    double x=xstart+dx/2.0;
    while(x<xend){
      xs.push_back(x);
      dx=cadence*exp(gauss.draw());
      x+=dx;
      ys.push_back(0);
      dys.push_back(1.0);
    }
    //for(int i=0;i<labels.size();i++)
    //  cout<<i<<","<<labels[i]<<": "<<values[i]<<","<<dvalues[i]<<endl;
    ///Set up the output stateSpace for this object
    stateSpace space(0);
    nativeSpace=space;
    setPrior(new uniform_dist_product(&nativeSpace,0));//we don't need a prior, since space is zero-D, but we need to have set something
    haveData();
    haveSetup();
  };
  ///Optioned interface
  void addOptions(Options &opt,const string &prefix=""){
    Optioned::addOptions(opt,prefix);
    addOption("mock_xstart","Start x-value for mock data sample grid. Default=-10","-10");
    addOption("mock_xend","End x-value for mock data sample grid. Default=10","10");
    addOption("mock_cadence","Typical sample period for mock data sample grid(days). Default=1","1");
    addOption("mock_jitter","Size of standard deviation in log(time-step-size). Default=0","0");
  };
};

///Define polynomial curve model
///
class poly_signal : public bayes_signal{
  int order;
  vector<int> coeff_idxs;
  vector<string> parnames;
  bool factored;
public:
  poly_signal(int order=2):order(order),factored(false){
  };
  //From bayes_signal
  ///This routine computes the model on an array of x values
  virtual std::vector<double> get_model_signal(const state &st, const std::vector<double> &xs)const{
    vector<double> coeffs(order);
    vector<double> ys(xs.size());
    for(int i=0;i<order;i++)
      coeffs[i]=st.get_param(coeff_idxs[i]);
    for (int i = 0; i < xs.size(); i++){
      double y=0;
      if(factored){
	y=1;
	for(int j=0;j<order;j++)
	  y *= xs[i] - coeffs[j];
      } else {
	double xn=1;
	for(int j=0;j<order;j++){
	  y  += xn*coeffs[j];
	  xn *= xs[i];
	}
      }
      ys[i] = y;
    }
    return ys;
  };
  //From StateSpaceInterface (via bayes_signal)
  ///This routine defines the options [generally on the command-line] that the class will be aware of.  It will generally be called very early by the main program.
  void addOptions(Options &opt,const string &prefix=""){
    Optioned::addOptions(opt,prefix);
    opt.add(Option("poly_factored","Polynomial model parameterized by (x-z[0])(x-z[1])..., to realize a nonlinear parameter space."));
    opt.add(Option("poly_order","Set order for polynomial model. (Default="+to_string(order)+")",to_string(order)));
  };
  //From bayes_component (via bayes_signal)
  ///This routine applies options to define the optional properties of this signal object.
  void setup(){
    if(optSet("poly_factored"))factored=true;
    *optValue("poly_order")>>order;
    coeff_idxs.resize(order,-1);
    haveSetup();
    stateSpace space(order);
    parnames.resize(order);
    if(factored)
      for(int i=0;i<order;i++) parnames[i]="z"+to_string(i);
    else
      for(int i=0;i<order;i++) parnames[i]="c"+to_string(i);
    space.set_names(parnames);  
    nativeSpace=space;
    const int uni=mixed_dist_product::uniform;
    valarray<double>    centers(0.0,order);
    valarray<double>     scales(10.0,order);
    valarray<int>         types(uni,order);
    setPrior(new mixed_dist_product(&nativeSpace,types,centers,scales));
  };
  //From StateSpaceInterface (via bayes_signal)
  ///This routine computes collects information about location of the parameters needed by the signal model from the state-space [which may involve other parameters that this class doesn't know about.  It generally will be called once late in the setup 
  void defWorkingStateSpace(const stateSpace &sp){
    checkSetup();//Call this assert whenever we need options to have been processed.
    for(int i=0;i<order;i++)coeff_idxs[i]=sp.requireIndex(parnames[i]);
    haveWorkingStateSpace();
  };
};

///Basic likelihood which assumes noise from a Gaussian process.
class gauss_proc_likelihood : public bayes_likelihood {
  bool zeroLogLike=false;
public:
  gauss_proc_likelihood(bayes_data *data,bayes_signal *signal):bayes_likelihood(nullptr,data,signal){};
  void addOptions(Options &opt,const string &prefix=""){
    Optioned::addOptions(opt,prefix);
    opt.add(Option("zeroLogLike","Set the log-likelihood to zero (thus purely sampling from prior)."));
  };
  void setup(){
    if(optSet("zeroLogLike"))zeroLogLike=true;
    cout<<"zeroLogLike = "<<(zeroLogLike?"true":"false")<<endl;
    bayes_likelihood::setup();
  };
  double evaluate_log(state &s){
    valarray<double>params=s.get_params();
    double result=0;
    if(not zeroLogLike)result=log_chi_squared(s);
    double post=result;
    post+=nativePrior->evaluate_log(s);
    #pragma omp critical
    {     
      if(post>best_post){
        best_post=post;
        best=state(s);
      }
      if(!isfinite(result)){
        cout<<"Whoa dude, loglike is NAN! What's up with that?"<<endl;
        cout<<"params="<<s.get_string()<<endl;
	result=-INFINITY;
      }
    }
    return result;
  };
  virtual double getFisher(const state &s0, vector<vector<double> >&fisher_matrix)override{
    return getFisher_chi_squared(s0,fisher_matrix);
  };
};
  



//***************************************************************************************8
//main test program
int main(int argc, char*argv[]){

  Options opt(true);
  //Create the sampler
  ptmcmc_sampler mcmc;
  bayes_sampler *s0=&mcmc;
  //Create the model components and likelihood;
  bayes_data *data=new mock_data();
  bayes_signal *signal=new poly_signal();
  bayes_likelihood *like=new gauss_proc_likelihood(data,signal);
  
  //prep command-line options
  s0->addOptions(opt);
  data->addOptions(opt);
  signal->addOptions(opt);
  like->addOptions(opt);

  //Add some command additional line options
  opt.add(Option("nchains","Number of consequtive chain runs. Default 1","1"));
  opt.add(Option("seed","Pseudo random number grenerator seed in [0,1). (Default=-1, use clock to seed.)","-1"));
  opt.add(Option("precision","Set output precision digits. (Default 13).","13"));

  
  int Nlead_args=1;

  bool parseBAD=opt.parse(argc,argv);
  if(parseBAD) {
    cout << "Usage:\n mcmc [-options=vals] " << endl;
    cout <<opt.print_usage()<<endl;
    return 1;
  }
    
  cout<<"flags=\n"<<opt.report()<<endl;

  double seed;
  int Nchain,output_precision;
  int Nsigma=1;
  int Nbest=10;
  string outname;
  ostringstream ss("");
  istringstream(opt.value("nchains"))>>Nchain;
  istringstream(opt.value("seed"))>>seed;
  //if seed<0 set seed from clock
  if(seed<0)seed=fmod(time(NULL)/3.0e7,1);
  istringstream(opt.value("precision"))>>output_precision;
  istringstream(opt.value("outname"))>>outname;

  //report
  cout.precision(output_precision);
  cout<<"\noutname = '"<<outname<<"'"<<endl;
  cout<<"seed="<<seed<<endl; 
  cout<<"Running on "<<omp_get_max_threads()<<" thread"<<(omp_get_max_threads()>1?"s":"")<<"."<<endl;

  //Should probably move this to ptmcmc/bayesian
  ProbabilityDist::setSeed(seed);
  globalRNG.reset(ProbabilityDist::getPRNG());//just for safety to keep us from deleting main RNG in debugging.
  
  //Setup components
  data->setup();  
  signal->setup();  
  like->setup();

  //Get the space/prior for use here
  stateSpace space;
  shared_ptr<const sampleable_probability_function> prior;  
  space=*like->getObjectStateSpace();
  cout<<"like.nativeSpace=\n"<<space.show()<<endl;
  prior=like->getObjectPrior();
  cout<<"Prior is:\n"<<prior->show()<<endl;
  valarray<double> scales;
  prior->getScales(scales);

  //Mock data
  cout<<"Drawing a state from the prior distribution."<<endl;
  state instate=prior->drawSample(*globalRNG);
  like->mock_data(instate);  

  //Bayesian sampling [assuming mcmc]:
  //Set the proposal distribution
  int Npar=space.size();
  int Ninit;
  proposal_distribution *prop=ptmcmc_sampler::new_proposal_distribution(Npar,Ninit,opt,prior.get(),&scales);
  cout<<"Proposal distribution is:\n"<<prop->show()<<endl;
  //set up the mcmc sampler (assuming mcmc)
  mcmc.setup(Ninit,*like,*prior,*prop,output_precision);

  //Prepare for chain output
  ss<<outname;
  string base=ss.str();
  
  //Loop over Nchains
  for(int ic=0;ic<Nchain;ic++){
    bayes_sampler *s=s0->clone();
    s->initialize();
    s->run(base,ic);
    s->analyze(base,ic,Nsigma,Nbest,*like);
    delete s;
  }
  
  //Dump summary info
  cout<<"best_post "<<like->bestPost()<<", state="<<like->bestState().get_string()<<endl;

  cout<<"instate:"<<instate.get_string()<<endl;
  
  //Next compute the Fisher matrix
  vector< vector<double> > fim(Npar,vector<double>(Npar));
  state sf=like->bestState();
  double err=like->getFisher(sf,fim);
  cout<<"Fisher:"<<endl;
  for(int i=0;i<Npar;i++){
    for(int j=0;j<=i;j++)cout<<"\t"<<fim[i][j];
    cout<<endl;
  }
  cout<<"Internally estimated Fisher error is "<<err<<endl;
  //Exact Fisher:
  vector<double>xfim(Npar*(Npar+1)/2,0);
  int icount=0;
  vector<double> xs=data->getLabels();  
  double N=xs.size();
  //cout<<"N="<<N<<" x={"<<xs[0];for(int i=1;i<N;i++)cout<<","<<xs[i];cout<<"}"<<endl;
  if(not opt.set("poly_factored")){
    vector<double> xksum(2*Npar-1,0);
    for(int i=0;i<N;i++){
      double xk=1;
      for(int k=0;k<2*Npar-1;k++){
	xksum[k]+=xk;
	xk*=xs[i];
      }
    }
    for(int i=0;i<Npar;i++){
      for(int j=0;j<=i;j++){
	xfim[icount]=xksum[i+j];
	icount++;
      }
    }
  } else {
    vector<double> zeros=sf.get_params_vector();
    for(int i=0;i<Npar;i++){
      for(int j=0;j<=i;j++){
	double sum=0;
	for(int l=0;l<N;l++){
	  double x=xs[l];
	  double prod=1;
	  for(int k=0;k<Npar;k++){
	    if(k!=i)prod*=(x-zeros[k]);
	    if(k!=j)prod*=(x-zeros[k]);
	  }
	  sum+=prod;
	}
	xfim[icount]=sum;
	icount++;
      }
    }
  }
  //Compute mean difference
  double sum=0,tot=0;
  icount=0;
  for(int i=0;i<Npar;i++){
    for(int j=0;j<=i;j++){
      double delta = xfim[icount]-fim[i][j];
      delta*=delta;
      if(i==j)delta*=2;
      sum+=delta;
      tot+=fim[i][j]*fim[i][j];
      if(i==j)tot+=fim[i][j]*fim[i][j];
      icount++;
    }
  }
  cout<<"Expected Fisher:"<<endl;
  icount=0;
  for(int i=0;i<Npar;i++){
    for(int j=0;j<=i;j++){
      cout<<"\t"<<xfim[icount];
      icount++;
    }
    cout<<endl;
  }
  cout<<"RMS difference="<<sqrt(sum/tot)<<endl;
  delete data;
  delete signal;
  delete like;
}


