//Written by John G Baker NASA-GSFC (2016)

//#include "mlfit.hh"
#include <valarray>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>
#include "omp.h"
#include "options.hh"
//#include <mcheck.h>
#include "bayesian.hh"
#include "proposal_distribution.hh"
#include "ptmcmc.hh"


using namespace std;

///Likelihood function objects
///
class constant_likelihood : public bayes_likelihood {
  int idx_p0,idx_p1,idx_p2;
public:
  constant_likelihood():bayes_likelihood(nullptr,nullptr,nullptr){};
  virtual void setup(){
    haveSetup();
    ///Set up the output stateSpace for this object
    //set nativeSpace
    int npar=3;
    stateSpace space(npar);
    string names[]={"p0","p1","p2"};
    space.set_names(names);  
    nativeSpace=space;
    defWorkingStateSpace(nativeSpace);
    best=state(&space,space.size());
    //Set the prior...
    const int uni=mixed_dist_product::uniform, gauss=mixed_dist_product::gaussian, pol=mixed_dist_product::polar, log=mixed_dist_product::log; 
    valarray<double>    centers((initializer_list<double>){  2.0,  -3.0,  5.0});
    valarray<double> halfwidths((initializer_list<double>){  2.0,   3.0,  5.0});
    valarray<int>         types((initializer_list<int>)   {  uni,   uni,  uni});
    setPrior(new mixed_dist_product(&nativeSpace,types,centers,halfwidths));
  };
  void defWorkingStateSpace(const stateSpace &sp){
    checkSetup();//Call this assert whenever we need options to have been processed.
    idx_p0=sp.requireIndex("p0");
    idx_p1=sp.requireIndex("p1");
    idx_p2=sp.requireIndex("p2");
    haveWorkingStateSpace();
  };
  int size()const{return 0;};
  double evaluate_log(state &s){
    valarray<double>params=s.get_params();
    //double result=log_poisson(s);
    double result=2;
    double post=result;
    post+=nativePrior->evaluate_log(s);//May need a mechanism to check that Prior is set
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
};

///A Gaussian
///
class gaussian_likelihood : public bayes_likelihood {
  int idx_p0,idx_p1,idx_p2;
  double x0[3];
  double lnnormfac;
  double twosigmasq;
public:
  gaussian_likelihood():bayes_likelihood(nullptr,nullptr,nullptr){};
  virtual void setup(){
    haveSetup();
    ///Set up the output stateSpace for this object
    //set nativeSpace
    int npar=3;
    stateSpace space(npar);
    string names[]={"p0","p1","p2"};
    space.set_names(names);  
    nativeSpace=space;
    defWorkingStateSpace(nativeSpace);
    best=state(&space,space.size());
    //Set the prior...
    const int uni=mixed_dist_product::uniform, gauss=mixed_dist_product::gaussian, pol=mixed_dist_product::polar, log=mixed_dist_product::log; 
    valarray<double>    centers((initializer_list<double>){  2.0,  -3.0,  5.0});
    valarray<double> halfwidths((initializer_list<double>){  2.0,   3.0,  5.0});
    valarray<int>         types((initializer_list<int>)   {  uni,   uni,  uni});
    setPrior(new mixed_dist_product(&nativeSpace,types,centers,halfwidths));
    for(int i=0;i<3;i++)x0[i]=centers[i];
    double sigma=0.5;
    twosigmasq=2*sigma*sigma;
    lnnormfac=-1.5*std::log(M_PI*twosigmasq);//1.5=3/2;3 for 3D
    cout<<"Theoretically expected ln(evidence)="<<-std::log(8*halfwidths[0]*halfwidths[1]*halfwidths[2])<<endl;
  };
  void defWorkingStateSpace(const stateSpace &sp){
    checkSetup();//Call this assert whenever we need options to have been processed.
    idx_p0=sp.requireIndex("p0");
    idx_p1=sp.requireIndex("p1");
    idx_p2=sp.requireIndex("p2");
    haveWorkingStateSpace();
  };
  int size()const{return 0;};
  double evaluate_log(state &s){
    valarray<double>params=s.get_params();
    //double result=log_poisson(s);
    double r2=0;
    double dx=params[idx_p0]-x0[0];
    r2+=dx*dx;
    dx=params[idx_p1]-x0[1];
    r2+=dx*dx;
    dx=params[idx_p2]-x0[2];
    r2+=dx*dx;
    const double sqrt2pi=sqrt(2*M_PI);
    double result = lnnormfac - r2/twosigmasq;
    double post=result;
    post+=nativePrior->evaluate_log(s);//May need a mechanism to check that Prior is set
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
};
  
///Double Gaussian shell from FerozEA2013 function objects
///
class gaussian_shell_2D_likelihood : public bayes_likelihood {
  int idx_p0,idx_p1;
  double x0[2],r0;
  double lnnormfac;
  double twosigmasq;
public:
  gaussian_shell_2D_likelihood():bayes_likelihood(nullptr,nullptr,nullptr){};
  virtual void setup(){
    haveSetup();
    ///Set up the output stateSpace for this object
    //set nativeSpace
    int npar=2;
    stateSpace space(npar);
    string names[]={"p0","p1"};
    space.set_names(names);  
    nativeSpace=space;
    defWorkingStateSpace(nativeSpace);
    best=state(&space,space.size());
    //Set the prior...
    const int uni=mixed_dist_product::uniform, gauss=mixed_dist_product::gaussian, pol=mixed_dist_product::polar, log=mixed_dist_product::log; 
    valarray<double>    centers((initializer_list<double>){  0.0,   0.0,  0.0});
    valarray<double> halfwidths((initializer_list<double>){  6.0,   6.0,  0.0});
    valarray<int>         types((initializer_list<int>)   {  uni,   uni,  uni});
    setPrior(new mixed_dist_product(&nativeSpace,types,centers,halfwidths));
    for(int i=0;i<npar;i++)x0[i]=centers[i];
    x0[0]=3.0;
    r0=2;
    double sigma=0.1;
    twosigmasq=2*sigma*sigma;
    lnnormfac=-0.5*std::log(M_PI*twosigmasq);
    double like_integral=exp(-lnnormfac-r0*r0/twosigmasq) + 2*M_PI*(1+erf(r0/sqrt(twosigmasq)));
    like_integral*=2.0;
    double prior_vol=4*halfwidths[0]*halfwidths[1];
    cout<<"Theoretically expected ln(evidence)="<<std::log(like_integral)-std::log(prior_vol)<<endl;
  };
  void defWorkingStateSpace(const stateSpace &sp){
    checkSetup();//Call this assert whenever we need options to have been processed.
    idx_p0=sp.requireIndex("p0");
    idx_p1=sp.requireIndex("p1");
    //idx_p2=sp.requireIndex("p2");
    haveWorkingStateSpace();
  };
  int size()const{return 0;};
  double evaluate_log(state &s){
    valarray<double>params=s.get_params();
    //double result=log_poisson(s);
    double r2=0;
    double dx=abs(params[idx_p0])-x0[0];
    r2+=dx*dx;
    dx=params[idx_p1]-x0[1];
    r2+=dx*dx;
    dx=sqrt(r2)-r0;
    r2=dx*dx;
    const double sqrt2pi=sqrt(2*M_PI);
    double result = lnnormfac - r2/twosigmasq;
    double post=result;
    post+=nativePrior->evaluate_log(s);//May need a mechanism to check that Prior is set
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
};

///Arbitrary-dimensional double Gaussian shell from FerozEA2013 function objects
///
class gaussian_shell_ND_likelihood : public bayes_likelihood {
  int dim;
  vector<int> idx_p;
  double x0,r0;
  double lnnormfac;
  double twosigmasq;
  bool one,logx;
public:
  gaussian_shell_ND_likelihood(int dim=2):dim(dim),one(false),logx(false),bayes_likelihood(nullptr,nullptr,nullptr){};
  void addOptions(Options &opt,const string &prefix=""){
    Optioned::addOptions(opt,prefix);
    addOption("gshell_dim",("Spatial dimension for Gaussian shell likelihood (default "+to_string(dim)+")").c_str(),to_string(dim).c_str());
    addOption("gshell_logx","Transform x to x'=exp(x) to test Log prior.");
    addOption("gshell_one","Just one shell instead of two.");
  };
  virtual void setup(){
    haveSetup();
    ///Set up the output stateSpace for this object
    //set nativeSpace
    *optValue("gshell_dim")>>dim;
    one=optSet("gshell_one");
    logx=optSet("gshell_logx");
    if(logx)one=true;//Log scaling will only work with the one-sided case.
    int npar=dim;
    stateSpace space(npar);
    string names[npar];
    idx_p.resize(dim);
    for(int i=0;i<npar;i++)names[i]="p"+to_string(i);
    space.set_names(names);  
    nativeSpace=space;
    defWorkingStateSpace(nativeSpace);
    best=state(&space,space.size());
    //Set the prior...
    const int uni=mixed_dist_product::uniform, gauss=mixed_dist_product::gaussian, pol=mixed_dist_product::polar, log=mixed_dist_product::log; 
    valarray<double>    centers(0.0,npar);
    valarray<double> halfwidths(6.0,npar);
    valarray<int>         types(uni,npar);
    if(one){
      centers[0]=3.0;
      halfwidths[0]=2.999;//Avoid zero to allow identical comparison with logx
      cout<<"Likelihood has just one Gaussian shell."<<endl;
    }
    double prior_vol=2*halfwidths[0]*pow(2*halfwidths[1],dim-1);
    if(logx){
      //If the shell function is L=f(x,...), we change to L=f(ln(x),...)
      //Then where otherwise we drew x from a uniform prior, now we draw
      //ln(x) uniformly. This should be mathematically equivalent, but
      //it exercises the code differently.
      double max=exp(centers[0]+halfwidths[0]),min=exp(centers[0]-halfwidths[0]);
      halfwidths[0]=sqrt(max/min);
      centers[0]=min*halfwidths[0];
      types[0]=log;
      cout<<"Logarithmic transformation of Gaussian shell."<<endl;
    }
    setPrior(new mixed_dist_product(&nativeSpace,types,centers,halfwidths));
    x0=3.0;
    r0=2;
    double sigma=0.1;
    twosigmasq=2*sigma*sigma;
    lnnormfac=-0.5*std::log(M_PI*twosigmasq);
    int sph_dim=dim-1;
    int n0=1;          //even
    double Acoeff=2;  //case
    if(dim%2==0){//odd case
      cout<<"odd"<<endl;
      n0=2;
      Acoeff*=M_PI;
    }
    for(int n=n0;n+2<=dim;n+=2){
      Acoeff*=M_PI*2.0/n;//recursion formula
      //cout<<" multiplying by "<<M_PI*2.0/n<<endl;
    }
    cout<<"Gaussian Shell area = "<<Acoeff<<"*r^n"<<endl;
    double nsphere_area=Acoeff*pow(r0,sph_dim);
    double like_integral=nsphere_area;//Good approx for r0/sigma>>1 since r-integral is normalized;
    if(not one)like_integral*=2;
    cout<<"Theoretically expected ln(evidence)="<<std::log(like_integral)-std::log(prior_vol)<<endl;
  };
  void defWorkingStateSpace(const stateSpace &sp){
    checkSetup();//Call this assert whenever we need options to have been processed.
    for(int i=0;i<dim;i++)idx_p[i]=sp.requireIndex("p"+to_string(i));
    haveWorkingStateSpace();
  };
  int size()const{return 0;};
  double evaluate_log(state &s){
    valarray<double>params=s.get_params();
    //double result=log_poisson(s);
    double x=params[idx_p[0]];
    if(not one)x=abs(x);//Reflect across x to make two shells
    if(logx){
      if(x<0)return -INFINITY;
      x=std::log(x);
    }
    double dx=x-x0;
    double r2=dx*dx;
    for(int i=1;i<dim;i++){//Already handled distinct dim-0 separately
      dx=params[idx_p[i]];
      r2+=dx*dx;
    }
    dx=sqrt(r2)-r0;
    r2=dx*dx;
    const double sqrt2pi=sqrt(2*M_PI);
    double result = lnnormfac - r2/twosigmasq;
    double post=result;
    post+=nativePrior->evaluate_log(s);//May need a mechanism to check that Prior is set
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
};


shared_ptr<Random> globalRNG;//used for some debugging... 

//***************************************************************************************8
//main test program
int main(int argc, char*argv[]){

  Options opt(true);
  //Create the sampler
  ptmcmc_sampler mcmc;
  bayes_sampler *s0=&mcmc;
  //Create the model components and likelihood;
  //bayes_data *data=new GRBpop_z_only_data();
  //bayes_signal *signal=new GRBpop_one_break_z_signal();
  //bayes_likelihood *like=new constant_likelihood();
  //bayes_likelihood *like=new gaussian_shell_2D_likelihood();
  bayes_likelihood *like=new gaussian_shell_ND_likelihood(2);
  //bayes_likelihood *like=new gaussian_likelihood();
  
  //prep command-line options
  s0->addOptions(opt);
  //data->addOptions(opt);
  //signal->addOptions(opt);
  like->addOptions(opt);

  //Add some command more line options
  opt.add(Option("nchains","Number of consequtive chain runs. Default 1","1"));
  opt.add(Option("seed","Pseudo random number grenerator seed in [0,1). (Default=-1, use clock to seed.)","-1"));
  opt.add(Option("precision","Set output precision digits. (Default 13).","13"));
  opt.add(Option("outname","Base name for output files (Default 'mcmc_output').","mcmc_output"));
  
  int Nlead_args=1;

  bool parseBAD=opt.parse(argc,argv);
  if(parseBAD) {
    cout << "Usage:\n mcmc [-options=vals] " << endl;
    cout <<opt.print_usage()<<endl;
    return 1;
  }
    
  cout<<"flags=\n"<<opt.report()<<endl;

  //Setup likelihood
  //data->setup();  
  //signal->setup();  
  like->setup();
  
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

  //Get the space/prior for use here
  stateSpace space;
  shared_ptr<const sampleable_probability_function> prior;  
  space=*like->getObjectStateSpace();
  cout<<"like.nativeSpace=\n"<<space.show()<<endl;
  prior=like->getObjectPrior();
  cout<<"Prior is:\n"<<prior->show()<<endl;
  valarray<double> scales;prior->getScales(scales);

  //Read Params
  int Npar=space.size();
  cout<<"Npar="<<Npar<<endl;
  
  //Bayesian sampling [assuming mcmc]:
  //Set the proposal distribution 
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
    //s->analyze(base,ic,Nsigma,Nbest,*like);
    delete s;
  }
  
  //Dump summary info
  cout<<"best_post "<<like->bestPost()<<", state="<<like->bestState().get_string()<<endl;
  //delete data;
  //delete signal;
  delete like;
}

