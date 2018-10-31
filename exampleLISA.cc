//Simplified likelihood for LISA example.  The simplified likelihood covers only
//extrinsic parameters based on  low-f limit, and short-duration observation
//as occurs for merger of ~1e6 Msun binaries. 
//Written by Sylvain Marsat AEI and John G Baker NASA-GSFC (2017)

//#include "mlfit.hh"
#include <valarray>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <complex>
#include "omp.h"
#include "options.hh"
//#include <mcheck.h>
#include "bayesian.hh"
#include "proposal_distribution.hh"
#include "ptmcmc.hh"
#ifdef USE_MPI
#include <mpi.h>
#endif

using namespace std;

const double MTSUN_SI=4.9254923218988636432342917247829673e-6;
const double PI=3.1415926535897932384626433832795029;
const complex<double> I(0.0, 1.0);
const bool narrowband=true;

// Routines for simplified likelihood 22 mode, frozen LISA, lowf, fixed masses (near 1e6) and fixed t0
static double funcphiL(double m1, double m2, double tRef,double phiRef){
  double MfROMmax22 = 0.14;
  double fRef = MfROMmax22/((m1 + m2) * MTSUN_SI);
  return -phiRef + PI*tRef*fRef;
}
static double funclambdaL(double lambd, double beta) {
  return -atan2(cos(beta)*cos(lambd)*cos(PI/3) + sin(beta)*sin(PI/3), cos(beta)*sin(lambd));
}
static double funcbetaL(double lambd, double beta) {
  return -asin(cos(beta)*cos(lambd)*sin(PI/3) - sin(beta)*cos(PI/3));
}
static double funcpsiL(double lambd, double beta, double psi) {
  return atan2(cos(PI/3)*cos(beta)*sin(psi) - sin(PI/3)*(sin(lambd)*cos(psi) - cos(lambd)*sin(beta)*sin(psi)), cos(PI/3)*cos(beta)*cos(psi) + sin(PI/3)*(sin(lambd)*sin(psi) + cos(lambd)*sin(beta)*cos(psi)));
}
static complex<double> funcsa(double d, double phi, double inc, double lambd, double beta, double psi) {
  complex<double> Daplus = I * (double)( 3./4 * (3 - cos(2*beta)) * cos(2*lambd - PI/3) );
  complex<double> Dacross = I * (3.0*sin(beta) * sin(2*lambd - PI/3));
  complex<double> a22 = 0.5/d * sqrt(5/PI) * pow(cos(inc/2), 4) * exp(2.*I*(-phi-psi)) * 0.5*(Daplus + I*Dacross);
  complex<double> a2m2 = 0.5/d * sqrt(5/PI) * pow(sin(inc/2), 4) * exp(2.*I*(-phi+psi)) * 0.5*(Daplus - I*Dacross);
  return a22 + a2m2;
}
static complex<double> funcse(double d, double phi, double inc, double lambd, double beta, double psi) {
  complex<double> Deplus = -I*(3./4 * (3 - cos(2*beta)) * sin(2*lambd - PI/3));
  complex<double> Decross = I*(3*sin(beta) * cos(2*lambd - PI/3));
  complex<double> e22 = 0.5/d * sqrt(5/PI) * pow(cos(inc/2), 4) * exp(2.*I*(-phi-psi)) * 0.5*(Deplus + I*Decross);
  complex<double> e2m2 = 0.5/d * sqrt(5/PI) * pow(sin(inc/2), 4) * exp(2.*I*(-phi+psi)) * 0.5*(Deplus - I*Decross);
  return e22 + e2m2;
}

double simpleCalculateLogLCAmpPhase(double d, double phiL, double inc, double lambdL, double betaL, double psiL) 
{
  //Simple likelihood for runcan 22 mode, frozen LISA, lowf, snr 200
  // normalization factor and injection values sainj, seinj hardcoded - read from Mathematica
  double factor = 216147.866077;
  complex<double> sainj = 0.33687296665053773 + I*0.087978055005482114;
  complex<double> seinj = -0.12737105239204741 + I*0.21820079314765678;
  //double d = params->distance / injectedparams->distance;
  complex<double> sa = funcsa(d, phiL, inc, lambdL, betaL, psiL);
  complex<double> se = funcse(d, phiL, inc, lambdL, betaL, psiL);
  double simplelogL = -1./2 * factor * (pow(abs(sa - sainj), 2) + pow(abs(se - seinj), 2));
  return simplelogL;
}

///Likelihood function objects
///
// Simplified LISA likelihood 22 mode, frozen LISA, lowf, fixed masses (near 1e6) and fixed t0
class simple_likelihood : public bayes_likelihood {
  int idx_phi,idx_d,idx_inc,idx_lambda,idx_beta,idx_psi;
public:
  simple_likelihood():bayes_likelihood(nullptr,nullptr,nullptr){};
  virtual void setup(){
    haveSetup();
    ///Set up the output stateSpace for this object
    //set nativeSpace
    int npar=6;
    stateSpace space(npar);
    string names[]={"d","phi","inc","lambda","beta","psi"};//double phiRef, double d, double inc, double lambd, double beta, double psi) 
    space.set_names(names);
    space.set_bound(1,boundary(boundary::wrap,boundary::wrap,0,2*M_PI));//set 2-pi-wrapped space for phi.  Turn on if not narrow banding
    if(not narrowband)space.set_bound(3,boundary(boundary::wrap,boundary::wrap,0,2*M_PI));//set 2-pi-wrapped space for lambda.
    //else space.set_bound(4,boundary(boundary::limit,boundary::limit,0.2,0.6));//set narrow limits for beta
    space.set_bound(5,boundary(boundary::wrap,boundary::wrap,0,M_PI));//set pi-wrapped space for pol.
    cout<<"Parameter space:\n"<<space.show()<<endl;
    
    nativeSpace=space;
    defWorkingStateSpace(nativeSpace);
    best=state(&space,space.size());
    //Set the prior...

    const int uni=mixed_dist_product::uniform, gauss=mixed_dist_product::gaussian, pol=mixed_dist_product::polar, cpol=mixed_dist_product::copolar, log=mixed_dist_product::log; 
    valarray<double> centers((initializer_list<double>){  1.667,  PI, PI/2,  PI,    0, PI/2});
    valarray<double>  scales((initializer_list<double>){  1.333,  PI, PI/2,  PI, PI/2, PI/2});
    valarray<int>      types((initializer_list<int>)   {    uni, uni, pol, uni, cpol,  uni});
    if(narrowband){
      centers[3] = 1.75*PI;scales[3] = PI/4.0;
      //centers[4] = 0.4    ;scales[4] = 0.2;
      centers[4] = PI/4    ;scales[4] = PI/4;
    }
    setPrior(new mixed_dist_product(&nativeSpace,types,centers,scales));
  };
  void defWorkingStateSpace(const stateSpace &sp){
    checkSetup();//Call this assert whenever we need options to have been processed.
    idx_d=sp.requireIndex("d");
    idx_phi=sp.requireIndex("phi");
    idx_inc=sp.requireIndex("inc");
    idx_lambda=sp.requireIndex("lambda");
    idx_beta=sp.requireIndex("beta");
    idx_psi=sp.requireIndex("psi");
    haveWorkingStateSpace();
  };
  int size()const{return 0;};
  double evaluate_log(state &s){
    valarray<double>params=s.get_params();
    double d=params[idx_d];
    double phi=params[idx_phi];
    double inc=params[idx_inc];
    double lambd=params[idx_lambda];
    double beta=params[idx_beta];
    double psi=params[idx_psi];
    double result=simpleCalculateLogLCAmpPhase(d, phi, inc, lambd, beta, psi);
    double post=result;
    post+=nativePrior->evaluate_log(s);//May need a mechanism to check that prior is set
    #pragma omp critical
    {     
      if(post>best_post){
        best_post=post;
        best=state(s);
      }
      if(!isfinite(post)){
        cout<<"Whoa dude, logpost is NAN! What's up with that?"<<endl;
        cout<<"params="<<s.get_string()<<endl;
	cout<<"like="<<result<<"  post="<<post<<endl; 
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

#ifdef USE_MPI
  MPI_Init( &argc, &argv );
  int myproc,nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if(true or myproc==0)cout<<"MPI running on "<<nproc<<" MPI processes."<<endl;
#endif

  Options opt(true);
  //Create the sampler
  ptmcmc_sampler mcmc;
  bayes_sampler *s0=&mcmc;
  //Create the model components and likelihood;
  //bayes_data *data=new GRBpop_z_only_data();
  //bayes_signal *signal=new GRBpop_one_break_z_signal();
  bayes_likelihood *like=new simple_likelihood();
  
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
  //mcmc.setup(Ninit,*like,*prior,*prop,output_precision);
  mcmc.setup(*like,*prior,output_precision);
  mcmc.select_proposal();

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
#ifdef USE_MPI
  MPI_Finalize();
#endif

}

