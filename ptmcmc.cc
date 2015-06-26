///High-level stuff for overall management of ptMCMC runs
///Written by John Baker at NASA-GSFC (2012-2015)
#include <sstream>
#include <fstream>
#include <iostream>
#include "ptmcmc.hh"
using namespace std;

///Set the proposal distribution. Calling routing responsible for deleting.
///Also returns choice of Ninit in first arg.
///This can be a static routine of ptmcmc_sampler class...
proposal_distribution* ptmcmc_sampler::new_proposal_distribution(int Npar, int &Ninit, const Options &opt, sampleable_probability_function * prior, const valarray<double>*halfwidths){
  int proposal_option,SpecNinit;
  double tmixfac,reduce_gamma_by,de_eps;
  bool de_mixing=false;
  istringstream(opt.value("prop"))>>proposal_option;
  istringstream(opt.value("de_ni"))>>SpecNinit;
  istringstream(opt.value("de_eps"))>>de_eps;
  istringstream(opt.value("de_reduce_gamma"))>>reduce_gamma_by;
  istringstream(opt.value("de_Tmix"))>>tmixfac;
  de_mixing=opt.set("de_mixing");
  valarray<double> sigmas;
  if(halfwidths!=nullptr)sigmas=*halfwidths;
  else if(proposal_option<2){
    cout<<"new_proposal_distribution: Called without defining haflwidths. Cannot apply proposal option 0 or 1."<<endl;
    exit(1);
  }
  Ninit = 1;
  proposal_distribution* prop=nullptr;
  
  //if(parallel_tempering)reduce_gamma_by*=2.0;//Turned this off since mlchain.
  switch(proposal_option){
  case 0:  //Draw from prior distribution   
    cout<<"Selected draw-from-prior proposal option"<<endl;
    prop=new draw_from_dist(*prior);
    break;
  case 1:  //gaussian   
    cout<<"Selected Gaussian proposal option"<<endl;
    prop=new gaussian_prop(sigmas/8.);
    break;
  case 2:  {  //range of gaussians
    int Nprop_set=4;
    cout<<"Selected set of Gaussian proposals option"<<endl;
    vector<proposal_distribution*> gaussN(Nprop_set,nullptr);
    vector<double>shares(Nprop_set);
    double fac=1;
    for(int i=0;i<Nprop_set;i++){
      fac*=2;
      gaussN[i]=new gaussian_prop(sigmas/fac);
      shares[i]=fac;
      cout<<"  sigma="<<sigmas[0]/fac<<", weight="<<fac<<endl;
    }
    prop=new proposal_distribution_set(gaussN,shares);
    break;
  }
  case 3:{
    cout<<"Selected differential evolution proposal option"<<endl;
    //differential_evolution(bool snooker=false, double gamma_one_frac=0.1,double b_small=0.0001,double ignore_frac=0.3):snooker(snooker),gamma_one_frac(gamma_one_frac),b_small(b_small),ignore_frac(ignore_frac)
    //prop=new differential_evolution();
    differential_evolution *de=new differential_evolution(0.0,0.3,de_eps,0.0);
    de->reduce_gamma(reduce_gamma_by);
    if(de_mixing)de->support_mixing(true);
    de->mix_temperatures_more(tmixfac);
    prop=de;
    
    Ninit=SpecNinit*Npar;//Need a huge pop of samples to avoid getting stuck in a peak unless occasionally drawing from prior.
    break;
  }
  case 4:{
    cout<<"Selected differential evolution with snooker updates proposal option"<<endl;
    //differential_evolution(bool snooker=false, double gamma_one_frac=0.1,double b_small=0.0001,double ignore_frac=0.3):snooker(snooker),gamma_one_frac(gamma_one_frac),b_small(b_small),ignore_frac(ignore_frac)
    differential_evolution *de=new differential_evolution(0.1,0.3,de_eps,0.0);
    //differential_evolution *de=new differential_evolution(0.1,0.3,0.0);
    de->reduce_gamma(reduce_gamma_by);
    if(de_mixing)de->support_mixing(true);
    de->mix_temperatures_more(tmixfac);
    prop=de;
    if(false){
      vector<proposal_distribution*>props(1);
      vector<double>shares(1);
      props[0]=de;shares[0]=1.0;
      prop=new proposal_distribution_set(props,shares);    
    }
    Ninit=SpecNinit*Npar;
    break;
  }
  case 5:{
    cout<<"Selected differential evolution proposal with prior draws option"<<endl;
    //differential_evolution(bool snooker=false, double gamma_one_frac=0.1,double b_small=0.0001,double ignore_frac=0.3):snooker(snooker),gamma_one_frac(gamma_one_frac),b_small(b_small),ignore_frac(ignore_frac)
    //prop=new differential_evolution();
    vector<proposal_distribution*>props(2);
    vector<double>shares(2);
    props[0]=new draw_from_dist(*prior);
    shares[0]=0.1;
    differential_evolution *de=new differential_evolution(0.0,0.3,de_eps,0.0);
    de->reduce_gamma(reduce_gamma_by);
    if(de_mixing)de->support_mixing(true);
    de->mix_temperatures_more(tmixfac);
    cout<<"de="<<de<<endl;
    props[1]=de;
    shares[1]=0.9;
    prop=new proposal_distribution_set(props,shares);    
    Ninit=SpecNinit*Npar;
    break;
  }
  case 6:{
    cout<<"Selected differential evolution (with snooker updates) proposal with prior draws option"<<endl;
    //differential_evolution(bool snooker=false, double gamma_one_frac=0.1,double b_small=0.0001,double ignore_frac=0.3):snooker(snooker),gamma_one_frac(gamma_one_frac),b_small(b_small),ignore_frac(ignore_frac)
    //prop=new differential_evolution();
    vector<proposal_distribution*>props(2);
    vector<double>shares(2);
    props[0]=new draw_from_dist(*prior);
    shares[0]=0.1;
    differential_evolution *de=new differential_evolution(0.1,0.3,de_eps,0.0);
    de->reduce_gamma(reduce_gamma_by);
    if(de_mixing)de->support_mixing(true);
    de->mix_temperatures_more(tmixfac);
    props[1]=de;
    shares[1]=0.9;
    prop=new proposal_distribution_set(props,shares);    
    Ninit=SpecNinit*Npar;
    break;
  }
  default:
    cout<<"new_proposal_distribution: Unrecognized value: proposal_option="<<proposal_option<<endl;
    exit(1);
  }
  return prop;
};

ptmcmc_sampler::ptmcmc_sampler(){
  //These pointers are managed by this class
  have_cc=false;
  cc=nullptr;
  have_cprop=false;
  cprop=nullptr;
  //These pointers are externally managed
  chain_llike=nullptr;
  chain_prior=nullptr;
  have_setup=false;
};

void ptmcmc_sampler::addOptions(Options &opt,const string &prefix){
  Optioned::addOptions(opt,prefix);
  addOption("nevery","Frequency to dump chain info. Default=1000.","1000");
  addOption("save_every","Frequency to store chain info. Default=1.","1");
  addOption("nsteps","How long to run the chain. Default=5000.","5000");
  addOption("nskip","Only dump every nskipth element. Default=10.","10");
  addOption("burn_frac","Portion of chain to disregard as burn-in for some calculations. Default=0.5","0.5");
  addOption("pt","Do parallel tempering.");
  addOption("pt_n","Number of parallel tempering chains. Default 20","20");
  addOption("pt_swap_rate","Frequency of parallel tempering swap_trials. Default 0.01","0.01");
  addOption("pt_Tmax","Max temp of parallel tempering chains. Default 100","100");
  addOption("pt_evolve_rate","Rate at which parallel tempering temps should be allowed to evolve. Default none.","0");
  addOption("pt_reboot_rate","Max frequency of rebooting poorly performing parallel tempering chains. Default 0","0");
  addOption("pt_reboot_every","How often to test for rebooting poorly performing parallel tempering chains. Default 0","0");
  addOption("pt_reboot_grace","Grace period protecting infant instances reboot. Default 0","0");
  addOption("pt_reboot_cut","Posterior difference cutoff defining poorly performing parallel tempering chains. Default 100","100");
  addOption("pt_reboot_thermal","Temperature dependent cutoff term in defining poorly performing parallel tempering chains. Default 0","0");
  addOption("pt_reboot_blindly","Do aggressive random rebooting at some level even if no gaps are found. Default 0","0");
  addOption("pt_reboot_grad","Let the reboot grace period depend linearly on temp level with given mean. (colder->longer)");
  addOption("prop","Proposal type (0-6). Default=4 (DE with Snooker updates w/o prior draws.)","4");
  addOption("de_ni","Differential-Evolution number of initialization elements per dimension. Default=10.","10");
  addOption("de_eps","Differential-Evolution gaussian scale. Default=1e-4.","1e-4");
  addOption("de_reduce_gamma","Differential Evolution reduce gamma parameter by some factor from nominal value. Default=1.","1");
  addOption("de_mixing","Differential-Evolution support mixing of parallel chains.");
  addOption("de_Tmix","Differential-Evolution degree to encourage mixing info from different temps.(default=300)","300");
};

void ptmcmc_sampler::processOptions(){
  
  *optValue("nevery")>>Nevery;
  *optValue("save_every")>>save_every;
  *optValue("nsteps")>>Nstep;
  *optValue("nskip")>>Nskip;
  *optValue("burn_frac")>>nburn_frac;
  parallel_tempering=optSet("pt");
  *optValue("pt_n")>>Nptc;
  *optValue("pt_evolve_rate")>>pt_evolve_rate;
  *optValue("pt_reboot_rate")>>pt_reboot_rate;
  *optValue("pt_reboot_every")>>pt_reboot_every;
  *optValue("pt_reboot_grace")>>pt_reboot_grace;
  *optValue("pt_reboot_cut")>>pt_reboot_cut;
  *optValue("pt_reboot_thermal")>>pt_reboot_thermal;
  *optValue("pt_reboot_blindly")>>pt_reboot_blindly;
  pt_reboot_grad=optSet("pt_reboot_grad");
  *optValue("pt_swap_rate")>>swap_rate;
  *optValue("pt_Tmax")>>Tmax;  
};

///Setup specific for the ptmcmc sampler
///
void ptmcmc_sampler::setup(int Ninit,bayes_likelihood &llike, sampleable_probability_function &prior, proposal_distribution &prop, int output_precision_){
  cout<<"SETUP("<<this<<")"<<endl;
  processOptions();
  chain_Nstep=Nstep;
  chain_Ninit=Ninit;
  chain_nburn=Nstep*nburn_frac;
  output_precision=output_precision_;
  cprop=&prop;
  chain_prior=&prior;
  chain_llike = &llike;
  have_setup=true;
}
///Initialization for the ptmcmc sampler
///
int ptmcmc_sampler::initialize(){
  cout<<"INIT("<<this<<")"<<endl;

  if(!have_setup){
    cout<<"ptmcmc_sampler::initialize.  Must call setup() before initialization!"<<endl;
    exit(1);
  }

  //Create the Chain 
  if(parallel_tempering){
    parallel_tempering_chains *ptc= new parallel_tempering_chains(Nptc,Tmax,swap_rate,save_every);
    cc=ptc;
    have_cc=true;
    if(pt_evolve_rate>0)ptc->evolve_temps(pt_evolve_rate);
    if(pt_reboot_rate>0)ptc->do_reboot(pt_reboot_rate,pt_reboot_cut,pt_reboot_thermal,pt_reboot_every,pt_reboot_grace,pt_reboot_grad,pt_reboot_blindly);
    ptc->initialize(chain_llike,chain_prior,chain_Ninit);
  } else {
    MH_chain *mhc= new MH_chain(chain_llike,chain_prior,-30,save_every);
    cc=mhc;
    have_cc=true;
    cprop->set_chain(cc);
    mhc->initialize(chain_Ninit);
  }
  cc->set_proposal(*cprop);
  return 0;
};

int ptmcmc_sampler::run(const string & base, int ic){

  if(!have_cc){
    cout<<"ptmcmc_sampler::run.  Must call initialize() before running!"<<endl;
    exit(1);
  }
  ios_base::openmode mode=ios_base::out;
  if(ic>0)mode=mode|ios_base::app;

  ostringstream ss;
  ss<<base<<".dat";
  ofstream out(ss.str().c_str(),mode);
  out.precision(output_precision);

  cout<<"\nRunning chain "<<ic<<endl;
  //FIXME: add this function in "likelihood" class
  chain_llike->reset();
  
  for(int i=0;i<=chain_Nstep;i++){
    cc->step();
    if(0==i%Nevery){
      cout<<"chain "<<ic<<" step "<<i;
      cout<<" MaxPosterior="<<chain_llike->bestPost()<<endl;
      cc->dumpChain(out,i-Nevery+1,Nskip);
      cout<<cc->status()<<endl;
    }
  }
  out<<"\n"<<endl;
  
  if(parallel_tempering){
    //FIXME  Want to replace this with a generic chain->report() type function...
    ss.str("");ss<<base<<"_PTstats.dat";
    ofstream outp(ss.str().c_str(),mode);
    outp.precision(output_precision);
    dynamic_cast<parallel_tempering_chains*>(cc)->dumpTempStats(outp);
    outp<<"\n"<<endl;
  }
  cout<<"Finished running chain "<<ic<<"."<<endl;
  return 0;
};

///Analyze results.
///This is a hack.  There is halfhearted attempt to get this in the form of a generic interface here.  We are
///folding together stuff that was in the main routine before.  The "generic" interface was then expanded to
///include all the necessary args to make this work.  The basic issue is that we need some details of both class
///chain, and new signal class.
int ptmcmc_sampler::analyze(const string & base, int ic,int Nsigma,int Nbest, bayes_signal &data, double tfinestart, double tfineend){
  if(!have_cc){
    cout<<"ptmcmc_sampler::analyze.  Must call initialize() before analyze()!"<<endl;
    exit(1);
  }
  ios_base::openmode mode=ios_base::out;
  if(ic>0)mode=mode|ios_base::app;
  ostringstream ss;
  //Analysis
  //Select 1-sigma chain points
  vector<int> idx_in_Nsigma;
  ss.str("");ss<<base<<"_1_sigma_samples_"<<ic<<".dat";
  ofstream outsamples(ss.str().c_str());
  ss.str("");ss<<base<<"_1_sigma_samples_fine_"<<ic<<".dat";
  ofstream outfinesamples(ss.str().c_str());
  ss.str("");ss<<base<<"_best_"<<ic<<".dat";
  ofstream outbest(ss.str().c_str());
  ss.str("");ss<<base<<"_best_fine_"<<ic<<".dat";
  ofstream outfinebest(ss.str().c_str());

  ss.str("");ss<<base<<"_"<<Nsigma<<"_sigma.dat";
  ofstream out1sigma(ss.str().c_str(),mode);
  out1sigma.precision(output_precision);

  cc->inNsigma(Nsigma,idx_in_Nsigma,chain_nburn);
  for(int i=0;i<(int)idx_in_Nsigma.size();i+=Nskip){
    int idx=idx_in_Nsigma[i];
    //cout<<"i,idx="<<i<<","<<idx<<":"<<cc->getState(idx,true).get_string()<<endl;
    //cout<<"i,idx="<<i<<","<<idx<<":"<<cc->getState(idx).get_string()<<endl;
    out1sigma<<i<<" "<<idx<<" "<<cc->getLogPost(idx,true)<<": ";
    valarray<double> p(cc->getState(idx,true).get_params());
    //valarray<double> p(cc->getState(idx).get_params());
    for(double p_j:p)out1sigma<<p_j<<" ";
    out1sigma<<endl;
  }
  out1sigma<<endl;
  outsamples.precision(output_precision);
  outfinesamples.precision(output_precision);
  double nfine=data.size()*2;
  
  for(int i=0;i<Nbest;i++){  
    int idx=idx_in_Nsigma[(rand()*idx_in_Nsigma.size())/RAND_MAX];
    state st=cc->getState(idx,true);
    //state st=cc->getState(idx);
    //vector<double> p=st.get_params_vector();
    outsamples<<"#"<<st.get_string()<<endl;
    data.write(outsamples,st);
    outsamples<<endl;
    outfinesamples<<"#"<<st.get_string()<<endl;
    data.write(outfinesamples,st,nfine,tfinestart,tfineend);
    outfinesamples<<endl;
  }

  outbest.precision(output_precision);    
  outfinebest.precision(output_precision);    
  int idx=idx_in_Nsigma[0];
  state st=cc->getState(idx,true);
  //state st=cc->getState(idx);
  //vector<double> p=st.get_params_vector();
  outbest<<"#"<<st.get_string()<<endl;
  data.write(outbest,st);
  outbest<<endl;
  outfinebest<<"#"<<st.get_string()<<endl;
  data.write(outfinebest,st,nfine,tfinestart,tfineend);
  outfinebest<<endl;
  
  cout<<"chain "<<ic<<": best_post "<<chain_llike->bestPost()<<", state="<<chain_llike->bestState().get_string()<<endl;
  return 0;
};


