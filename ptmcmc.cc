///High-level stuff for overall management of ptMCMC runs
///Written by John Baker at NASA-GSFC (2012-2015)
#include <sstream>
#include <fstream>
#include <iostream>
#include "ptmcmc.hh"
using namespace std;

///Set the proposal distribution. Calling routing responsible for deleting.
///Also returns choice of Ninit in first arg.
///This version is a static routine of ptmcmc_sampler class, but should be considered deprecated in favor of select_proposal below
proposal_distribution* ptmcmc_sampler::new_proposal_distribution(int Npar, int &Ninit, const Options &opt, const sampleable_probability_function * prior, const valarray<double>*halfwidths){
  int proposal_option,SpecNinit;
  double tmixfac,reduce_gamma_by,de_eps,gauss_1d_frac;
  bool de_mixing=false;
  istringstream(opt.value("prop"))>>proposal_option;
  istringstream(opt.value("gauss_1d_frac"))>>gauss_1d_frac;
  istringstream(opt.value("de_ni"))>>SpecNinit;
  istringstream(opt.value("de_eps"))>>de_eps;
  istringstream(opt.value("de_reduce_gamma"))>>reduce_gamma_by;
  istringstream(opt.value("de_Tmix"))>>tmixfac;
  de_mixing=opt.set("de_mixing");
  
  return new_proposal_distribution_guts(Npar, Ninit, prior, halfwidths, proposal_option, SpecNinit, tmixfac, reduce_gamma_by, de_eps, gauss_1d_frac,de_mixing);
};

proposal_distribution* ptmcmc_sampler::select_proposal(){
    valarray<double> scales;
    chain_prior->getScales(scales);
    int Npar=chain_prior->get_space()->size();
    int proposal_option,SpecNinit;
    double tmixfac,reduce_gamma_by,de_eps,gauss_1d_frac,gauss_draw_frac,cov_draw_frac;
    bool de_mixing=false;
    bool gauss_temp_scaled=false;
    string covariance_file;
    (*optValue("prop"))>>proposal_option;
    (*optValue("gauss_1d_frac"))>>gauss_1d_frac;
    (*optValue("gauss_draw_frac"))>>gauss_draw_frac;
    (*optValue("cov_draw_frac"))>>cov_draw_frac;
    (*optValue("covariance_file"))>>covariance_file;
    //Do some sanity checking/fixing
    if(gauss_1d_frac<0)gauss_1d_frac=0;
    if(gauss_1d_frac>1)gauss_1d_frac=1;
    if(gauss_draw_frac<0)gauss_draw_frac=0;
    if(gauss_draw_frac>1)gauss_draw_frac=1;
    if(covariance_file=="")cov_draw_frac=0;
    if(cov_draw_frac<0)cov_draw_frac=0;
    if(cov_draw_frac>1)cov_draw_frac=1;
    if(gauss_draw_frac+cov_draw_frac>1){
      double scale=gauss_draw_frac+cov_draw_frac;
      gauss_draw_frac/=scale;
      cov_draw_frac/=scale;
    }

    (*optValue("de_ni"))>>SpecNinit;
    (*optValue("de_eps"))>>de_eps;
    (*optValue("de_reduce_gamma"))>>reduce_gamma_by;
    (*optValue("de_Tmix"))>>tmixfac;
    gauss_temp_scaled=optSet("gauss_temp_scaled");
    de_mixing=optSet("de_mixing");
    cprop=new_proposal_distribution_guts(Npar, chain_Ninit, chain_prior, &scales, proposal_option, SpecNinit, tmixfac, reduce_gamma_by, de_eps, gauss_1d_frac,de_mixing,gauss_draw_frac,cov_draw_frac,gauss_temp_scaled,covariance_file);
    //cprop=new_proposal_distribution(Npar, chain_Ninit, opt, chain_prior, scales);
    cout<<"Proposal distribution is:\n"<<cprop->show()<<endl;
    have_cprop=true;
};

proposal_distribution* ptmcmc_sampler::new_proposal_distribution_guts(int Npar, int &Ninit, const sampleable_probability_function * prior, const valarray<double>*halfwidths,
								      int proposal_option,int SpecNinit,
								      double tmixfac,double reduce_gamma_by,double de_eps,double gauss_1d_frac,
								      bool de_mixing,
								      double gauss_draw_frac, double cov_draw_frac, bool gauss_temp_scaled,
								      const string &covariance_file
								      ){
  valarray<double> sigmas;
  if(halfwidths!=nullptr)sigmas=*halfwidths;
  else if(proposal_option<2){
    cout<<"new_proposal_distribution: Called without defining halfwidths. Cannot apply proposal option 0 or 1."<<endl;
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
    prop=new gaussian_prop(sigmas/8.,gauss_1d_frac,gauss_temp_scaled);
    break;
  case 2:  {  //range of gaussians
    int Nprop_set=4;
    int iprop=0;
    if(cov_draw_frac>0)Nprop_set+=1;
    cout<<"Selected set of Gaussian proposals option"<<endl;
    vector<proposal_distribution*> gaussN(Nprop_set,nullptr);
    vector<double>shares(Nprop_set);
    double fac=1;
    Eigen::MatrixXd covar;
    read_covariance(covariance_file,prior->get_space(),covar);
    if(cov_draw_frac>0)gaussN[iprop]=new gaussian_prop(covar,gauss_1d_frac,gauss_temp_scaled);
    iprop++;
    for(int i=iprop;i<Nprop_set;i++){
      fac*=2;
      gaussN[i]=new gaussian_prop(sigmas/fac,gauss_1d_frac,gauss_temp_scaled);
      shares[i]=fac;
      cout<<"  sigma="<<sigmas[0]/fac<<", weight="<<fac<<endl;
    }
    prop=new proposal_distribution_set(gaussN,shares);
    break;
  }
  case 3:{
    cout<<"Selected differential evolution proposal option"<<endl;
    //c.f. differential_evolution(double snooker=0.0, double gamma_one_frac=0.1,double b_small=0.0001,double ignore_frac=0.3);    vector<proposal_distribution*>props(2);
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
    //c.f. differential_evolution(double snooker=0.0, double gamma_one_frac=0.1,double b_small=0.0001,double ignore_frac=0.3);    vector<proposal_distribution*>props(2);
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
    //cout<<"de="<<de<<endl;
    props[1]=de;
    shares[1]=0.9;
    prop=new proposal_distribution_set(props,shares);    
    Ninit=SpecNinit*Npar;
    break;
  }
  case 6:{
    cout<<"Selected differential evolution (with snooker updates) proposal with prior draws option"<<endl;
    //c.f. differential_evolution(double snooker=0.0, double gamma_one_frac=0.1,double b_small=0.0001,double ignore_frac=0.3);    
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
  case 7:{
    cout<<"Selected differential evolution with snooker updates proposal option including gaussian draws"<<endl;
    //c.f. differential_evolution(double snooker=0.0, double gamma_one_frac=0.1,double b_small=0.0001,double ignore_frac=0.3);    vector<proposal_distribution*>props(2);
    differential_evolution *de=new differential_evolution(0.1,0.3,de_eps,0.0);
    //differential_evolution *de=new differential_evolution(0.1,0.3,0.0);
    de->reduce_gamma(reduce_gamma_by);
    if(de_mixing)de->support_mixing(true);
    de->mix_temperatures_more(tmixfac);
    Ninit=SpecNinit*Npar;
    //plus range of gaussians
    int Nprop_set=7;
    cout<<"Selected set of Gaussian proposals option"<<endl;
    cout<<"  gauss_draw_frac="<<gauss_draw_frac<<"\n  cov_draw_frac="<<cov_draw_frac<<endl;
    vector<proposal_distribution*> set(Nprop_set,nullptr);
    vector<double>shares(Nprop_set);
    int iprop=0;
    set[iprop]=de;
    iprop++;
    Eigen::MatrixXd covar;
    read_covariance(covariance_file,prior->get_space(),covar);
    if(cov_draw_frac>0){
      set[iprop]=new gaussian_prop(covar,gauss_1d_frac,gauss_temp_scaled);
      iprop++;
      Nprop_set+=1;
      set.push_back(nullptr);
    }
    double gshare=gauss_draw_frac;
    shares[0]=1-gshare-cov_draw_frac;
    //double sum=(pow(2,2*(Nprop_set-1)+1)-2)*2/3.0,stepfac=4.0;
    double sum=(pow(2,Nprop_set)-2),stepfac=2.0;
    double fac=1;
    for(int i=iprop;i<Nprop_set;i++){
      fac*=stepfac;
      set[i]=new gaussian_prop(sigmas/100.0/fac,gauss_1d_frac,gauss_temp_scaled);
      shares[i]=fac/sum*gshare;
      cout<<"  sigmas[0]="<<sigmas[0]/100.0/fac<<", weight="<<shares[i]<<endl;
    }
    for(int i=0;i<Nprop_set;i++)cout<<" Option 7 prop: set["<<i<<"] = "<<set[i]->show()<<endl;
      
    prop=new proposal_distribution_set(set,shares);
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
  dump_n=1;
  restarting=false;
  start_time=1e100;//A huge number we are unlikely to surpass
};

///For restartable interface:
void ptmcmc_sampler::checkpoint(string path){
  ostringstream ss;
  ss<<path<<"/step_"<<istep<<"-cp/";
  string dir=ss.str();
  cout<<"Writing checkpoint files to dir:"<<dir<<endl;
  mkdir(dir.data(),ACCESSPERMS);
  ss<<"ptmcmc.cp";
  ofstream os;
  openWrite(os,ss.str());
  cc->checkpoint(dir);
  writeInt(os,istep);
  //cout<<"show:"<<cc->show()<<endl;;
  //cout<<"status:"<<cc->status()<<endl;
}

void ptmcmc_sampler::restart(string path){
  cout<<"Restarting from checkpoint files in dir:"<<path<<endl;
  ostringstream ss;
  ss<<path<<"/";
  string dir=ss.str();
  ss<<"ptmcmc.cp";
  ifstream os;
  openRead(os,ss.str());
  cc->restart(dir);
  readInt(os,istep);
  restarting=false;
  //cout<<"show:"<<cc->show(true)<<endl;;
  //cout<<"status:"<<cc->status()<<endl;
}

///\brief Provide indicative state
///If not initialized, then try to read params from file provided, or else draw a random state
state ptmcmc_sampler::getState(){
  if(!have_setup){
    cout<<"ptmcmc_sampler::getState.  Must call setup() before getState!"<<endl;
    exit(1);
  }
  if(have_cc){
    return cc->getState();
  }
  else if(paramfile==""){
    //cout<<"about to draw from prior="<<endl;
    //cout<<chain_prior->show()<<endl;
    return chain_prior->drawSample(*ProbabilityDist::getPRNG());//Probably should have the sampler own a 'global' RNG from which everything (including this) is derived...
  }
  else {//open file, read params and set state.
    ifstream parfile(paramfile);
    vector<double>pars;
    string line;
    const stateSpace *sp=chain_llike->getObjectStateSpace();
    if(getline(parfile, line)){
      std::istringstream iss(line);
      double parval;
      while (iss>>parval&&pars.size()<sp->size())pars.push_back(parval);
      return state(sp,pars);
    } else {
      cout<<"ptmcmc_sampler::getState: Could not read from file '"<<paramfile<<"'. Quitting."<<endl;
      exit(1);
    }
}
    
};

void ptmcmc_sampler::addOptions(Options &opt,const string &prefix){
  bayes_sampler::addOptions(opt,prefix);
  addOption("checkp_at_step","Step at which to checkpoint and stop","-1");
  addOption("checkp_at_time","Elapsed walltime (hours) after which to checkpoint and stop","-1");
  addOption("restart_dir","Directory with checkpoint data to restart from.","");
  addOption("nevery","Frequency to dump chain info. Default=5000.","5000");
  addOption("save_every","Frequency to store chain info. Default=1.","1");
  addOption("nsteps","How long to run the chain. Default=5000.","5000");
  addOption("nskip","Only dump every nskipth element. Default=10.","10");
  addOption("burn_frac","Portion of chain to disregard as burn-in for some calculations. Default=0.5","0.5");
  addOption("pt","Do parallel tempering.");
  addOption("pt_n","Number of parallel tempering chains. Default 20","20");
  addOption("pt_swap_rate","Frequency of parallel tempering swap_trials. Default 0.01","0.01");
  addOption("pt_Tmax","Max temp of parallel tempering chains. Default 1e6","1e6");
  addOption("pt_evolve_rate","Rate at which parallel tempering temps should be allowed to evolve. Default none.","0");
  addOption("pt_evolve_lpost_cut","Tolerance limit for disordered log-posterior values in temperature evolution. Default no limit.","-1");
  addOption("pt_reboot_rate","Max frequency of rebooting poorly performing parallel tempering chains. Default 0","0");
  addOption("pt_reboot_every","How often to test for rebooting poorly performing parallel tempering chains. Default 0","0");
  addOption("pt_reboot_grace","Grace period protecting infant instances reboot. Default 0","0");
  addOption("pt_reboot_cut","Posterior difference cutoff defining poorly performing parallel tempering chains. Default 100","100");
  addOption("pt_reboot_thermal","Temperature dependent cutoff term in defining poorly performing parallel tempering chains. Default 0","0");
  addOption("pt_reboot_blindly","Do aggressive random rebooting at some level even if no gaps are found. Default 0","0");
  addOption("pt_reboot_grad","Let the reboot grace period depend linearly on temp level with given mean. (colder->longer)");
  addOption("pt_dump_n","How many of the coldest chains to dump; 0 for all. (default 1)","1");
  addOption("pt_stop_evid_err","Set a value to specify a stopping criterion based on evidence consistency. (default 0)","0");
  addOption("prop","Proposal type (0-7). Default=4 (DE with Snooker updates w/o prior draws.)","4");
  addOption("gauss_1d_frac","With Gaussian proposal distribution variants, specify a fraction which should be taken in one random parameter direction. Default=0","0");
  addOption("gauss_draw_frac","With Gaussian proposal distribution variants, specify a fraction of Gaussian draws. Default=0.20","0.20");
  addOption("gauss_temp_scaled","With Gaussian proposal distribution variants, scale (co)variance with chain-temp. Default=not");
  addOption("cov_draw_frac","With Gaussian proposal dist variants and a covariance file set, specify a fraction of Gaussian draws with defined covariance. Default=0.50","0.50");
  addOption("covariance_file","Specify file with covariance data for relevant proposal distribution optoins.Default=none","");
  addOption("de_ni","Differential-Evolution number of initialization elements per dimension. Default=10.","10");
  addOption("de_eps","Differential-Evolution gaussian scale. Default=1e-4.","1e-4");
  addOption("de_reduce_gamma","Differential Evolution reduce gamma parameter by some factor from nominal value. Default=1.","1");
  addOption("de_mixing","Differential-Evolution support mixing of parallel chains.");
  addOption("de_Tmix","Differential-Evolution degree to encourage mixing info from different temps.(default=300)","300");
  addOption("chain_init_file","Specify chain file from which to draw initializtion points, rather than from prior.","");  
  addOption("chain_ess_stop","Stop MCMC sampling the first time the specified effective sample size is reached. (default never)","-1.0");
  
};

void ptmcmc_sampler::processOptions(){
  bayes_sampler::processOptions();
  *optValue("checkp_at_step")>>checkp_at_step;
  *optValue("checkp_at_time")>>checkp_at_time;
  *optValue("restart_dir")>>restart_dir;
  if(not restart_dir.size()==0)restarting=true;
  *optValue("nevery")>>Nevery;
  *optValue("save_every")>>save_every;
  *optValue("nsteps")>>Nstep;
  *optValue("nskip")>>Nskip;
  *optValue("burn_frac")>>nburn_frac;
  parallel_tempering=optSet("pt");
  *optValue("pt_n")>>Nptc;
  if(Nptc>1)parallel_tempering=true;
  *optValue("pt_evolve_rate")>>pt_evolve_rate;
  *optValue("pt_evolve_lpost_cut")>>pt_evolve_lpost_cut;
  *optValue("pt_reboot_rate")>>pt_reboot_rate;
  *optValue("pt_reboot_every")>>pt_reboot_every;
  *optValue("pt_reboot_grace")>>pt_reboot_grace;
  *optValue("pt_reboot_cut")>>pt_reboot_cut;
  *optValue("pt_reboot_thermal")>>pt_reboot_thermal;
  *optValue("pt_reboot_blindly")>>pt_reboot_blindly;
  pt_reboot_grad=optSet("pt_reboot_grad");
  *optValue("pt_swap_rate")>>swap_rate;
  *optValue("pt_Tmax")>>Tmax;  
  *optValue("pt_dump_n")>>dump_n;if(dump_n>Nptc||dump_n<0)dump_n=Nptc;  
  *optValue("pt_stop_evid_err")>>pt_stop_evid_err;  
  *optValue("chain_init_file")>>initialization_file;
  *optValue("chain_ess_stop")>>ess_stop;
  if(checkp_at_time>0)start_time=omp_get_wtime();
};

///Setup specific for the ptmcmc sampler
///
void ptmcmc_sampler::setup(int Ninit,bayes_likelihood &llike, const sampleable_probability_function &prior, proposal_distribution &prop, int output_precision_){
  //cout<<"SETUP("<<this<<")"<<endl;
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

void ptmcmc_sampler::setup(bayes_likelihood &llike, const sampleable_probability_function &prior, int output_precision_){
  //cout<<"SETUP("<<this<<")"<<endl;
  processOptions();
  chain_Nstep=Nstep;
  chain_nburn=Nstep*nburn_frac;
  output_precision=output_precision_;
  chain_prior=&prior;
  chain_llike = &llike;
  have_setup=true;
}

///Initialization for the ptmcmc sampler
///
int ptmcmc_sampler::initialize(){
  //cout<<"INIT("<<this<<")"<<endl;

  if(!have_setup){
    cout<<"ptmcmc_sampler::initialize.  Must call setup() before initialization!"<<endl;
    exit(1);
  }

  int Ninit=chain_Ninit;
  if(restarting or Nstep<=0)Ninit=0;
  //Create the Chain 
  if(parallel_tempering){
    parallel_tempering_chains *ptc= new parallel_tempering_chains(Nptc,Tmax,swap_rate,save_every);
    cc=ptc;
    have_cc=true;
    if(pt_evolve_rate>0)ptc->evolve_temps(pt_evolve_rate,pt_evolve_lpost_cut);
    if(pt_reboot_rate>0)ptc->do_reboot(pt_reboot_rate,pt_reboot_cut,pt_reboot_thermal,pt_reboot_every,pt_reboot_grace,pt_reboot_grad,pt_reboot_blindly);
    ptc->initialize(chain_llike,chain_prior,Ninit,initialization_file);
  } else {
    MH_chain *mhc= new MH_chain(chain_llike,chain_prior,-30,save_every);
    cc=mhc;
    have_cc=true;
    cprop->set_chain(cc);
    if(initialization_file!="")mhc->initialize(Ninit,initialization_file);
    else mhc->initialize(Ninit);
  }
  cc->set_proposal(*cprop);
  return 0;
};

int ptmcmc_sampler::run(const string & base, int ic){

  if(!have_cc&&chain_Nstep>0){
    cout<<"ptmcmc_sampler::run.  Must call initialize() before running!"<<endl;
    exit(1);
  }
  ios_base::openmode mode=ios_base::out;
  if(ic>0 or restarting)mode=mode|ios_base::app;

  if(ic>0 and restarting){
    cout<<"ptmcmc_sampler::run: Can't restart except for single chain ic=0.  A little more coding is required if this is needed."<<endl;
    exit(1);
  }
  
  //ostringstream ss;
  //ss<<base<<".dat";
  ofstream *out=new ofstream[dump_n];
  for(int ich=0;ich<dump_n;ich++){
    ostringstream ssi;
    if(parallel_tempering)ssi<<base<<"_t"<<ich<<".dat";
    else ssi<<base<<".dat";
    out[ich].open(ssi.str().c_str(),mode);
    out[ich].precision(output_precision);
  }
  
  cout<<"\nRunning chain "<<ic<<" for up to "<<chain_Nstep<<" steps."<<endl;
  //FIXME: add this function in "likelihood" class
  chain_llike->reset();
  
  for(istep=0;istep<=chain_Nstep;istep++){//istep is member variable to facilitate checkpointing
    if(restarting)restart(restart_dir);
      
    if(istep==checkp_at_step or ( checkp_at_time>0 and (omp_get_wtime()-start_time)/3600 > checkp_at_time ) ){//checkpointing test
      checkpoint(".");
      exit(0);
    }
    
    cc->step();
    bool stop=false;
    if(0==istep%Nevery){
      cout<<"chain "<<ic<<" step "<<istep<<endl;
      cout<<"   MaxPosterior="<<chain_llike->bestPost()<<endl;
      if(parallel_tempering){
	parallel_tempering_chains *ptc=dynamic_cast<parallel_tempering_chains*>(cc);
	for(int ich=0;ich<dump_n;ich++)ptc->dumpChain(ich,out[ich],istep-Nevery+1,Nskip);
	double bestErr=ptc->bestEvidenceErr();
	if(bestErr<pt_stop_evid_err){
	  stop=true;
	  cout<<"ptmcmc_sampler::run: Stopping based on pt_stop_evid_err criterion."<<endl; 
	}	
      }
      else cc->dumpChain(out[0],istep-Nevery+1,Nskip);
      cout<<cc->status()<<endl;      
      
      if(0==istep%(Nevery*4)){
	cout<<"Effective sample size test"<<endl;
	auto ess_len=cc->report_effective_samples();
	if(ess_stop>0 and ess_len.first>ess_stop){
	  stop=true;
	  cout<<"ptmcmc_sampler::run: Stopping based on chain_ess_stop Effective Sample Size criterion."<<endl; 
	}	
      }
    }
    if(stop)break;
  }
  for(int ich=0;ich<dump_n;ich++)out[ich]<<"\n"<<endl;
  
  if(parallel_tempering){
    //FIXME  Want to replace this with a generic chain->report() type function...
    ostringstream ss("");ss<<base<<"_PTstats.dat";
    ofstream outp(ss.str().c_str(),mode);
    outp.precision(output_precision);
    dynamic_cast<parallel_tempering_chains*>(cc)->dumpTempStats(outp);
    outp<<"\n"<<endl;
  }
  cout<<"Finished running chain "<<ic<<"."<<endl;
  delete [] out;
  return 0;
};

///Analyze results.
///This is a hack.  There is halfhearted attempt to get this in the form of a generic interface here.  We are
///folding together stuff that was in the main routine before.  The "generic" interface was then expanded to
///include all the necessary args to make this work.  The basic issue is that we need some details of both class
///chain, and new signal class.
int ptmcmc_sampler::analyze(const string & base, int ic,int Nsigma,int Nbest, bayes_likelihood &like){
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

  for(int i=0;i<Nbest;i++){  
    int idx=idx_in_Nsigma[(rand()*idx_in_Nsigma.size())/RAND_MAX];
    state st=cc->getState(idx,true);
    //state st=cc->getState(idx);
    //vector<double> p=st.get_params_vector();
    outsamples<<"#"<<st.get_string()<<endl;
    like.write(outsamples,st);
    outsamples<<endl;
    outfinesamples<<"#"<<st.get_string()<<endl;
    like.writeFine(outfinesamples,st);
    outfinesamples<<endl;
  }

  outbest.precision(output_precision);    
  outfinebest.precision(output_precision);    
  int idx=idx_in_Nsigma[0];
  state st=cc->getState(idx,true);
  //state st=cc->getState(idx);
  //vector<double> p=st.get_params_vector();
  outbest<<"#"<<st.get_string()<<endl;
  like.write(outbest,st);
  outbest<<endl;
  outfinebest<<"#"<<st.get_string()<<endl;
  like.writeFine(outfinebest,st);
  outfinebest<<endl;
  
  cout<<"chain "<<ic<<": best_post "<<chain_llike->bestPost()<<", state="<<chain_llike->bestState().get_string()<<endl;
  return 0;
};

void ptmcmc_sampler::read_covariance(const string &file,const stateSpace *ss,Eigen::MatrixXd &covar){

  //Do what;
};

void ptmcmc_sampler::write_covariance(const Eigen::MatrixXd &cov, const stateSpace *ss, const string &file){
  //Do what;
};

