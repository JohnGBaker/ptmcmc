///High-level stuff for overall management of ptMCMC runs
///Written by John Baker at NASA-GSFC (2012-2015)
#include <sstream>
#include <fstream>
#include <iostream>
#include "ptmcmc.hh"
#ifdef USE_MPI
#include <mpi.h>
#endif
#include "test_proposal.hh"
#include <csignal>

using namespace std;

void ptmcmc_sampler::select_proposal(){
    //valarray<double> scales;
    //chain_prior->getScales(scales);
    vector<double> scalesvec;chain_llike->getScales(scalesvec);
    valarray<double> scales(scalesvec.data(), scalesvec.size());
    if(reporting())cout<<"Selecting proposal for stateSpace:\n"<<chain_prior->get_space()->show()<<endl;
    int Npar=chain_prior->get_space()->size();
    int proposal_option,SpecNinit;
    double tmixfac,reduce_gamma_by,de_g1_frac,de_eps,gauss_1d_frac,prior_draw_frac,prior_draw_Tpow,gauss_draw_frac,gauss_step_fac,cov_draw_frac,sym_prop_frac,like_prop_frac,unlikely_alpha;
    bool de_mixing=false;
    bool gauss_temp_scaled=false;
    string covariance_file;
    //(*optValue("prop"))>>proposal_option;
    (*optValue("prior_draw_frac"))>>prior_draw_frac;
    (*optValue("prior_draw_Tpow"))>>prior_draw_Tpow;
    (*optValue("gauss_1d_frac"))>>gauss_1d_frac;
    (*optValue("gauss_draw_frac"))>>gauss_draw_frac;
    (*optValue("gauss_step_fac"))>>gauss_step_fac;
    (*optValue("cov_draw_frac"))>>cov_draw_frac;
    (*optValue("sym_prop_frac"))>>sym_prop_frac;
    (*optValue("like_prop_frac"))>>like_prop_frac;
    (*optValue("covariance_file"))>>covariance_file;
    bool adapt_more=optSet("prop_adapt_more");
    //Do some sanity checking/fixing
    if(prior_draw_frac<0)prior_draw_frac=0;
    if(prior_draw_frac>1)prior_draw_frac=1;
    if(gauss_1d_frac<0)gauss_1d_frac=0;
    if(gauss_1d_frac>1)gauss_1d_frac=1;
    if(gauss_draw_frac<0)gauss_draw_frac=0;
    if(gauss_draw_frac>1)gauss_draw_frac=1;
    if(gauss_step_fac<1)gauss_step_fac=1;
    if(covariance_file=="")cov_draw_frac=0;
    if(cov_draw_frac<0)cov_draw_frac=0;
    if(cov_draw_frac>1)cov_draw_frac=1;
    if(prior_draw_frac+gauss_draw_frac+cov_draw_frac>1){
      double scale=gauss_draw_frac+cov_draw_frac;
      gauss_draw_frac/=scale;
      cov_draw_frac/=scale;
    }

    (*optValue("de_ni"))>>SpecNinit;
    (*optValue("de_eps"))>>de_eps;
    (*optValue("de_reduce_gamma"))>>reduce_gamma_by;
    (*optValue("de_g1_frac"))>>de_g1_frac;
    (*optValue("de_Tmix"))>>tmixfac;
    (*optValue("de_unlikely_alpha"))>>unlikely_alpha;
    gauss_temp_scaled=optSet("gauss_temp_scaled");
    de_mixing=optSet("de_mixing");



    //Guts of proposal set up, formerly "proposal_option 7" now generalized
    int Ng=6;
    int Nprop_set=1+Ng;
    //If adaptive proposal mixing, then we only want to adapt on the gaussian portion, so we make the set hierachically
    if(prop_adapt_rate>0){//hierarchical, so this is just the top level
      Nprop_set=2;
    }
    vector<proposal_distribution*> set(Nprop_set,nullptr);;
    vector<double>shares(Nprop_set);
    vector<double>hot_shares(Nprop_set);
    double Tpow=0;
    int iprop=0;
    double gshare=gauss_draw_frac;

    //Differential_evolution component    
    differential_evolution *de=new differential_evolution(0.1,de_g1_frac,de_eps,0.0,unlikely_alpha);
    //differential_evolution *de=new differential_evolution(0.1,0.3,0.0);
    de->reduce_gamma(reduce_gamma_by);
    if(de_mixing)de->support_mixing(true);
    de->mix_temperatures_more(tmixfac);
    chain_Ninit=SpecNinit*Npar;
    
    shares[0]=1-gshare-cov_draw_frac-prior_draw_frac;
    if(shares[0]<0)shares[0]=0;    
    set[iprop]=de;
    iprop++;
    
    //optionally add prior draws
    if(prior_draw_frac>0){		 
      set[iprop]=new draw_from_dist(*chain_prior);
      set.push_back(nullptr);
      shares.push_back(0);
      shares[iprop]=prior_draw_frac;
      hot_shares[iprop]=1;
      hot_shares.push_back(0);
      Tpow=prior_draw_Tpow;
      iprop++;
      Nprop_set++;
    }
    //Next optionally add a specific covariance gaussian (top level if hierarchical)
    Eigen::MatrixXd covar;
    read_covariance(covariance_file,chain_prior->get_space(),covar);
    if(cov_draw_frac>0){
      set[iprop]=new gaussian_prop(covar,gauss_1d_frac,gauss_temp_scaled);	
      set.push_back(nullptr);
      shares.push_back(0);
      hot_shares.push_back(0);
      shares[iprop]=cov_draw_frac;
      iprop++;
      Nprop_set+=1;
    }
    //double sum=(pow(2,2*(Nprop_set-1)+1)-2)*2/3.0,stepfac=4.0;
    vector<proposal_distribution*> gset(Ng,nullptr);
    vector<double> gshares(Ng);
    double sum=(pow(2,Ng+1)-2),stepfac=gauss_step_fac;
    double fac=pow(2.0/gauss_step_fac,4.0);
    double sharefac=1;
    if(prop_adapt_rate>0){
      for(int i=0;i<Ng;i++){
	fac*=stepfac;
	gset[i]=new gaussian_prop(scales/100.0/fac,gauss_1d_frac,gauss_temp_scaled);
	sharefac*=2;
	gshares[i]=sharefac/sum;
      }
      set[iprop]=new proposal_distribution_set(gset,gshares,prop_adapt_rate);
      shares[iprop]=gshare;
    } else {
      for(int i=iprop;i<Nprop_set;i++){
	fac*=stepfac;
	set[i]=new gaussian_prop(scales/100.0/fac,gauss_1d_frac,gauss_temp_scaled);
	sharefac*=2;
	shares[i]=sharefac/sum*gshare;
      }
    }
    if(adapt_more)
      cprop=new proposal_distribution_set(set,shares,prop_adapt_rate,Tpow,hot_shares);
    else
      cprop=new proposal_distribution_set(set,shares,0,Tpow,hot_shares);
    
    //TODO: Incorporate this into the existing top-level set above
    //if(reporting())cout<<"sym_prop_frac="<<sym_prop_frac<<" nsym="<<chain_prior->get_space()->get_potentialSyms().size()<<endl;
    
    bool add_on=false;
    vector<proposal_distribution*> add_on_props={cprop};
    vector<double> add_on_shares={1};
    if(sym_prop_frac>0 and chain_prior->get_space()->get_potentialSyms().size()>0){
      if(reporting())cout<<"Adding stateSpace potential symmetries to proposal."<<endl;
      double rate=0;
      if(adapt_more)rate=prop_adapt_rate;
      proposal_distribution_set *symprops=involution_proposal_set(*chain_prior->get_space(),rate).clone();
      add_on_props.push_back(symprops);
      add_on_shares.push_back(sym_prop_frac);
      add_on_shares[0]-=sym_prop_frac;
      add_on=true;
    }
    if(like_prop_frac>0 and chain_llike->get_proposals().size()>0){
      if(reporting())cout<<"Adding likelihood-based elements to proposal."<<endl;
      double rate=0;
      if(adapt_more)rate=prop_adapt_rate;
      proposal_distribution_set *likeprops=new proposal_distribution_set(chain_llike->get_proposals(),chain_llike->get_prop_shares(),rate,0,vector<double>(),false);//likelihood retains possession of these pointers
      add_on_props.push_back(likeprops);
      add_on_shares.push_back(like_prop_frac);
      add_on_shares[0]-=like_prop_frac;
      add_on=true;
    }
    if(add_on){
      if(add_on_shares[0]<0)add_on_shares[0]=0;
      cprop=new proposal_distribution_set(add_on_props,add_on_shares,prop_adapt_rate);
    }
    //cprop=new_proposal_distribution(Npar, chain_Ninit, opt, chain_prior, scales);
    if(reporting())cout<<"Proposal distribution is:\n"<<cprop->show()<<endl;
    have_cprop=true;
}

void ptmcmc_sampler::test_prop(){
    //Perform propopsal testing if called for
    string indexstring;(*optValue("prop_test_index"))>>indexstring;
    //cout<<"indexstring='"<<indexstring<<"'"<<endl;
    if(indexstring.length()==0)return;
    //Look for "+" at the end to to potentially increase rigor of the testing
    int rigor=0;
    while((not indexstring.empty()) and indexstring.back()=='+'){
      rigor++;
      indexstring.pop_back();
    }
    //Check for L to indicate looping
    bool looping=false;
    if((not indexstring.empty()) and indexstring.back()=='L'){
      looping=true;
      indexstring.pop_back();
    }
    //Process multiindex
    vector<int> multiindex;
    if(indexstring!="." and not indexstring.empty()){
      //Parse the index string
      stringstream ss(indexstring);
      string elem;
      while(getline(ss,elem,'-'))
	multiindex.push_back(atoi(elem.c_str()));
      cout<<" Testing proposal starting at multiindex:";
      for(int i:multiindex)cout<<" "<<i;
      cout<<endl;
    }
    //Next generate the test distribution
    bool fits=false;
    vector<double> cent;
    vector<double> sigma;
    chain_prior->getScales(sigma);
    while(not fits){
      state s0=chain_prior->drawSample(*ProbabilityDist::getPRNG());
      int n=s0.size();
      cent=s0.get_params_vector();
      vector<double>topv(n),bottomv(n);
      double nsigcut=1.0;
      for(int i=0;i<n;i++){
	topv[i]=cent[i]+sigma[i]*nsigcut;
	bottomv[i]=cent[i]-sigma[i]*nsigcut;
      }
      state top(s0.getSpace(),topv);
      state bottom(s0.getSpace(),bottomv);
      double logcut=-100;
      cout<<"Checking limits: \ntop="<<top.get_string()<<"\nbottom="<<bottom.get_string()<<endl;
      if(chain_prior->evaluate_log(top)>logcut and chain_prior->evaluate_log(bottom)>logcut)fits=true;
      else for(auto &s:sigma)s/=2;
      
      if(fits){
	gaussian_dist_product dist(chain_prior->get_space(),cent,sigma,true);
	cout<<"Test distribution is: "<<dist.show()<<endl;
	int fac=pow(2,rigor);
	int samples=10000*fac,ncyc=500*fac,tries=10*fac;	
	cout<<"Testing on "<<samples<<" samples, applying proposal through "<<ncyc<<" cycles and calibrating with "<<tries<<" trials."<<endl;
	test_proposal testprop(*cprop,dist,looping,"./",multiindex);
	fits=testprop.test(samples,ncyc,tries,0,0.005);
	if(not fits)for(auto &s:sigma)s*=4;
      }
    }
};

void ptmcmc_sampler::Init(int &argc, char*argv[]){return ptmcmc_sampler::Init();}
void ptmcmc_sampler::Init(){
#ifdef USE_MPI
  //MPI_Init( &argc, &argv );
  MPI_Init( nullptr, nullptr);
  int myproc,nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if(myproc==0)cout<<"MPI running on "<<nproc<<" MPI processes.\n"<<endl;
#else
  cout<<"Running without MPI (not compiled for MPI)."<<endl;
#endif
};

void ptmcmc_sampler::Quit(){
#ifdef USE_MPI
  MPI_Finalize();
#endif
  exit(0);
};

bool ptmcmc_sampler::reporting(){return ptmcmc_sampler::static_reporting();};

bool ptmcmc_sampler::static_reporting(){
  bool report=true;
#ifdef USE_MPI
  int myproc,nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  report= myproc==0;
#endif
  return report;
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
  if(cc->outputAllowed()){
    cout<<"Writing checkpoint files to dir:"<<dir<<endl;
    mkdir(dir.data(),ACCESSPERMS);
    ss<<"ptmcmc.cp";
    ofstream os;
    openWrite(os,ss.str());
    cc->checkpoint(dir);
    writeInt(os,istep);
  } else {
    cc->checkpoint(dir);
  }
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
  cout<<"Restart proposal report:\n"<<cc->report_prop(1)<<"\nacceptance report:\n"<<cc->report_prop(0)<<endl;	  
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
  //addOption("pt","Do parallel tempering.");
  addOption("pt","Number of parallel tempering chains. Default off.","0");
  addOption("pt_swap_rate","Frequency of parallel tempering swap_trials. Default 0.10","0.10");
  addOption("pt_Tmax","Max temp of parallel tempering chains. Default 1e9","1e9");
  addOption("pt_evolve_rate","Rate at which parallel tempering temps should be allowed to evolve. Default none.","0.01");
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
  addOption("prop","Defunct.","");
  addOption("gauss_1d_frac","With Gaussian proposal distribution variants, specify a fraction which should be taken in one random parameter direction. Default=0.5","0.5");
  addOption("gauss_draw_frac","With Gaussian proposal distribution variants, specify a fraction of Gaussian draws. Default=0.20","0.20");
  addOption("prior_draw_frac","Add prior draws to general proposal (prop7). Default=0","0");
  addOption("prior_draw_Tpow","Power for thermal_weighting of any prior draws in proposal. Default=0","0");
  addOption("gauss_step_fac","With Gaussian proposal distribution variants, specify scale-spacing of Gaussian components. Default=2","2");
  addOption("gauss_temp_scaled","With Gaussian proposal distribution variants, scale (co)variance with chain-temp. Default=not");
  addOption("cov_draw_frac","With Gaussian proposal dist variants and a covariance file set, specify a fraction of Gaussian draws with defined covariance. Default=0.50","0.50");
  addOption("covariance_file","Specify file with covariance data for relevant proposal distribution optoins.Default=none","");
  addOption("prop_adapt_rate","Specify a scaling rate (eg 1e-3) for adaptation of sub-proposal fractions, Default=0","0");
  addOption("prop_adapt_more","Adapt more broadly, not just Gaussian mixtures");
  addOption("sym_prop_frac","Fractional rate at which to apply and stateSpace symmetries as proposals. (Default=0)","0"); 
  addOption("like_prop_frac","Fractional rate at which to apply proposal defined by likelihood. (Default=0)","0"); 
  addOption("de_ni","Differential-Evolution number of initialization elements per dimension. Default=50.","50");
  addOption("de_eps","Differential-Evolution gaussian scale. Default=1e-4.","1e-4");
  addOption("de_reduce_gamma","Differential Evolution reduce gamma parameter by some factor from nominal value. Default=4.","4");
  addOption("de_g1_frac","Differential Evolution reduce fraction of times gamma parameter set to 1. Default=0.3.","0.3");
  addOption("de_mixing","Differential-Evolution support mixing of parallel chains.");
  addOption("de_Tmix","Differential-Evolution degree to encourage mixing info from different temps.(default=300)","300");
  addOption("de_unlikely_alpha","Scaling power for rejecting unlikely past states in differential evolution draws.(Default 0, ie none)","0");
  
  addOption("prop_test_index","String providing (multi-)index value indicating proposal to test. Append L to loop and +s for more rigor. (eg '.' for all, '0-1L+' for rigorous test looping over sub-proposals below second member of first member of nested set, Default: no test)","");
  addOption("chain_init_file","Specify chain file from which to draw initializtion points, rather than from prior.","");  
  addOption("chain_ess_stop","Stop MCMC sampling the first time the specified effective sample size is reached. (default never)","-1.0");
  addOption("chain_ess_limit","Look for efficiencies in ESS calculation based on assumed limit. (default -1=no limit, 0=based on ess_limit,or as given)","-1.0");
  addOption("checkp_on_sigterm_within","Set to n to check every n steps for SIGTERM and checkpoint if recieved. (Default 0, don't checkpoint on SIGTERM","0");  
  addOption("chain_dprior_min","Specify a minimum change in log prior beyond which proposal will be rejected without evaluating the likelihood. (Default=-30)","-30");  
};


void ptmcmc_sampler::processOptions(){
  bayes_sampler::processOptions();
  *optValue("checkp_at_step")>>checkp_at_step;
  *optValue("checkp_at_time")>>checkp_at_time;
  *optValue("checkp_on_sigterm_within")>>checkp_on_sigterm_every;
  *optValue("restart_dir")>>restart_dir;
  if(not restart_dir.size()==0)restarting=true;
  *optValue("nevery")>>Nevery;
  *optValue("save_every")>>save_every;
  *optValue("nsteps")>>Nstep;
  *optValue("nskip")>>Nskip;
  *optValue("burn_frac")>>nburn_frac;
  //parallel_tempering=optSet("pt");
  *optValue("pt")>>Nptc;
  if(Nptc>1)parallel_tempering=true;
  else parallel_tempering=false;
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
  *optValue("pt_dump_n")>>dump_n;
  if(dump_n>Nptc||dump_n<0)dump_n=Nptc;
  if(Nptc==0)dump_n=1;
  (*optValue("prop_adapt_rate"))>>prop_adapt_rate;
  *optValue("pt_stop_evid_err")>>pt_stop_evid_err;  
  *optValue("chain_init_file")>>initialization_file;
  *optValue("chain_ess_stop")>>ess_stop;
  *optValue("chain_dprior_min")>>dpriormin;
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
  have_cprop=true;
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
  
  if(!have_setup or !have_cprop){
    cout<<"ptmcmc_sampler::initialize.  Must call setup() and set proposal before initialization!"<<endl;
    exit(1);
  }
  if(reporting())cout<<"ptmcmc_sampler: Initializing with prior:\n"<<chain_prior->show()<<endl;
  int Ninit=chain_Ninit;
  if(restarting or Nstep<=0)Ninit=0;
  //Create the Chain 
  if(parallel_tempering){
    parallel_tempering_chains *ptc= new parallel_tempering_chains(Nptc,Tmax,swap_rate,save_every,pt_stop_evid_err>0,pt_stop_evid_err>0,dpriormin);
    cc=ptc;
    have_cc=true;
    if(pt_evolve_rate>0)ptc->evolve_temps(pt_evolve_rate,pt_evolve_lpost_cut);
    if(pt_reboot_rate>0)ptc->do_reboot(pt_reboot_rate,pt_reboot_cut,pt_reboot_thermal,pt_reboot_every,pt_reboot_grace,pt_reboot_grad,pt_reboot_blindly);
    ptc->initialize(chain_llike,chain_prior,Ninit,initialization_file);
  } else {
    MH_chain *mhc= new MH_chain(chain_llike,chain_prior,dpriormin,save_every);
    cc=mhc;
    have_cc=true;
    //cprop->set_chain(cc);
    if(initialization_file!="")mhc->initialize(Ninit,initialization_file);
    else mhc->initialize(Ninit);
  }
  cc->set_proposal(*cprop);
  cprop->set_chain(cc);
  //cout<<"About to test"<<endl;
  test_prop();
  return 0;
};

int ptmcmc_sampler::run(const string & base, int ic){
  if(reporting())cout<<"ptmcmc_sampler:running with omp_num_threads="<<omp_get_num_threads()<<endl;
  
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
    if(cc->outputAllowed())out[ich].open(ssi.str().c_str(),mode);
    out[ich].precision(output_precision);
  }
  
  if(reporting())cout<<"\nRunning chain "<<ic<<" for up to "<<chain_Nstep<<" steps."<<endl;
  //FIXME: add this function in "likelihood" class
  chain_llike->reset();

  //Trigger processing of SIGTERM (possibly) for checkpointing
  if(checkp_on_sigterm_every>0)signal(SIGINT, ptmcmc_sampler::handle_sigterm);
  
  for(istep=0;istep<=chain_Nstep;istep++){//istep is member variable to facilitate checkpointing
    if(restarting)restart(restart_dir);

    //checkpointing test
    bool checkpoint_now=(istep==checkp_at_step);
    bool mpichecknow=false;
    if (checkp_at_time>0 and (!checkpoint_now) and istep%100==0){
      //occasionally check time condition
      checkpoint_now=( (omp_get_wtime()-start_time)/3600 > checkp_at_time );
      mpichecknow=true;
    }
    if (checkp_on_sigterm_every>0 and istep%checkp_on_sigterm_every==0){
      //check if need to stop because a process has recieved SIGTERM
      checkpoint_now = checkpoint_now or terminate_signaled;
      mpichecknow=true;
    }
#ifdef USE_MPI
    if(mpichecknow){
      int nproc;
      MPI_Comm_size(MPI_COMM_WORLD, &nproc);
      if(nproc>1){
	//We have a time-dependent test.
	//It is possible that clocks aren't exactly synchronized or that
	//some procs reach this point at different times so we have to
	//make sure that all procs try to checkpoint at the same step.
	bool checkpoint_now_this_proc=checkpoint_now;
	MPI_Allreduce(&checkpoint_now_this_proc,&checkpoint_now,1,MPI_C_BOOL,MPI_LOR,MPI_COMM_WORLD);
      }
#endif   
    }   
    if(checkpoint_now){
      if(reporting())cout<<"Checkpointing triggered."<<endl;
      checkpoint(".");
      Quit();
    }
    
    cc->step();
    bool stop=false;
    if(0==istep%Nevery){
      if(reporting()){
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
	} else {
	  cc->dumpChain(out[0],istep-Nevery+1,Nskip);
	}
      }
      string ccstatus=cc->status();
      if(reporting())cout<<ccstatus<<endl;      
      
      if(0==istep%(Nevery*4)){
	if(prop_adapt_rate>0){
	  //Proposal report?
	  int nproc=1,myproc=0;
	  #ifdef USE_MPI
	  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
          #endif
	  if(nproc>1){
	    if(istep==0 and reporting())cout<<"Limited proposal reports with multiple MPI procs."<<endl;
	  }
	}
	if(reporting())
	  cout<<"Proposal report:\n"<<cc->report_prop(1)<<"\nacceptance report:\n"<<cc->report_prop(0)<<endl;	  
	
	double esslimit;
	*optValue("chain_ess_limit")>>esslimit;
	if(esslimit==0){
	  if(ess_stop>0)esslimit=2.0*ess_stop;
	  else esslimit=-1;
	}
	if(reporting()){
	  cout<<"Effective sample size test";
	  if(esslimit>0)cout<<" (assuming ess<"<<esslimit<<" for efficiency).";
	  cout<<endl;
	}
	auto ess_len=cc->report_effective_samples(-1,save_every*1000,save_every,esslimit);
	if(ess_stop>0 and ess_len.first>ess_stop){
	  stop=true;
	  cout<<"ptmcmc_sampler::run: Stopping based on chain_ess_stop Effective Sample Size criterion."<<endl; 
	}	
      }
    }
    //Need to broadcast any STOP decision to all procs with MPI
#ifdef USE_MPI
    int nproc;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    if(nproc>1){
      MPI_Bcast(&stop,1,MPI_C_BOOL,0,MPI_COMM_WORLD);
    }
#endif
    if(stop)break;
  }

  if(checkp_on_sigterm_every>0)signal(SIGINT, SIG_DFL);// resort default signal handling

  for(int ich=0;ich<dump_n;ich++)out[ich]<<"\n"<<endl;
  
  if(false and parallel_tempering){ //disabled to reduce output (and deprecated)
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

bool ptmcmc_sampler::terminate_signaled=false;
void ptmcmc_sampler::handle_sigterm(int signum){
  cout<<"ptmcmc: Recieved termination signal."<<endl;
  terminate_signaled=true;
};

void ptmcmc_sampler::read_covariance(const string &file,const stateSpace *ss,Eigen::MatrixXd &covar){

  //Do what;
};

void ptmcmc_sampler::write_covariance(const Eigen::MatrixXd &cov, const stateSpace *ss, const string &file){
  //Do what;
};


