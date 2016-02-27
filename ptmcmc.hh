///Sampler realization for running ptMCMC chains.
///
///Written by John Baker at NASA-GSFC (2015)

#include "bayesian.hh"
#include "chain.hh"
#include "proposal_distribution.hh"

///This is a generic interface for content that was previously implemented in gleam::main
class ptmcmc_sampler : public bayes_sampler {
private:
//These pointers are to objects are (potentially) managed by this class
  chain *cc;
  proposal_distribution *cprop;
  bool have_cc,have_cprop;
  //These pointers are externally owned.
  bayes_likelihood *chain_llike;
  sampleable_probability_function *chain_prior;
  bool have_setup;
  //options params
  int chain_Nstep,chain_Ninit,chain_nburn,output_precision;
  double swap_rate,pt_reboot_rate,pt_reboot_cut,pt_reboot_thermal,pt_reboot_blindly,pt_evolve_rate,pt_evolve_lpost_cut,Tmax;
  int Nstep,Nskip,Nptc,Nevery,save_every,pt_reboot_every,pt_reboot_grace,dump_n;;
  double nburn_frac;
  bool parallel_tempering,pt_reboot_grad;

public:
  static proposal_distribution* new_proposal_distribution(int Npar, int &Ninit, const Options &opt, sampleable_probability_function * prior, const valarray<double>*halfwidths);
  ptmcmc_sampler();
  void addOptions(Options &opt,const string &prefix="");
  int run(const string & base, int ic=0);
  void setup(int Ninit, bayes_likelihood &llike, sampleable_probability_function &prior, proposal_distribution &prop,int output_precision=15);
  int initialize();
  int analyze(const string & base, int ic, int Nsigma, int Nbest, bayes_likelihood &like);
  bayes_sampler * clone(){
    //cout<<"cloning:"<<this<<")"<<endl;
    if(have_cc||have_cprop){
      cout<<"ptmcmc_sampler::clone(): Cannot clone after instantiating chain/prop."<<endl;
      exit(1);
    }
    ptmcmc_sampler* s=new ptmcmc_sampler();
    s->copyOptioned(*this);
    if(have_setup)s->setup(chain_Ninit,*chain_llike,*chain_prior,*cprop,output_precision);
    return s;
  };
  ///must delete managed pointers
  ~ptmcmc_sampler(){
      if(have_cc)delete cc;
    if(have_cprop)delete cprop;
  };
  
private:
  void processOptions();
  };
