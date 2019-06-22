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
  const sampleable_probability_function *chain_prior;
  bool have_setup;
  //options params
  int chain_Nstep,chain_Ninit,chain_nburn,output_precision;
  double swap_rate,pt_reboot_rate,pt_reboot_cut,pt_reboot_thermal,pt_reboot_blindly,pt_evolve_rate,pt_evolve_lpost_cut,Tmax,pt_stop_evid_err;
  int Nstep,Nskip,Nptc,Nevery,save_every,pt_reboot_every,pt_reboot_grace,dump_n;;
  double nburn_frac;
  bool parallel_tempering,pt_reboot_grad;
  int istep;
  bool restarting;
  string restart_dir;
  int checkp_at_step;
  double checkp_at_time;
  double start_time;
  double ess_stop;
  string initialization_file;
public:
  static void read_covariance(const string &file,const stateSpace *ss,Eigen::MatrixXd &covar);
  static void write_covariance(const Eigen::MatrixXd &cov, const stateSpace *ss, const string &file);
  static proposal_distribution* new_proposal_distribution(int Npar, int &Ninit, const Options &opt, const sampleable_probability_function * prior, const valarray<double>*halfwidths);
  static proposal_distribution* new_proposal_distribution_guts(int Npar, int &Ninit, const sampleable_probability_function * prior, const valarray<double>*halfwidths, int proposal_option,int SpecNinit, double tmixfac,double reduce_gamma_by,double de_eps,double gauss_1d_frac, bool de_mixing=false, double gauss_draw_frac=0.2, double cov_draw_frac=0, bool gauss_temp_scaled=false, const string &covariance_file="");
							       
  //proposal_distribution* select_proposal();
  void select_proposal();
  ptmcmc_sampler();
  virtual void checkpoint(string path)override;
  virtual void restart(string path)override;
  void addOptions(Options &opt,const string &prefix="");
  int run(const string & base, int ic=0);
  void setup(bayes_likelihood &llike, const sampleable_probability_function &prior, int output_precision=15);
  void setup(bayes_likelihood &llike, int output_precision=15){setup(llike,*llike.getObjectPrior(),output_precision);};
  void setup(int Ninit,bayes_likelihood &llike, const sampleable_probability_function &prior, proposal_distribution &prop,int output_precision=15);//deprecated
  int initialize();
  int analyze(const string & base, int ic, int Nsigma, int Nbest, bayes_likelihood &like);
  bayes_sampler * clone(){
    //cout<<"cloning:"<<this<<")"<<endl;
    if(have_cc){
      cout<<"ptmcmc_sampler::clone(): Cannot clone after instantiating chain/prop."<<endl;
      exit(1);
    }
    ptmcmc_sampler* s=new ptmcmc_sampler();
    s->copyOptioned(*this);
    if(have_setup)s->setup(*chain_llike,*chain_prior,output_precision);
    if(have_cprop)s->select_proposal();
    cout<<"prior is:"<<chain_prior->show()<<endl;
    return s;
  };
  ///must delete managed pointers
  ~ptmcmc_sampler(){
    if(have_cc)delete cc;
    if(have_cprop)delete cprop;
  };
  state getState();
  
private:
  void processOptions();

public:
  static void Init(int &argc, char*argv[]);
  static void Quit();
  bool reporting();
};

