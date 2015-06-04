///Chains for MCMC
///
///The base class chain, is not useful, and is defined in mcmc.hh
///The Metropolis-Hastings MH_chain and parallel_tempering_chains types should be useful.
///John G Baker - NASA-GSFC (2013-2014)

#include "mcmc.hh"
#include "probability_function.hh"
//#include <memory>
//#include "include/ProbabilityDist.h"
//#include "include/newran.h"
//#include <valarray>
//#include <vector>
//#include <sstream>
//#include <cmath>
//#include <iostream>


class parallel_tempering_chains;
//** useful CHAIN Classes *******

/// A markov (or non-Markovian) chain based on some variant of the Metropolis-Hastings algorithm
/// May add "burn-in" distinction later.
class MH_chain: public chain{
  int Ntries;
  int Naccept;
  int last_type;
  int add_every_N;
  vector<int> types;
  vector<state> states;
  vector<double> lposts;
  vector<double> llikes;
  state current_state;
  double current_lpost,current_llike;
  //  vector<double> chainsteps; 
  vector<double> acceptance_ratio; 
  vector<double> invtemps; 
  //may be more efficient (esp in parallel proc) to change temp, than move chain, so save temp history
  //some algorithms use adaptive temperature
  probability_function *llikelihood;
  sampleable_probability_function *lprior;
  double minPrior;
  proposal_distribution *default_prop;
  bool default_prop_set;
protected:
  int Nhist;
  int get_state_idx(int i=-1);
  double invtemp;
public:
  //virtual ~MH_chain(){};//Assure correct deletion of any child via base pointer
  MH_chain(probability_function * log_likelihood, sampleable_probability_function *log_prior,double minPrior=-30,int add_every_N=1);
  ///If you know how long you are going to run, it can be more efficient to reserve up front, rather than resizing every step  
  virtual void reserve(int nmore);
  virtual int capacity(){return states.capacity();};
  ///Initialize the chain with one or more states.  Technically we do consider these as part of the chain, for output/analysis purposes as they are not selected based on the MH criterion
  void initialize(uint n=1);
  void reboot();
  void add_state(state newstate,double log_like=999,double log_post=999);
  void set_proposal(proposal_distribution &proposal);
  void step();
  void step(proposal_distribution &prop,void *data=nullptr);
  double expectation(double (*test_func)(state s),int Nburn=0);
  double variance(double (*test_func)(state s),double fmean,int Nburn=0);
  state getState(int elem=-1,bool raw_indexing=false)override;
  double getLogPost(int elem=-1,bool raw_indexing=false)override;
  double getLogLike(int elem=-1,bool raw_indexing=false)override;
  //double getLogPrior(int elem=-1,bool raw_indexing=false)override;
  string getStateStringLabels();
  string getStateString();
  void dumpChain(ostream &os,int Nburn=0,int ievery=1);
  //return raw_indexing point after burn-in
  virtual int i_after_burn(int nburn=0){return Ninit+int(nburn/add_every_N);}
  virtual string show();
  virtual string status();
  virtual double invTemp(){return invtemp;};
  void resetTemp(double new_invtemp);
  friend parallel_tempering_chains;
};

// A parallel tempering set of markov (or non-Markovian) chain
// May add "burn-in" distinction later.
class parallel_tempering_chains: public chain{
  //in this implementation, temps are kept with chains
  //states are swapped in exchange steps.
  //General reference: Littenberg-Cornish arxiv:1008.1577
  //Mainly based on KatzenbergerEA06, but for now we don't implement any temperature
  //tuning.  For now we use aspects of Katzenberger's tuning algorithm for diagnostics.
  //Thus we say "replicas" move between chains/temps.
  int Ntemps;
  int add_every_N;
  ///swap rate per chain.  Effective max of 1/(Ntemps-1), ie 1 swap per step total.
  double swap_rate;
  double max_reboot_rate;
  int test_reboot_every;
  double reboot_thresh;
  double reboot_thermal_thresh;
  int reboot_grace;
  int reboot_graduate;
  double reboot_aggression;
  vector<MH_chain> chains;//The actual chains.
  //vector<int> ichains;//Which chain is each replica attached to now
  //Next 3 are diagnostic, based on the auto-tuning alogorithm of KatzenbergerEA06
  //(arXiv:cond-mat/0602085) which has this replica last visited?
  //direction= -1 for 0-temp,+1 for max temp,0 for neither yet,
  vector<int> directions;
  vector<int> ups;//on how many steps has this temp had "up" history
  vector<int> downs;
  vector<int> instances;
  vector<int> instance_starts;
  vector<proposal_distribution*> props;//probably leaking now from props member, maybe should change to unique_ptr<proposal_distribution>
  vector<int> swap_accept_count;
  vector<int> swap_count;
  vector<double> temps;
  vector<double> log_eratio_up,log_eratio_down,tryrate,swaprate,up_frac;
  bool do_evolve_temps;
  double evolve_temp_rate;
  //internal function
  void pry_temps(int ipry, double rate);
  
 public:
  virtual ~parallel_tempering_chains(){  };//assure correct deletion of any child
  parallel_tempering_chains(int Ntemps,int Tmax,double swap_rate=0.01,int add_every_N=1);
  void initialize( probability_function *log_likelihood, sampleable_probability_function *log_prior,int n=1);
  void set_proposal(proposal_distribution &proposal);
  void step();
  ///reference to zero-temerature chain.
  MH_chain & c0(){return chains[0];};
  state getState(int elem=-1,bool raw_indexing=false)override{return c0().getState(elem,raw_indexing);};
  double getLogPost(int elem=-1,bool raw_indexing=false)override{return c0().getLogPost(elem,raw_indexing);};
  double getLogLike(int elem=-1,bool raw_indexing=false)override{return c0().getLogLike(elem,raw_indexing);};
  //double getLogPrior(int elem=-1,bool raw_indexing=false)override{return c0().getLogPrior(elem,raw_indexing);};
  void dumpChain(ostream &os,int Nburn=0,int ievery=1){  dumpChain(0,os,Nburn,ievery);}
  void dumpChain(int ichain,ostream &os,int Nburn=0,int ievery=1){
    chains[ichain].dumpChain(os,Nburn,ievery);
  };
  void dumpTempStats(ostream &os);
  virtual int i_after_burn(int nburn=0){return c0().i_after_burn(nburn);}
  virtual int capacity(){return c0().capacity();};
  virtual string show();
  virtual string status();
  virtual int multiplicity(){return Ntemps;};
  virtual chain* subchain(int index){
    if(index>=0&&index<Ntemps)return &chains[index];
    else {
      cout<<"parallel_tempering_chains::subchain:index out of range. ("<<index<<" of "<<Ntemps<<")"<<endl;
      exit(1);
    }};
  //thermal integration evidence???
  //This function computes the evidence ratio between chains a two different temps;
  double log_evidence_ratio(int ia,int ib,int ilen,int every=1);
  bool evolve_temps(double rate=0.01){
    do_evolve_temps=1;
    evolve_temp_rate=rate;
  };
  void do_reboot(double rate,double threshhold,double thermal,int every,int grace=0,bool graduate=false,double aggression=0){max_reboot_rate=rate;reboot_thresh=threshhold;reboot_thermal_thresh=thermal;test_reboot_every=every;reboot_grace=grace;reboot_aggression=aggression;reboot_graduate=graduate;
    cout<<"Will reboot every "<<" aggression="<<reboot_aggression<<endl;
  };
};  
    
