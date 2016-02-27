///Chains for MCMC
///
///The base class chain, is not useful, and is defined in mcmc.hh
///The Metropolis-Hastings MH_chain and parallel_tempering_chains types should be useful.
///John G Baker - NASA-GSFC (2013-2014)
#ifndef CHAIN_HH
#define CHAIN_HH

#include "states.hh"
#include "probability_function.hh"
//#include "probability_function.hh"
//#include <memory>
//#include "include/ProbabilityDist.h"
//#include "include/newran.h"
//#include <valarray>
//#include <vector>
//#include <sstream>
//#include <cmath>
//#include <iostream>


class proposal_distribution;

/// A generalized chain has results like a chain, but we specify
/// anything about *how* those results are produced.  Standard analysis can be applied. 
class chain {
  //TODO?
  //autocorrelation
  //convergence,MAP,...
  static int idcount;
protected:
  int id;
  int Nsize,Ninit; //Maybe move this history stuff to a subclass
  int Nfrozen;
  int dim;
  shared_ptr<Random> rng; //Random number generator for this chain. Each chain has its own so that the results can be threading invariant
  //This function is defined for just for sorting below
  //static bool AgtB(const pair<double,double> & A,const pair<double,double>&B){return A.second>B.second;};
  double get_uniform(){
    double x=rng->Next(); //pick a number
    //rngout<<x<<" 1"<<endl;
    //#pragma omp critical
    //cout<<id<<":"<<x<<endl;//" rng="<<rng<<endl;
    return x;
  };
public:
  virtual ~chain(){};
  chain():
    rng(new MotherOfAll(ProbabilityDist::getPRNG()->Next()))
    //rng(ProbabilityDist::getPRNG())//This should recover identical results to pre_omp version...?
    //rng(globalRNG)
  {
    id=idcount;idcount++;
    //cout<<"chain::chain():Set  id="<<id<<" one of "<<idcount<<" instance(s)."<<endl;//debug
    //cout<<"chain::chain():Set rng="<<rng.get()<<" one of "<<rng.use_count()<<" instance(s)."<<endl;//debug
    //Each chain has its own pseudo (newran) random number generator seeded with a seed drawn from the master PRNG
    //As long as the chains are created in a fixed order, this should allow threading independence of the result.
    Nfrozen=-1;
  };
  virtual shared_ptr<Random> getPRNG(){return rng;};
  virtual string show(){
    ostringstream s;
    s<<"chain(id="<<id<<"size="<<Nsize<<")\n";
    return s.str();
  };
  virtual string status(){
    ostringstream s;
    s<<"chain(id="<<id<<"N="<<Nsize<<"):"<<this->getState().get_string();
    return s.str();
  };
  virtual void reserve(int nmore){};//If you know how long you are going to run, it can be more efficient to reserve up front, rather than resizing ever step
  virtual int capacity(){return -1;};//If you know how long you are going to run, it can be more efficient to reserve up front, rather than resizing ever step
  virtual int size(){if(Nfrozen==-1)return Nsize;else return Nfrozen;};
  virtual state getState(int elem=-1,bool raw_indexing=false){cout<<"chain::getState: Warning. You've called a dummy function."<<endl;state s;return s;};//base class is not useful
  virtual double getLogPost(int elem=-1,bool raw_indexing=false){cout<<"chain::getLogPost: Warning. You've called a dummy function."<<endl;return 0;};//get log posterior from a chain sample
  virtual double getLogLike(int elem=-1,bool raw_indexing=false){cout<<"chain::getLogLike: Warning. You've called a dummy function."<<endl;return 0;};//get log prior from a chain sample
  //virtual double getLogPrior(int elem=-1,bool raw_indexing=false){cout<<"chain::getLogPrior: Warning. You've called a dummy function."<<endl;return 0;};//get log prior from a chain sample
  virtual int getDim(){return dim;};
  ///Test the expectation value for some test function.
  ///This could be, for example:
  ///1. An expected parameter value (test_func should return the parameter value)
  ///2. Probability of being in some bin region of param space (test_func returns 1 iff in that region)
  virtual double  Pcond(bool (*test_func)(state s)){return dim;};
  virtual int i_after_burn(int nburn=0){return nburn;}
  virtual void inNsigma(int Nsigma,vector<int> & indicies,int nburn=0){
    //cout<<" inNsigma:this="<<this->show()<<endl;
    int ncount=size()-this->i_after_burn(nburn);
    //cout<<"size="<<size()<<"   nburn="<<nburn<<"   i_after_burn="<<this->i_after_burn(nburn)<<endl;
    int inNstd_count=(int)(erf(Nsigma/sqrt(2.))*ncount);
    vector<pair<double,double> > lpmap;
    //double zero[2]={0,0};
    //lpmap.resize(ncount,zero);
    //cout<<"ncount="<<ncount<<"   inNstd_count="<<inNstd_count<<endl;
    lpmap.resize(ncount);
    for(uint i=0;i<ncount;i++){
      int idx=i+this->i_after_burn(nburn);
      //cout<<idx<<" state:"<<this->getState(idx,true).get_string()<<endl;
      lpmap[i]=make_pair(-this->getLogPost(idx,true),idx);
    }
    //sort(lpmap.begin(),lpmap.end(),chain::AgtB);
    sort(lpmap.begin(),lpmap.end());
    indicies.resize(inNstd_count);
    for(uint i=0;i<inNstd_count;i++){
      indicies[i]=lpmap[i].second;
      //cout<<"  :"<<lpmap[i].first<<" "<<lpmap[i].second<<endl;
    }
    return;
  };
  virtual void set_proposal(proposal_distribution &proposal){
    cout<<"chain::step: No base-class set_proposal operation defined!"<<endl;
    exit(1);
  };
  virtual void step(){
    cout<<"chain::step: No base-class step() operation defined!"<<endl;
    exit(1);
  };
  virtual void dumpChain(ostream &os,int Nburn=0,int ievery=1){
    cout<<"chain::step: No base-class dumpChain() operation defined!"<<endl;
    exit(1);
  };
  //Interface for optional features of derived clases.
  ///This function should be overloaded to indicate the number of subchains in a hierarchical chain;
  virtual int multiplicity(){return 1;};
  ///subchains should be indexed from 0 to multiplicity-1
  virtual chain* subchain(int index){return this;};
  ///return the inverse temperature for this chain
  virtual double invTemp(){return 1.0;};
  ///Temporarily freeze the length of the chain as reported by size()
  void history_freeze(){Nfrozen=Nsize;};
  void history_thaw(){Nfrozen=-1;};
  int get_id(){return id;};
};


//** useful CHAIN Classes *******

class parallel_tempering_chains;
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
  double evolve_temp_rate,evolve_temp_lpost_cut;
  int maxswapsperstep;
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
  void dumpChain(ostream &os,int Nburn=0,int ievery=1){ dumpChain(0,os,Nburn,ievery);}
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
  bool evolve_temps(double rate=0.01,double lpost_cut=-1){
    do_evolve_temps=1;
    evolve_temp_rate=rate;
    evolve_temp_lpost_cut=lpost_cut;
    return true;
  };
  void do_reboot(double rate,double threshhold,double thermal,int every,int grace=0,bool graduate=false,double aggression=0){max_reboot_rate=rate;reboot_thresh=threshhold;reboot_thermal_thresh=thermal;test_reboot_every=every;reboot_grace=grace;reboot_aggression=aggression;reboot_graduate=graduate;
    cout<<"Will reboot every "<<" aggression="<<reboot_aggression<<endl;
  };
}; 
#endif    
