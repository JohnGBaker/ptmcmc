///States for MCMC
///
///state objects are tied to a stateSpace object which specifies to domain.
///boundary objects allow specification of the domain bounraies in each dimension.
///Written by John G Baker - NASA-GSFC (2013-2014)

#ifndef PTMCMC_PROPOSAL_HH
#define PTMCMC_PROPOSAL_HH
#include "states.hh"
#include "chain.hh"
#include "probability_function.hh"
#include <Eigen/Eigen>

using namespace std;

//************** PROPOSAL DISTRIBUTION classes ******************************


///A class for defining proposal distributions.
///Useful proposal distributions often reference a chain.
class proposal_distribution {
protected:
  double log_hastings;
  int last_type;
public:
  virtual ~proposal_distribution(){};
  proposal_distribution(){log_hastings=0;last_type=0;};
  virtual double log_hastings_ratio(){return log_hastings;};//(proposal part of Hasting ratio for most recent draw;
  virtual void set_chain(chain *c){};//Not always needed.
  //virtual state draw(state &s,Random &rng){return s;};//The base class won't be useful.
  virtual state draw(state &s,chain *caller){return s;};//The base class won't be useful.
  ///Some proposals require some set-up, such as a chain of sufficient length.
  virtual bool is_ready() {return true;};
  virtual proposal_distribution* clone()const{return new proposal_distribution(*this);};
  virtual string show(){return "UnspecifiedProposal()";};
  ///Return type of draw (where that is relevant, otherwise 0)
  virtual int type(){return last_type;}
  virtual bool support_mixing(){return false;}
  virtual void accept(){};//external trigger to indicate that proposal was accepted (available for adaptive proposals)
  virtual void reject(){};//external trigger to indicate that proposal was rejected (available for adaptive proposals)
  virtual string report(){return "DefaultProp";};//For status reporting from adaptive proposals
};


//Draw from a distribution

//A trivial wrapper
class draw_from_dist: public proposal_distribution{
  const sampleable_probability_function &dist;
public:
  draw_from_dist(const sampleable_probability_function &dist):dist(dist){};
  //state draw(state &s,Random &rng){return dist.drawSample(rng);};
  state draw(state &s,chain *caller){
    state newstate=dist.drawSample(*(caller->getPRNG()));
    //If we are more likely to draw the newstate than the oldstate this goes into the Hastings ratio.
    log_hastings=dist.evaluate_log(s)-dist.evaluate_log(newstate);
    return newstate;
  };
  draw_from_dist* clone()const{return new draw_from_dist(*this);};
  string show(){return "DrawFrom["+dist.show()+"]()";};
};

///A multidimensional gaussian step proposal distribution.
///The constructor argument sigmas specifies the standard deviation width in each dimension.
///The optional constructor argument oneDfrac indicates a fraction of times that the Gaussian
///draw should be restricted to one randomly and uniformly selected dimension.
///
///Added feature: Nontrivial covariance.
///This is built atop the diagonal multidimensional Gaussian
///but we allow that and additional parameter vector transformation is performed.
///The matrix M for this transformation is computed to diagonalize a covariance matrix
///passed in initially.  The old behavior assumed the covariance was already diagonal
///so that M is the identity matrix.
class gaussian_prop: public proposal_distribution{
  bool identity_trans;
  Eigen::MatrixXd diagTransform;
  valarray<double>sigmas;
  gaussian_dist_product *dist;
  double oneDfrac;
  bool scaleWithTemp;
public:
  gaussian_prop(valarray<double> sigmas,double oneDfrac=0.0, bool scaleWithTemp=false):sigmas(sigmas),oneDfrac(oneDfrac),scaleWithTemp(scaleWithTemp){
    identity_trans=true;
    valarray<double> zeros(0.0,sigmas.size());
    dist = new gaussian_dist_product(nullptr,zeros, sigmas);
    if(oneDfrac<0||oneDfrac>1){
      cout<<"gaussian_prop(constructor I): We require 0<=oneDfrac<=1. "<<endl;
      exit(1);
    }
  };
  gaussian_prop(Eigen::MatrixXd &covar,double oneDfrac=0.0, bool scaleWithTemp=false):sigmas(sigmas),oneDfrac(oneDfrac),scaleWithTemp(scaleWithTemp){
    identity_trans=false;
    if(covar.rows() != covar.cols()){
      cout<<"gaussian_prop(constructor II): covar must be a square matrix!"<<endl;
      exit(-1);
    }
    int ndim=covar.rows();
    sigmas.resize(ndim);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
    diagTransform=eigenSolver.eigenvectors();
    Eigen::VectorXd Evalues = eigenSolver.eigenvalues();
    for(int i=0;i<ndim;i++)sigmas[i]=sqrt(Evalues(i));
    valarray<double> zeros(0.0,ndim);
    dist = new gaussian_dist_product(nullptr,zeros, sigmas);
    if(oneDfrac<0||oneDfrac>1){
      cout<<"gaussian_prop(constructor II): We require 0<=oneDfrac<=1. "<<endl;
      exit(1);
    }
    cout<<" constructing gaussian_prop with cov matrix=\n"<<covar<<endl;
    cout<<" Eigenvalues="<<Evalues<<endl;
    cout<<" transform=\n"<<diagTransform<<endl;
    cout<<" test=\n"<<diagTransform*Evalues.asDiagonal()*diagTransform.inverse()<<endl;
  };
  virtual ~gaussian_prop(){delete dist;};
  //state draw(state &s,Random &rng){
    //Umm is this Markovian. Do we need to set log_hastings for...
    //state offset=dist->drawSample(rng);
    //return s.add(offset);
  //};
  state draw(state &s,chain *caller){
    state offset=dist->drawSample(*(caller->getPRNG()));;
    double x=1;
    if(oneDfrac>0)x=caller->getPRNG()->Next();
    if(oneDfrac>0&&x<oneDfrac){ //step along one principal axis direction
      int ndim=offset.size();
      int i=ndim*caller->getPRNG()->Next();
      valarray<double>vals(0.0,ndim);
      vals[i]=offset.get_param(i);
      offset=state(offset.getSpace(),vals);
      //cout<<"gaussian_prop:drew: offset="<<offset.get_string()<<endl;
      last_type=1;
    } else last_type=0;
    if(not identity_trans){//if non-trivial, we need to transform to state basis.
      //cout<<"state before transform:"<<offset.get_string()<<endl;
      valarray<double> data;
      offset.get_params_array(data);
      Eigen::Map<Eigen::VectorXd> vec(&data[0],offset.size());
      vec=diagTransform*vec;
      for(int i=0;i<offset.size();i++)offset.set_param(i,vec(i));
      //cout<<"state after transform:"<<offset.get_string()<<endl;
    }
    if(scaleWithTemp)s.scalar_mult(1.0/sqrt(caller->invTemp()));
    return s.add(offset);
  };
  gaussian_prop* clone()const{return new gaussian_prop(*this);};
  string show(){ostringstream ss; ss<<"StepBy"<<(identity_trans?"":"Covar")<<"["<<dist->show()<<"](1Dfrac="<<oneDfrac<<",scaleWithTemp="<<scaleWithTemp<<")";return ss.str();};
};


//Typically you want to draw from a set of various proposal distributions.
class proposal_distribution_set: public proposal_distribution{
  int Nsize;
  vector<proposal_distribution*>proposals;
  vector<double>shares;
  vector<double>bin_max;
  //Needed for distribution evolution
  double adapt_rate;
  double target_acceptance_rate;
  vector<bool>last_accepted;
  int adapt_count;
  int adapt_every;
  int last_dist;
public:
  virtual ~proposal_distribution_set(){for(auto prop:proposals)if(prop)delete prop;};//delete proposals
  virtual proposal_distribution_set* clone()const;
  proposal_distribution_set(vector<proposal_distribution*> &props,vector<double> &shares,double adapt_rate=0,double target_acceptance_rate=0.2);
  ///For proposals which draw from a chain, we need to know which chain
  void set_chain(chain *c){for(int i=0;i<Nsize;i++)proposals[i]->set_chain(c);};
  ///Randomly select from proposals i in 0..n and draw.
  ///Sets type() value at i+10*proposals[i].type() 
  //state draw(state &s,Random &rng);
  state draw(state &s,chain *caller);
  void accept();
  void reject();
  string show();
  string report();//For status reporting on adaptive
};  
    
///DifferentialEvolution
///Based mainly on (ter Braak and Vrugt 08, Stat Comput (2008) 18: 435–446)
///Also SampsonEA2011 ArXiV:1105.2088
///The basic algorithm involves choosing a random state from the past history and proposing a jump in that
/// direction. terBraakEA also add a "small" random jump 
///Comments: SampsonEA comment that "It can be shown that this approach is asymptotically Markovian in the limit as one uses the full past history of the chain."  terBraak don't show this, but state "we conjecture that DE- MCZ is ergodic and converges to a stationary distribution with [the right] pdf."  The basis for this is by general reference to the Adaptive Metropolis-Hastings paper (HaarioEA 2001).  
///The basic implementation works with just one chain and draws directly from its past.  This is a little different from the terBraak model which draws exclusively from other chains.
///We well also implement a more sophisticated option which allows working with chain-sets, including parallel tempering chain sets.  Here we will be able to
///provide the option of 1. excluding the self-chain from the 
class differential_evolution: public proposal_distribution {
  bool have_chain;
  int dim;
  //The chain will manage the history.
  //int Mcount;
  //int save_every;//to save memory it is not necessary to save every element of the history, nearby samples are not independent anyway.
  //int Scount;
  //vector<state> history;
  chain *ch;
  ///recommended to sometimes try unit scaling, which promotes mode-hopping.
  double gamma_one_frac;
  double reduce_gamma_fac;
  ///size of small random jumps (terBraak)
  double b_small; 
  ///We ignore the first Mcount*ignore_frac entries in the history;
  double ignore_frac;
  //Comment:Citing that work terBraak also comments that (retaining)  the "full [sampled] past is required to guarantee ergodicity of the chain."  In HaaioEA, however, it seems clear that retaining a fixed recent fraction of the past (they give 1/2 as an example) would also be sufficient for their theorem.  I haven't carefully studied it, but this also seems to be supported by Roberts+Rosenthal2007, perhaps satisfying their "diminishing adaptation" condition.
  double snooker;
  //SampsonEA don't use it, but terBraakEA seem to prefer a variant on DE where the scaling is set by a separate drawing of states;  I guess this propotes variety.
  bool do_support_mixing;
  double temperature_mixing_factor;
  /*
  static int icount;
  static vector< vector< int > > mixcounts;
  static vector< int > trycounts;
  */
  
  //SOME INTERNAL FUNCTIONS
  //state draw_standard(state &s, Random &rng);
  //state draw_snooker(state &s, Random & rng);
  //state  draw_from_chain(Random &rng);
  state draw_standard(state &s, chain *caller);
  state draw_snooker(state &s, chain *caller);
  int  draw_i_from_chain(chain *caller,chain *c);
  state  draw_from_chain(chain *caller);
  int get_min_start_size(){return dim*10;};//Don't run below this chain length
  int get_min_cut_size(){return dim*100;};  //Don't ignore early part below this length
public:
  void reduce_gamma(double factor){reduce_gamma_fac=factor;};
  void mix_temperatures_more(double factor){temperature_mixing_factor=factor;};
  differential_evolution(double snooker=0.0, double gamma_one_frac=0.1,double b_small=0.0001,double ignore_frac=0.3);
  void set_chain(chain *c){ch=c;have_chain=true;dim=ch->getDim();};
  ///Verify that the chain is set and that the chain has some minimum length of 10*dim.
  bool is_ready(){return(have_chain&&ch->size()>=get_min_start_size());};
  //state draw(state &s,Random &rng);
  state draw(state &s,chain *caller);
  virtual differential_evolution* clone()const{return new differential_evolution(*this);};
  string show();
  bool support_mixing(bool do_it){do_support_mixing=do_it;return do_it;};
  bool support_mixing(){return do_support_mixing;};
}; 
#endif


/*
Idea:
  Intra-specific differential evolution.  It is non-adaptive to have sex with a frog.  Similarly, if the posterior surface includes well-separated peaks of wildly differing sizes and shapes, then the combined information from all of them might not be very good for any of them.
  An option is to use unsupervised learning techniques to select interesting sub-regions of the parameter space which may be more coherently related to each other, then to (some of the time) run DE independently in each.
  A relatively simple implementation would be to use something like K-means clustering to cluster the parameter-space points, and to run DE separately on the cluster points.  (It might be tricky to figure out how to do the detailed balance if the computation suggests a jump out of the cluster region.)  There could be another algorithm for jumping between clusters (say prop to evidence?).
  It might be that we want not to do exactly K-means clustering, but something similar, with posterior weighting in calculating the centers.  Then we might test for the coherence of the clusters, possible splitting or merging them.  Useful statistics might be the cluster variance matrix and the cluster parameter expectation value.  By comparing the posterior at the expectation value with the maximum, we might be able to decide when clusters should be divided or merged.
  The clustering could be managed by arrays of vectors of references to the chain points, with occasional updates of the clustering centers, or just occasional updates to the whole population set overall, keeping it fixed erstwhile... and not referencing the more recent chain history. 
*/
