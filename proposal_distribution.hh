///States for MCMC
///
///state objects are tied to a stateSpace object which specifies to domain.
///boundary objects allow specification of the domain bounraies in each dimension.
///Written by John G Baker - NASA-GSFC (2013-2014)

#ifndef PTMCMC_PROPOSAL_HH
#define PTMCMC_PROPOSAL_HH
#include "states.hh"
#include "chain.hh"
#include "restart.hh"
#include "probability_function.hh"
#include <Eigen/Eigen>

using namespace std;

//************** PROPOSAL DISTRIBUTION classes ******************************


///Base class for defining proposal distributions.
///
///Useful proposal distributions often reference a chain.  Support for chain-referencing is provided here.
///Added more recently is support for user-defined proposals.  These must, at least reference a user_parent_object
///so support for that is also now included here.  Furthermore, if the user-proposal needs access to any
///instance-specific history, then support for managing that is provided as well.
///
///Copies and clones:
///
///When a new copy of the proposal is needed, as for new/parallel chains, the copy should generally be constructed by the
///clone function.  This will also ensure that (for user-defined functionality) the user parent object can be made aware
///of the new instance.  Note that we do not generally expect that the caller of the clone function can have access to the
///user parent object, and we do not constrain the nature of the parent object, so it cannot be owned by the proposal_distribution
///or its cloning object.  This means that we cannot arrange for the parent_object to be copied during a cloning operation.
///Instead the parent object is fixed and the parent_object must provide, by its own means, a new object for instance specific
///data each time a clone is constructed.  The new_user_instance_object_function is provided to allow the parent object to
///provide that.  If no new_user_instance_object_function is provided, then the user_instance_object will remain NULL and the
///user proposal will not be able to access any instance specific information.
class proposal_distribution: public restartable{
  static int idcount;
protected:
  int id;
  double log_hastings;
  int last_type;
  int accept_count;
  int reject_count;
  chain *ch;
  void *user_parent_object;
  void *user_instance_object;
  void * (*new_user_instance_object_function)(void *object,int id);
  void set_instance(){
    if(user_parent_object and new_user_instance_object_function){
      //ostringstream ss;ss<<"proposal_distribution::set_instance: this="<<this<<" Calling new_user_instance_object_function="<<(void*)new_user_instance_object_function<<" user_parent_object="<<user_parent_object;cout<<ss.str()<<endl;
      user_instance_object=new_user_instance_object_function(user_parent_object,id);
    }else user_instance_object=nullptr;
    //ostringstream ss;ss<<"proposal_distribution::set_instance: this="<<this<<" user_parent_object="<<user_parent_object<<" new_user_instance_object_function="<<(void*)new_user_instance_object_function<<" user_instance_object="<<user_instance_object;cout<<ss.str()<<endl;
  }
public:
  virtual ~proposal_distribution(){};
  proposal_distribution(void *user_parent_object=nullptr,void * (*new_user_instance_object_function)(void *object,int id)=nullptr):
    user_parent_object(user_parent_object),new_user_instance_object_function(new_user_instance_object_function){
    id=idcount;idcount++;log_hastings=0;last_type=0;accept_count=0;reject_count=0;ch=nullptr;
    //ostringstream ss;ss<<"proposal_distribution::constructor: this="<<this<<" user_parent_object="<<user_parent_object<<" new_user_instance_object_function="<<(void*)new_user_instance_object_function;cout<<ss.str()<<endl;
    set_instance();
  };
  virtual double log_hastings_ratio(){return log_hastings;};//(proposal part of Hasting ratio for most recent draw;
  virtual void set_chain(chain *c){ch=c;};//Not always needed.
  //virtual state draw(state &s,Random &rng){return s;};//The base class won't be useful.
  virtual state draw(state &s,chain *caller){return s;};//The base class won't be useful.
  ///Some proposals require some set-up, such as a chain of sufficient length.
  virtual bool is_ready() {return true;};
  virtual proposal_distribution* clone()const{return new proposal_distribution(*this);};
  virtual string show(){return "UnspecifiedProposal()";};
  ///Return type of draw (where that is relevant, otherwise 0)
  virtual int type(){return last_type;}
  virtual bool support_mixing(){return false;}
  virtual void accept(){accept_count++;};//external trigger to indicate that proposal was accepted (available for adaptive proposals)
  virtual void reject(){reject_count++;};//external trigger to indicate that proposal was rejected (available for adaptive proposals)
  virtual void accept( int count){accept_count=count;};//external trigger to indicate that proposal was accepted (available for adaptive proposals)
  virtual void reject(int count){reject_count=count;};//external trigger to indicate that proposal was rejected (available for adaptive proposals)
  virtual void checkpoint(string path)override{};
  virtual void restart(string path)override{};
  virtual string report(int style=0){;//For status reporting on acceptance rate(0) or adaptive shares(1)
    ostringstream ss;
    if(style==0)ss<<accept_count*1.0/(accept_count+reject_count)<<"("<<accept_count<<")";
    return ss.str();
  };
  virtual string test(const vector<state> &samples,Random &rng){return "";};//Perform an intrinsic test and return the output
};

//Apply an stateSpaceInvolution map as a proposal:
class involution_proposal: public proposal_distribution{
  stateSpaceInvolution involution;
public:
  involution_proposal(stateSpaceInvolution &involution):involution(involution){};
  state draw(state &s,chain *caller){
    involution.set_random(*(caller->getPRNG()));
    log_hastings=log(involution.jacobian(s));
    return involution.transformState(s);
  };
  involution_proposal* clone()const{return new involution_proposal(*this);};
  string show(){return "Involution["+involution.get_label()+"]()";};
  string test(const vector<state> &samples,Random &rng){
    ostringstream outs;
    outs<<"Testing involution."<<endl;
    double testsum=0;
    for(auto s:samples){
      involution.set_random(rng);
      testsum+=involution.test_involution(s,1000000);
    }
    outs<<"RMS: "<<sqrt(testsum/samples.size())<<endl;
    return outs.str();
  };
  stateSpaceInvolution get_involution(){return involution;};
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
  gaussian_prop(const valarray<double> &sigmas,double oneDfrac=0.0, bool scaleWithTemp=false):sigmas(sigmas),oneDfrac(oneDfrac),scaleWithTemp(scaleWithTemp){
    identity_trans=true;
    valarray<double> zeros(0.0,sigmas.size());
    dist = new gaussian_dist_product(nullptr,zeros, sigmas);
    if(oneDfrac<0||oneDfrac>1){
      cout<<"gaussian_prop(constructor I): We require 0<=oneDfrac<=1. "<<endl;
      exit(1);
    }
  };
  gaussian_prop(vector<double> &sigmas,double oneDfrac=0.0, bool scaleWithTemp=false):
    gaussian_prop(valarray<double>(sigmas.data(),sigmas.size()),oneDfrac,scaleWithTemp){};

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
  gaussian_prop* clone()const{
    gaussian_prop* clone =new gaussian_prop(*this);
    valarray<double> zeros(0.0,sigmas.size());
    clone->dist=new gaussian_dist_product(nullptr,zeros, sigmas);
    return clone;
  }; 
  string show(){
    ostringstream ss; ss<<"StepBy"<<(identity_trans?"":"Covar")<<"["<<(dist?dist->show():"<null>")<<"](1Dfrac="<<oneDfrac<<",scaleWithTemp="<<scaleWithTemp<<")";return ss.str();};
};

///This class implements a user-defined multidimensional gaussian step proposal distribution.
///
///This generalizes the general-covariance variant of the multidimensional Gaussian (gaussian_prop)
///
///but we allow that and additional parameter vector transformation is performed.
///The matrix M for this transformation is computed to diagonalize a covariance matrix
///passed in initially.
class user_gaussian_prop: public proposal_distribution{
  Eigen::MatrixXd diagTransform;
  valarray<double>sigmas;
  gaussian_dist_product *dist;
  int ndim;
  string label;
  bool (*user_check_update)(const void *parent_object, void* instance_object, const state &s, const vector<double> &randoms, vector<double> &covarvec);
  bool check_update_registered;
  vector<int>idx_map;
  int nrand;
  bool first_draw;
protected:
  stateSpace domainSpace; //Note the Transform can be applied as long as this can be identified as a subspace.
public:
  user_gaussian_prop(const stateSpace &sp,const vector<double> &covarvec=vector<double>(), int nrand=0, const string label="",void *user_parent_object=nullptr,void * (*new_user_instance_object_function)(void*object,int id)=nullptr);
  user_gaussian_prop(void *user_parent_object, bool (*function)(const void *parent_object, void* instance_object, const state &, const vector<double> &randoms, vector<double> &covarvec),const stateSpace &sp,const vector<double> &covarvec=vector<double>(), int nrand=0, const string label="",void * (*new_user_instance_object_function)(void*object,int id)=nullptr);
  virtual ~user_gaussian_prop(){delete dist;};
  user_gaussian_prop* clone()const;
  string get_label()const {return label;};
  //void register_reference_object(void *object){
  //  ostringstream ss;ss<<"this="<<this<<" registering reference_object="<<object<<", thread="<<omp_get_thread_num()<<endl;cout<<ss.str()<<endl;
  //   user_object=object;};    
  void register_check_update(bool (*function)(const void *parent_object, void* instance_object, const state &, const vector<double> &randoms, vector<double> &covarvec)){
    user_check_update=function;
    check_update_registered=true;
  };
  state draw(state &s,chain *caller){
    //print some debugging info
    if(first_draw){
      //ostringstream ss;ss<<"this="<<this<<" drawing for chainID="<<caller->get_id()<<" on thread="<<omp_get_thread_num()<<endl;cout<<ss.str()<<endl;
      first_draw=false;
    }
    //first restrict to the relevant subspace for this step distribution

    if(idx_map.size()==0){
      idx_map=s.projection_indices_by_name(&domainSpace);
      for(int i=0;i<s.size();i++)
	if(i<0)cout<<"use_gaussian_prop['"+label+"']:draw: Param '"+domainSpace.get_name(i)+"' not found in stateSpace! Will ignore."<<endl;
    }
    vector<double> sparams(domainSpace.size());
    for(int i=0;i<ndim;i++)if(idx_map[i]>=0)sparams[i]=s.get_param(idx_map[i]);
    state ss=state(&domainSpace,sparams);
    last_type=check_update(ss,caller);
    state offset=dist->drawSample(*(caller->getPRNG()));;
    double x=1;
    valarray<double> data;
    offset.get_params_array(data);
    Eigen::Map<Eigen::VectorXd> vec(&data[0],offset.size());
    vec=diagTransform*vec;
    state newstate=s.scalar_mult(0);
    for(int i=0;i<ndim;i++)if(idx_map[i]>=0)newstate.set_param(idx_map[i],vec(i));
    newstate=s.add(newstate);
    return newstate;
  };
  string show(){
    ostringstream ss; ss<<"StepBy"<<"UserCovar["+label+"]";return ss.str();};
protected:
  void reset_dist(const vector<double> &covarvec){
    Eigen::MatrixXd cov(ndim,ndim);
    if(covarvec.size()==ndim){ //covarvec is understood as diag of diagonal matrix
      for(int i=0;i<ndim;i++)cov(i,i)=covarvec[i];
    } else if(covarvec.size()==(ndim*(ndim+1))/2){ //covarvec is concatenated UL side of covariance.  E.g. 3D identify is {1,0,0,1,0,1}
      //Convert vector-form covariance to Eigen symmetric matrix
      int ic=0;
      for(int i=0;i<ndim;i++){
	for(int j=i;j<ndim;j++){
	  cov(i,j)=covarvec[ic];
	  cov(j,i)=cov(i,j);
	  ic++;
	}
      }
    } else {
      cout<<"gaussian_prop:reset_dist Covar vector has unexpeced size,";
      cout<<"ndim="<<ndim<<", UL size="<<(ndim*(ndim+1))/2<<" but got "<<covarvec.size()<<" "<<endl;
      if(dist){
	cout<<" Skipping update!"<<endl;
	return;
      } else {
	cout<<" Setting to identity matrix!"<<endl;
	for(int i=0;i<ndim;i++)cov(i,i)=1;
      }
    }
    //Now perform the update
    sigmas.resize(ndim);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(cov);
    diagTransform=eigenSolver.eigenvectors();
    Eigen::VectorXd evalues = eigenSolver.eigenvalues();
    for(int i=0;i<ndim;i++){
      if(evalues[i]<0){
	cout<<"user_gaussian_prop["+label+"]: Warning. Negative covariance eigenvalue set to zero."<<endl;
	evalues[i]=0;
      }
      sigmas[i]=sqrt(evalues(i));
    }
    valarray<double> zeros(0.0,ndim);
    delete dist;
    dist = new gaussian_dist_product(nullptr,zeros, sigmas);
    //cout<<"this="<<this<<" created dist="<<dist<<endl;
  };    
  ///This function updates the proposal if user has provided an update callback function
  bool check_update(const state &s, chain *caller){
    if(!check_update_registered)return false;
    //Generate random values
    vector<double>randoms(nrand);
    for(auto & x : randoms)x=(caller->getPRNG()->Next());
    //Call user function
    vector<double> covarvec;
    bool renewing=user_check_update(user_parent_object, user_instance_object, s, randoms, covarvec);
    if(!renewing)return false;
    reset_dist(covarvec);
    return true;
  };

};


//Typically you want to draw from a set of various proposal distributions.
class proposal_distribution_set: public proposal_distribution{
  int Nsize;
  vector<proposal_distribution*>proposals;
  vector<double>shares;
  vector<double>bin_max;
  //Needed for distribution evolution
  double adapt_rate;
  vector<bool>last_accepted;
  int adapt_count;
  int adapt_every;
  int last_dist;
  bool own_pointers;
  //Needed for thermal scalings
  double Tpow;
  vector<double>hot_shares;
  void reset_bins();

public:
  virtual ~proposal_distribution_set(){
    //cout<<"Deleting: "<<show()<<endl;
    if(own_pointers){
      for(auto prop:proposals){
	//cout<<" -Deleting: "<<prop->show()<<endl;
	if(prop)delete prop;
      }}};//delete proposals
  virtual proposal_distribution_set* clone()const;
  proposal_distribution_set(const vector<proposal_distribution*> &props,const vector<double> &shares,double adapt_rate=0,double Tpow=0,vector<double> hot_shares=vector<double>(),bool take_pointers=true);
  ///For proposals which draw from a chain, we need to know which chain
  void set_chain(chain *c){ch=c;for(int i=0;i<Nsize;i++)proposals[i]->set_chain(c);};
  ///Randomly select from proposals i in 0..n and draw.
  ///Sets type() value at i+10*proposals[i].type() 
  //state draw(state &s,Random &rng);
  state draw(state &s,chain *caller);
  void accept();
  
  void reject();
  virtual void checkpoint(string path)override;
  virtual void restart(string path)override;
  string show();
  string report(int style=0);//For status reporting on acceptance rate(0) or adaptive shares(1)
  vector<proposal_distribution*>  members()const{return proposals;}; 
};

///Convenience constructor for a set of involution proposals based on the state
proposal_distribution_set involution_proposal_set(const stateSpace &space,double adapt_rate=0);

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
  ///recommended to sometimes try unit scaling, which promotes mode-hopping.
  double gamma_one_frac;
  double reduce_gamma_fac;
  ///size of small random jumps (terBraak)
  double b_small; 
  ///We ignore the first Mcount*ignore_frac entries in the history;
  double ignore_frac;
  ///Discount low-probability history
  double unlikely_alpha;
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
  differential_evolution(double snooker=0.0, double gamma_one_frac=0.1,double b_small=0.0001,double ignore_frac=0.3,double unlikely_alpha=0);
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
