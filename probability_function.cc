///Probability funtions for MCMC.
///
///state objects are tied to a stateSpace object which specifies to domain.
///boundary objects allow specification of the domain bounraies in each dimension.
///Written by John G Baker - NASA-GSFC (2013-2014)
#include "probability_function.hh"
#include "chain.hh"

gaussian_dist_product::gaussian_dist_product(stateSpace *space,unsigned int N):sampleable_probability_function(space){
  dim=N;
  x0s.resize(N,0);
  sigmas.resize(N,1);
  dists.resize(dim);
  for(size_t i=0;i<dim;i++)dists[i]=new GaussianDist(x0s[i],sigmas[i]);
};

gaussian_dist_product::gaussian_dist_product(stateSpace *space, valarray<double>&x0s,valarray<double>&sigmas):x0s(x0s),sigmas(sigmas),sampleable_probability_function(space){
  dim=x0s.size();
  if(dim!=sigmas.size()){
    cout<<"gaussian_dist_product(constructor): Array sizes mismatch.\n";
    exit(1);
  }
  dists.resize(dim);
  for(size_t i=0;i<dim;i++){
    dists[i]=new GaussianDist(x0s[i],sigmas[i]);
  }
};

gaussian_dist_product::~gaussian_dist_product(){
  while(dists.size()>0){
    delete dists.back();
    dists.pop_back();
  }
};

state gaussian_dist_product::drawSample(Random &rng){
  valarray<double> v(dim);    
  //cout<<"gaussian_dist_product::drawSample: drawing vector of len "<<dim<<"for space at"<<space<<endl;//debug
  for(uint i=0;i<dim;i++){
    double number=dists[i]->draw(&rng);
    v[i]=number;
    //cout<<"  "<<number;//debug
  }
  //cout<<endl;//debug
  return state(space,v);
};

double gaussian_dist_product::evaluate(state &s){
  if(s.invalid())return 0;
  if(dim!=s.size()){
    cout<<"gaussian_dist_product:evaluate: State size mismatch.\n";
    exit(1);
  }
  double result=1;
  vector<double> pars=s.get_params_vector();
  for(uint i=0;i<dim;i++)result*=dists[i]->pdf(pars[i]);
  return result;
};

///Given a unit hypercube point with uniform distribution
///return corresponding parameter state with this distribution
state gaussian_dist_product::invcdf(const state &s)const{
  vector<double> pars=s.get_params_vector();
  valarray<double> v(dim);    
  for(uint i=0;i<dim;i++){
    //If the "boundary" limits the domain of the  distribution, then we take that into account here
    //Could make this faster by precomputing the range...
    double min,max;
    space->get_bound(i).getDomainLimits(min,max);
    double cdfmin=dists[i]->cdf(min);//zero maps to here before calling generic invcdf
    double cdfmax=dists[i]->cdf(max);//one maps to here before calling generic invcdf
    double val=cdfmin+pars[i]*(cdfmax-cdfmin);
    v[i]=dists[i]->invcdf(val);
  }
  return state(space,v);
};
  
string gaussian_dist_product::show()const{
  ostringstream s;
  s<<"GaussianSampleableProb(\n";
  for( int i=0;i<dim;i++)
    s<<"  "<<(space?space->get_name(i):"[???]")<<" from "<<dists[i]->show()<<"\n";
  s<<")\n";
  return s.str();    
};

// An example class for defining likelihoods/priors/etc
// from which we can draw samples based on the ProbabilityDist class.
// Default version is flat within unit range on each parameter.
uniform_dist_product::uniform_dist_product(stateSpace *space , int N):sampleable_probability_function(space){
  dim=N;
  min.resize(N,0);
  max.resize(N,1);
  dists.resize(dim);
  for(uint i=0;i<dim;i++)dists[i]=new UniformIntervalDist(min[i],max[i]);
};

uniform_dist_product::uniform_dist_product(stateSpace *space,valarray<double>&min_corner,valarray<double>&max_corner):min(min_corner),max(max_corner),sampleable_probability_function(space){
  dim=min.size();
  if(dim!=max.size()){
    cout<<"prob_dist_product(constructor): Array sizes mismatch.\n";
    exit(1);
  }
  dists.resize(dim);
  for(uint i=0;i<dim;i++)dists[i]=new UniformIntervalDist(min[i],max[i]);
};

uniform_dist_product::~uniform_dist_product(){
  while(dists.size()>0){
    delete dists.back();
    dists.pop_back();
  }
};

state uniform_dist_product::drawSample(Random &rng){
  valarray<double> v(dim);
  for(uint i=0;i<dim;i++){
    double number=dists[i]->draw(&rng);
      v[i]=number;
  }
  return state(space,v);
};

double uniform_dist_product::evaluate(state &s){
  if(s.invalid())return 0;
  if(dim!=s.size()){
    cout<<"uniform_dist_product:evaluate: State size mismatch.\n";
      exit(1);
  }
  double result=1;
  vector<double> pars=s.get_params_vector();
  for(uint i=0;i<dim;i++)result*=dists[i]->pdf(pars[i]);
  return result;
};

///Given a unit hypercube point with uniform distribution
///return corresponding parameter state with this distribution
state uniform_dist_product::invcdf(const state &s)const{
  vector<double> pars=s.get_params_vector();
  valarray<double> v(dim);    
  for(uint i=0;i<dim;i++){
    //If the "boundary" limits the domain of the  distribution, then we take that into account here
    //Could make this faster by precomputing the range...
    double min,max;
    space->get_bound(i).getDomainLimits(min,max);
    double cdfmin=dists[i]->cdf(min);//zero maps to here before calling generic invcdf
    double cdfmax=dists[i]->cdf(max);//one maps to here before calling generic invcdf
    double val=cdfmin+pars[i]*(cdfmax-cdfmin);
    v[i]=dists[i]->invcdf(val);
  }
  return state(space,v);
};
  
string uniform_dist_product::show()const{
  ostringstream s;
  s<<"UniformSampleableProb(\n";
  for( int i=0;i<dim;i++)
    s<<"  "<<(space?space->get_name(i):"[???]")<<" from "<<dists[i]->show()<<"\n";
  s<<")\n";
  return s.str();    
};



// A class for defining likelihoods/priors/etc from an independent mix of gaussian and flat priors
// from which we can draw samples based on the ProbabilityDist class.
// unit normal range on each parameter.

mixed_dist_product::mixed_dist_product(stateSpace *space,unsigned int N):sampleable_probability_function(space){
  //cout<<"mixed_dist_product::mixed_dist_product("<<space<<","<<N<<"): Constructing this="<<this<<endl;//debug;
  //cout<<"mixed_dist_product:Creating with space="<<space<<endl;
  dim=N;
  centers.resize(N,0);
  types.resize(N,0);
  halfwidths.resize(N,1);
  dists.resize(dim);
  for(size_t i=0;i<dim;i++){
    dists[i]=new GaussianDist(centers[i],halfwidths[i]);
    types[i]=gaussian;
  }
};

mixed_dist_product::mixed_dist_product(stateSpace *space,valarray<int> &types,valarray<double>&centers,valarray<double>&halfwidths):types(types),centers(centers),halfwidths(halfwidths),sampleable_probability_function(space){
  //cout<<"mixed_dist_product::mixed_dist_product("<<space<<",types,centers,halfwidths): Constructing this="<<this<<endl;//debug;
  dim=centers.size();
  if(dim!=halfwidths.size()||dim!=types.size()){
    cout<<"mixed_dist_product(constructor): Array sizes mismatch.\n";
      exit(1);
  }
  dists.resize(dim);
  for(size_t i=0;i<dim;i++){
    if(types[i]==uniform)
	dists[i]=new UniformIntervalDist(centers[i]-halfwidths[i],centers[i]+halfwidths[i]);
    else if(types[i]==gaussian)
      dists[i]=new GaussianDist(centers[i],halfwidths[i]);
    else if(types[i]==polar)
      dists[i]=new UniformPolarDist();//note centers,halfwidths are ignored
    else {
      cout<<"mixed_dist_product(constructor): Unrecognized type. types["<<i<<"]="<<types[i]<<endl;
      exit(1);
      }
  }
};

mixed_dist_product::~mixed_dist_product(){
  while(dists.size()>0){
    delete dists.back();
    dists.pop_back();
  }
};

state mixed_dist_product::drawSample(Random &rng){
  valarray<double> v(dim);
  for(uint i=0;i<dim;i++){
    double number=dists[i]->draw(&rng);
    v[i]=number;
  }
  state s(space,v);
  //cout<<"mixed_dist_product::drawSample(): drew state:"<<s.show()<<endl;//debug
  return s; 
};

double mixed_dist_product::evaluate(state &s){
  if(s.invalid())return 0;
  if(dim!=s.size()){
    cout<<"mixed_dist_product:evaluate: State size mismatch.\n";
    exit(1);
  }
  double result=1;
  vector<double> pars=s.get_params_vector();
  for(uint i=0;i<dim;i++){
    double pdfi=dists[i]->pdf(pars[i]);
      //cout<<"subspace prior for "<<s.getSpace()->get_name(i)<<" -> "<<pdfi<<endl;
    result*=pdfi;
  }
  return result;
};

///Given a unit hypercube point with uniform distribution
///return corresponding parameter state with this distribution
state mixed_dist_product::invcdf(const state &s)const{
  vector<double> pars=s.get_params_vector();
  valarray<double> v(dim);    
  for(uint i=0;i<dim;i++){
    //If the "boundary" limits the domain of the  distribution, then we take that into account here
    //Could make this faster by precomputing the range...
    double min,max;
    space->get_bound(i).getDomainLimits(min,max);
    double cdfmin=dists[i]->cdf(min);//zero maps to here before calling generic invcdf
    double cdfmax=dists[i]->cdf(max);//one maps to here before calling generic invcdf
    double val=cdfmin+pars[i]*(cdfmax-cdfmin);
    v[i]=dists[i]->invcdf(val);
  }
  return state(space,v);
};
  
string mixed_dist_product::show()const{
  ostringstream s;
  s<<"SampleableProb(\n";
  for( int i=0;i<dim;i++)
    s<<"  "<<(space?space->get_name(i):"[???]")<<" from "<<dists[i]->show()<<"\n";
  s<<")\n";
  return s.str();    
};


//Draw samples from a chain
//This base version provides support only on the chain points themselves (no interpolation) interpolative variants (eg using approximate nearest neighbor approach) could be developed if needed.  With chain support this can be used as prior as long as the proposal distribution is strictly compatible (in particular only proposing points in the reference chain).

chain_distribution::chain_distribution(chain &c, int istart):c(c),istart(istart),sampleable_probability_function(c.getState().getSpace()){
    last_sample=-1;
  };


state chain_distribution::drawSample(Random &rng){
    double xrand=rng.Next();
    //rngout<<xrand<<" 2"<<endl;
    //cout<<" :"<<xrand<<" 2 rng="<<&rng<<endl;

    int i=(int)(xrand*(c.size()-istart))+istart; //pick a chain element
    //cout<<"chain_distribution:drew sample "<<i<<" = "<<c.getState(i).get_string()<<endl;
    //cout<<" &c="<<&c<<endl;
    last_sample=i;
    return c.getState(i);
};
double chain_distribution::evaluate_log(state &s){
    if(s.invalid())return -INFINITY;
    state last_state;
    double diff=0;
    //if chain_distribution is used with omp parallelism then we may need to make "last_sample" a thread-local variable or maybe test the thread which set last_sample
    //cout<<"chain_distribution: last_sample="<<last_sample<<", evaluating log for state "<<s.get_string()<<endl;
    if(last_sample>=0 and last_sample<c.size()){//verify state equivalence
      last_state=c.getState(last_sample);
      state sdiff=s.scalar_mult(-1).add(last_state);
      diff=sdiff.innerprod(sdiff);
    } 
    if(last_sample<0||last_sample>=c.size()||diff>0){
      cout<<"chain_distribution:evaluate_log: Cannot evaluate on arbitrary state point. Expected sample("<<last_sample<<")="<<last_state.get_string()<<"!="<<s.get_string()<<".\n";
      exit(1);
    }
    return c.getLogPost(last_sample);
};

