///Probability funtions for MCMC.
///
///state objects are tied to a stateSpace object which specifies to domain.
///boundary objects allow specification of the domain bounraies in each dimension.
///Written by John G Baker - NASA-GSFC (2013-2014)
#include "probability_function.hh"
#include "chain.hh"

gaussian_dist_product::gaussian_dist_product(const stateSpace *space,unsigned int N):sampleable_probability_function(space){
  dim=N;
  x0s.resize(N,0);
  sigmas.resize(N,1);
  dists.resize(dim);
  for(size_t i=0;i<dim;i++)dists[i]=new GaussianDist(x0s[i],sigmas[i]);
};

gaussian_dist_product::gaussian_dist_product(const stateSpace *space, const valarray<double>&x0s,const valarray<double>&sigmas, bool wrap_probability):x0s(x0s),sigmas(sigmas),sampleable_probability_function(space),wrap_probability(wrap_probability){
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

state gaussian_dist_product::drawSample(Random &rng)const{
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

double gaussian_dist_product::evaluate(state &s)const{
  if(s.invalid())return 0;
  if(dim!=s.size()){
    cout<<"gaussian_dist_product:evaluate: State size mismatch.\n";
    exit(1);
  }
  double result=1;
  vector<double> pars=s.get_params_vector();
  if(wrap_probability){
    for(uint i=0;i<dim;i++){
      boundary b=space->get_bound(i);
      double resulti=dists[i]->pdf(pars[i]);
      if(b.isWrapped()){
	const double tol=1e-12;
	double xplus=pars[i],xminus=pars[i],delta=1;
	double xmin,xmax,width;
	b.getDomainLimits(xmin,xmax);
	width=xmax-xmin;
	int count=0;
	while(delta>tol and count<100){
	  xplus+=width;
	  xminus-=width;
	  delta=dists[i]->pdf(xminus)+dists[i]->pdf(xplus);
	  resulti+=delta;
	  count++;
	}
      }
      result*=resulti;
    }
  }
  else for(uint i=0;i<dim;i++)result*=dists[i]->pdf(pars[i]);
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
  
string gaussian_dist_product::show(int ii)const{
  //cout<<"Entering GDP: ii="<<ii<<endl;
  ostringstream s;
  if(ii<0){
    s<<"SampleableDistProduct(\n";
    for( int i=0;i<dim;i++)
      s<<"  "<<(space?space->get_name(i):"[???]")<<" from "<<dists[i]->show()<<"\n";
    s<<")\n";
  } else {
    if(ii>dim){
      cout<<"gaussian_dist_product::show: Index out of range."<<endl;
      exit(1);
    }
    s<<dists[ii]->show();
  }
  return s.str();    
};

// An example class for defining likelihoods/priors/etc
// from which we can draw samples based on the ProbabilityDist class.
// Default version is flat within unit range on each parameter.
uniform_dist_product::uniform_dist_product(const stateSpace *space , int N):sampleable_probability_function(space){
  dim=N;
  min.resize(N,0);
  max.resize(N,1);
  dists.resize(dim);
  for(uint i=0;i<dim;i++)dists[i]=new UniformIntervalDist(min[i],max[i]);
};

uniform_dist_product::uniform_dist_product(const stateSpace *space,const valarray<double>&min_corner,const valarray<double>&max_corner):min(min_corner),max(max_corner),sampleable_probability_function(space){
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

state uniform_dist_product::drawSample(Random &rng)const{
  valarray<double> v(dim);
  for(uint i=0;i<dim;i++){
    double number=dists[i]->draw(&rng);
      v[i]=number;
  }
  return state(space,v);
};

double uniform_dist_product::evaluate(state &s)const{
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
  
string uniform_dist_product::show(int ii)const{
  //cout<<"Entering UDP: ii="<<ii<<endl;
  ostringstream s;
  if(ii<0){
    s<<"SampleableDistProduct(\n";
    for( int i=0;i<dim;i++)
      s<<"  "<<(space?space->get_name(i):"[???]")<<" from "<<dists[i]->show()<<"\n";
    s<<")\n";
  } else {
    if(ii>dim){
      cout<<"uniform_dist_product::show: Index out of range."<<endl;
      exit(1);
    }
    s<<dists[ii]->show();
  }
  return s.str();    
};



// A class for defining likelihoods/priors/etc from an independent mix of gaussian and flat priors
// from which we can draw samples based on the ProbabilityDist class.
// unit normal range on each parameter.

mixed_dist_product::mixed_dist_product(const stateSpace *space,unsigned int N):sampleable_probability_function(space){
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

mixed_dist_product::mixed_dist_product(const stateSpace *space,const valarray<int> &types,const valarray<double>&centers,const valarray<double>&halfwidths,bool verbose):types(types),centers(centers),halfwidths(halfwidths),verbose(verbose),sampleable_probability_function(space){
  //cout<<"mixed_dist_product::mixed_dist_product("<<space<<",types,centers,halfwidths): Constructing this="<<this<<endl;//debug;
  dim=centers.size();
  if(dim!=halfwidths.size()||dim!=types.size()||space->size()>dim){
    cout<<"mixed_dist_product(constructor): Array sizes mismatch.\n";
    cout<<"Ncenters="<<dim<<" Nscales="<<halfwidths.size()<<" Ntypes="<<types.size()<<" space_dim="<<space->size()<<endl;
    exit(1);
  }
  if(space->size()!=dim)dim=space->size();
  dists.resize(dim);
  for(size_t i=0;i<dim;i++){
    if(types[i]==uniform)
	dists[i]=new UniformIntervalDist(centers[i]-halfwidths[i],centers[i]+halfwidths[i]);
    else if(types[i]==gaussian)
      dists[i]=new GaussianDist(centers[i],halfwidths[i]);
    else if(types[i]==polar)//Uniform polar projection distribution, with polar angle measured 0 to pi from north pole
      dists[i]=new UniformPolarDist(centers[i]-halfwidths[i],centers[i]+halfwidths[i]);
    else if(types[i]==copolar)//Uniform polar projection distribution, with polar angle measured to +/- pi/2 from equator
      dists[i]=new UniformCoPolarDist(centers[i]-halfwidths[i],centers[i]+halfwidths[i]);
    else if(types[i]==log)
      dists[i]=new UniformLogDist(centers[i]/halfwidths[i],centers[i]*halfwidths[i]);//halfwidths is taken as multiplicative in this case
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

state mixed_dist_product::drawSample(Random &rng)const{
  valarray<double> v(dim);
  //cout<<"mixed_dist_product::drawSample():space="<<space->show()<<endl;
  for(uint i=0;i<dim;i++){
    double number=dists[i]->draw(&rng);
    v[i]=number;
    if(verbose){
      ostringstream ss;
      ss<<"mixed_dist_product::drawSample "<<space->get_name(i)<<"->"<<v[i]<<"\n";
      cout<<ss.str();
    }
  }
  state s(space,v);
  //cout<<"mixed_dist_product::drawSample(): drew state:"<<s.show()<<endl;//debug
  return s; 
};

double mixed_dist_product::evaluate(state &s)const{
  if(s.invalid())return 0;
  if(dim!=s.size()){
    cout<<"mixed_dist_product:evaluate: State size mismatch.\n";
    exit(1);
  }
  double result=1;
  vector<double> pars=s.get_params_vector();
  for(uint i=0;i<dim;i++){
    double pdfi=dists[i]->pdf(pars[i]);
    if(verbose){
      ostringstream ss;
      ss<<"subspace prior for "<<s.getSpace()->get_name(i)<<"="<<pars[i]<<" -> "<<pdfi<<"\n";
      cout<<ss.str();
    }
      result*=pdfi;
  }
  if(verbose){
    ostringstream ss;
    ss<<"prior result = "<<result<<"\n";
    cout<<ss.str();
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
  
string mixed_dist_product::show(int ii)const{
  //cout<<"Entering MDP: ii="<<ii<<endl;
  ostringstream s;
  if(ii<0){
    s<<"SampleableDistProduct(\n";
    for( int i=0;i<dim;i++)
      s<<"  "<<(space?space->get_name(i):"[???]")<<" from "<<dists[i]->show()<<"\n";
    s<<")\n";
  } else {
    if(ii>dim){
      cout<<"mixed_dist_product::show: Index out of range."<<endl;
      exit(1);
    }
    s<<dists[ii]->show();
  }
  //cout<<"exiting MDP with: "<<s.str()<<endl;
  return s.str();
};

///
///Generic class for defining sampleable probability distribution from a direct product of independent state spaces.
///
independent_dist_product::independent_dist_product(const stateSpace *product_space,  const sampleable_probability_function *subspace1_dist, const sampleable_probability_function *subspace2_dist)
  :independent_dist_product(product_space,vector<const sampleable_probability_function*>(initializer_list<const sampleable_probability_function*>{subspace1_dist,subspace2_dist})){};

independent_dist_product::independent_dist_product(const stateSpace *product_space,  const sampleable_probability_function *subspace1_dist, const sampleable_probability_function *subspace2_dist, const sampleable_probability_function *subspace3_dist)
  :independent_dist_product(product_space,vector<const sampleable_probability_function*>(initializer_list<const sampleable_probability_function*>{subspace1_dist,subspace2_dist,subspace3_dist})){};

independent_dist_product::independent_dist_product(const stateSpace *product_space,  const sampleable_probability_function *subspace1_dist, const sampleable_probability_function *subspace2_dist, const sampleable_probability_function *subspace3_dist, const sampleable_probability_function *subspace4_dist)
  :independent_dist_product(product_space,vector<const sampleable_probability_function*>(initializer_list<const sampleable_probability_function*>{subspace1_dist,subspace2_dist,subspace3_dist,subspace4_dist})){};

independent_dist_product::independent_dist_product(const stateSpace *product_space, const vector<const sampleable_probability_function*> &subspace_dists):sampleable_probability_function(product_space){
  dim=product_space->size();
  Nss=subspace_dists.size();
  ss_dists.resize(Nss);
  ss.resize(Nss);
  ss_indices.resize(Nss);
  index_ss.resize(dim);
  index_ss_index.resize(dim);
  int dim_count=0;
  int icount=0;
  for(size_t i=0;i<Nss;i++){
    if(subspace_dists[i]->getDim()>0){//skip adding any empty subspaces
      ss_indices[icount].resize(0);
      ss_dists[icount]=subspace_dists[i];
      ss[icount]=subspace_dists[i]->get_space();
      int ss_dim=ss_dists[icount]->getDim();
      dim_count+=ss_dim;
      icount++;
    }
  }
  Nss=icount;
  ss_dists.resize(Nss);//redo this incase some some empty subspaces were skipped
  ss.resize(Nss);
  ss_indices.resize(Nss);
  
  //Check that the spaces are commensurate and define parameter mapping
  if(dim!=dim_count){
    cout<<"independent_dist_product(constructor): Total dimension of subspaces does not match product space dimension:"<<endl;
    cout<<"product_space: "<<product_space->show()<<endl<<"subspaces:"<<endl;
    for(size_t i=0;i<Nss;i++)cout<<ss[i]->show()<<endl;    
    exit(1);
  }    
  for(int i=0;i<dim;i++){
    string name=space->get_name(i);
    bool found=false;
    for(int j=0;j<Nss;j++){
      int index=ss[j]->get_index(name);
      if(index>=0){
	if(!found){
	  found=true;
	  index_ss[i]=j;
	  index_ss_index[i]=index;
	  ss_indices[j].push_back(i);
	} else {
	  cout<<"independent_dist_product(constructor): Found name '"<<name<<"' in multiple spaces.  Those beyond the first will be ignored as needed."<<endl;
	  exit(1);
	  //ss_indices[j].push_back(i)
	}
      }
    }
    if(!found){
      cout<<"independent_dist_product(constructor): Did not find name '"<<name<<"' among subspace names."<<endl;
      exit(1);
    }
  }
};
  
//Take the direct product state of subspace samples
state independent_dist_product::drawSample(Random &rng)const{
  valarray<double> param_vals(dim);
  vector<state> substates(Nss);
  state s();
  //draw a sample from each subspace
  //cout<<"dim="<<dim<<" Nss="<<Nss<<endl;
  for(int i=0;i<Nss;i++){
    //cout<<"i="<<i<<":\ndist="<<ss_dists[i]->show()<<endl;
    substates[i]=ss_dists[i]->drawSample(rng);
    //cout<<" [drew "<<substates[i].size()<<" parameter values.]"<<endl;
  }
  //then map the parameters to the product space
  for(int i=0;i<dim;i++)param_vals[i]=substates[index_ss[i]].get_param(index_ss_index[i]);
  //construct product space state, then return.
  state outstate(space,param_vals); 
  return outstate;
};

//Take product state of subspace evaluate()s
double independent_dist_product::evaluate(state &s)const{
  if(s.invalid())return 0;
  if(dim!=s.size()){
    cout<<"independent_dist_product:evaluate: State size mismatch.\n";
    exit(1);
  }
  double result=1;
  vector<double> pars=s.get_params_vector();
  //for each subspace map the product-space params to the subspace and evaluate
  for(int i=0;i<Nss;i++){
    int ss_dim=ss[i]->size();
    valarray<double> ss_pars(ss_dim);
    for(int j=0;j<ss_dim;j++)ss_pars[j]=pars[ss_indices[i][j]];
    state substate(ss[i],ss_pars);
    result*=ss_dists[i]->evaluate(substate);
  }
  return result;
};


///Given a unit hypercube point with uniform distribution
///return corresponding parameter state with this distribution
state independent_dist_product::invcdf(const state &s)const{
  //image is the direct product state of subspace invcdf images
  vector<double> pars=s.get_params_vector();
  valarray<double> v(dim);    
  //for each subspace map the product-space params to the subspace
  //then evaluate invcdf on the subspace and map the results back into v
  for(int i=0;i<Nss;i++){
    int ss_dim=ss[i]->size();
    valarray<double> ss_pars(ss_dim);
    for(int j=0;j<ss_dim;j++)ss_pars[j]=pars[ss_indices[i][j]];
    state substate(ss[i],ss_pars);
    state outsubstate=ss_dists[i]->invcdf(substate);
    //Now map back
    for(int j=0;j<ss_dim;j++)v[ss_indices[i][j]]=outsubstate.get_param(j);
  }
  return state(space,v);
};

string independent_dist_product::show(int ii)const{
  //cout<<"Entering IDP: ii="<<ii<<endl;
  ostringstream s;
  if(ii<0){
    s<<"IndependentDistProduct(\n";
    for( int i=0;i<dim;i++)
      s<<"  "<<(space?space->get_name(i):"[???]")<<" from "<<show(i)<<"\n";
    s<<")\n";
  } else {
    if(ii>dim){
      cout<<"independent_dist_product::show: Index out of range."<<endl;
      exit(1);
    }
    s<<ss_dists[index_ss[ii]]->show(index_ss_index[ii]);
  }
  //cout<<"exiting IDP with: "<<s.str()<<endl;
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

