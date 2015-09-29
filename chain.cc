///Chains for MCMC
///
///The base class chain, is not useful, but MH_chain and parallel_tempering_chain should be.
///John G Baker - NASA-GSFC (2013-2014)
#include "chain.hh"
#include "proposal_distribution.hh"

// Static data
int chain::idcount=0;


//** CHAIN Classes *******

// A markov (or non-Markovian) chain based on some variant of the Metropolis-Hastings algorithm
// May add "burn-in" distinction later.
MH_chain::MH_chain(probability_function * log_likelihood, sampleable_probability_function *log_prior,double minPrior,int add_every_N):
  llikelihood(log_likelihood),lprior(log_prior),minPrior(minPrior),add_every_N(add_every_N){
  Nsize=0;Nhist=0;invtemp=1;Ntries=1;Naccept=1;last_type=-1;
  dim=log_prior->getDim();
  default_prop_set=false;
  //cout<<"creating chain at this="<<this->show()<<" with lprior="<<lprior->show()<<endl;//debug
  Ninit=0;
};

void MH_chain::reserve(int nmore){//If you know how long you are going to run, it can be more efficient to reserve up front, rather than resizing ever step
    states.reserve(Nsize+nmore);
    lposts.reserve(Nsize+nmore);
    llikes.reserve(Nsize+nmore);
    acceptance_ratio.reserve(Nsize+nmore);
    invtemps.reserve(Nsize+nmore);
  };

void MH_chain::initialize(uint n){
  if(Nhist>0){
    cout<<"MH_chain::initialize: Cannot re-initialize."<<endl;
    exit(1);
  }
  Ninit=n;
  for(uint i=0;i<n;i++){
    state s=lprior->drawSample(*rng);
    int icnt=0,icntmax=100;
    while(s.invalid()&&icnt<100){
      icnt++;
      if(icnt>=icntmax)
	cout<<"MH_chain::initialize: Having trouble drawing a valid state.  Latest state:"<<s.show()<<"...was invalid in space:"<<s.getSpace()->show()<<endl;
      s=lprior->drawSample(*rng);
    }
    //cout <<"starting with Nsize="<<Nsize<<endl;//debug
    add_state(s);
  }
  Nhist=0;//Don't count these.
};

void MH_chain::reboot(){
  Nsize=0;Nhist=0;Ntries=1;Naccept=1;last_type=-1;
  cout<<"Rebooting chain (id="<<id<<")"<<endl;
  Nfrozen=-1;
  types.clear();
  states.clear();
  lposts.clear();
  llikes.clear();
  acceptance_ratio.clear();
  invtemps.clear(); 
  initialize(Ninit);
};

void MH_chain::add_state(state newstate,double log_like,double log_post){
  //cout<<"this="<<0<<",adding state: like="<<log_like<<",post="<<log_post<<endl;
  //if log_like or log_post can be passed in to save computation.  Values passed in are assumed to equal the evaluation results.
    //Value 999 signals need to reevaluate.  If true evaluated value was 999, then reevaluating should yield 999 anyway.
    //double newllike=log_like;
    current_llike=log_like;
    if(log_like==999)current_llike=llikelihood->evaluate_log(newstate);
    current_lpost=log_post;
    if(log_post==999)current_lpost=lprior->evaluate_log(newstate)+invtemp*current_llike;
    current_state=newstate;
    //cout<<"testing:"<<Nhist<<"%"<<add_every_N<<"="<<Nhist%add_every_N<<endl;
    if(Nhist%add_every_N==0){//save state info every add_every_N
      //cout<<" ADDED state "<<Nsize<<":"<<newstate.get_string()<<endl;
      //cout<<"adding state "<<Nsize<<":"<<newstate.get_string()<<endl;
      states.push_back(current_state);
      lposts.push_back(current_lpost);
      llikes.push_back(current_llike);
      acceptance_ratio.push_back(Naccept/(double)Ntries);
      invtemps.push_back(invtemp);
      types.push_back(last_type);
      Nsize++;
    }
    Nhist++;
    //cout<<"Added state "<<Nsize-1<<" = ("<<newstate.get_string()<<") lposts,llikes->"<<newlpost<<","<<newllike<<" = "<<lposts[Nsize-1]<<","<<llikes[Nsize-1]<<endl;
    //cout<<"Nsize="<<Nsize<<endl;//debug
};

void MH_chain::set_proposal(proposal_distribution &proposal){
    default_prop=&proposal;
    default_prop->set_chain(this);
    default_prop_set=true;  
    //Other possible actions, such as a burn-in chain, or a combination of chains would be possible here.
};

void MH_chain::step(){
    if(!default_prop_set){
      cout<<"MH_chain::step() cannot step without setting a proposal.  Either use set_propsal or specify in  the step call."<<endl;
      exit(1);
    }
    step(*default_prop);
};

void MH_chain::step(proposal_distribution &prop,void *data){
    if(Nsize==0){
      cout<<"MH_chain:step: Can't step before initializing chain.\n"<<endl;
      //cout<<"this="<<this<<endl;
      exit(1);
    }
    //cout<<"in MH_chain::step(prop="<<&prop<<")"<<endl;
    double oldlprior=current_lpost-invtemp*current_llike;
    //state newstate=prop.draw(current_state,*rng);
    state newstate=prop.draw(current_state,this);
    double newlike,newlpost,newlprior=lprior->evaluate_log(newstate);
    //cout<<"MH_chain::step: newlprior="<<newlprior<<endl;
    if(newlprior-oldlprior>minPrior){//Avoid spurious likelihood calls where prior effectively vanishes.  
      newlike=llikelihood->evaluate_log(newstate);
      newlpost=newlike*invtemp+newlprior;
    } else {
      //cout<<"for state:"<<newstate.show()<<endl;//debug
      //cout<<"lprior="<<newlprior<<"-"<<oldlprior<<"<"<<minPrior<<" -> outside prior range:"<<lprior->show()<<endl;//debug
      newlike=newlpost=-INFINITY;
    }
    //Now the test: 
    double log_hastings_ratio=prop.log_hastings_ratio();
    log_hastings_ratio+=newlpost-current_lpost;
    bool accept=true;
    //cout<<Nhist<<"("<<invtemp<<"): ("<<newlike<<","<<newlpost<<")vs.("<<oldlpost<<")->"<<log_hastings_ratio<<endl;//debug
    if(log_hastings_ratio<0){
      double x=get_uniform(); //pick a number
      accept=(log(x)<log_hastings_ratio);
      //cout<<"     log(x)="<<log(x)<<" -> "<<(accept?"accept":"reject")<<endl;//debug
    }
    //#pragma omp critical    
    //cout<<"("<<get_id()<<"):"<<newlike<<" "<<log_hastings_ratio<<" -> "<<(accept?"accepted":"rejected")<<endl;
    Ntries++;
    if(accept){
      Naccept++;
      //cout<<"        accepting"<<endl;//debug
      last_type=prop.type();
      add_state(newstate,newlike,newlpost);
      //chainsteps.push_back(Nhist-Ninit);
    }
    else {
      //cout<<"        rejecting"<<endl;//debug
      add_state(current_state,current_llike,current_lpost);
      //cout<<"Nhist="<<Nhist<<"        rejected"<<endl;
    }
    // cout<<"incremented NHist="<<Nhist<<endl;
};

double MH_chain::expectation(double (*test_func)(state s),int Nburn){
    double sum=0;
    for(int i=Nburn;i<Nhist;i+=add_every_N)sum+=test_func(states[get_state_idx(i)]);
    return sum/int((Nhist-Nburn)/add_every_N);
};

int MH_chain::get_state_idx(int i){
  if(i<0||i>=Nhist)i=Nhist-1;
  return Ninit+i/add_every_N;
}

double MH_chain::variance(double (*test_func)(state s),double fmean,int Nburn){
    double sum=0;
    for(int i=Nburn;i<Nhist;i+=add_every_N){
      double diff=test_func(states[get_state_idx(i)])-fmean;
      sum+=diff*diff;
    }
    return sum/int((Nhist-Nburn)/add_every_N);
};

state MH_chain::getState(int elem,bool raw_indexing){//defaults (-1,false)  
  //cout<<"MH_chain::getState()"<<endl;
  if(elem<0||elem>=Nsize||(!raw_indexing&&elem>=Nhist)){
    //cout<<"returning current state."<<endl;
    return current_state;//Current state by default
  } else if (raw_indexing) { 
    //cout<<"returning states["<<elem<<"]"<<endl;
    return states[elem];
  } else {
    //cout<<"returning states["<<get_state_idx(elem)<<"]"<<endl;
    return states[get_state_idx(elem)];
  }
};

double MH_chain::getLogPost(int elem,bool raw_indexing){//defaults (-1,false)  
  if(elem<0||elem>=Nsize||(!raw_indexing&&elem>=Nhist))
    return current_lpost;//Current state by default
  else if (raw_indexing) 
    return lposts[elem];
  else
    return lposts[get_state_idx(elem)];
};

double MH_chain::getLogLike(int elem,bool raw_indexing){
  if(elem<0||elem>=Nsize||(!raw_indexing&&elem>=Nhist))
      return current_llike;
  else if (raw_indexing)
      return llikes[elem];
  else
      return llikes[get_state_idx(elem)];
};

void MH_chain::resetTemp(double new_invtemp){
  invtemp=new_invtemp;
  current_lpost=current_lpost=lprior->evaluate_log(current_state)+invtemp*current_llike;;
};
   
string MH_chain::getStateStringLabels(){
    ostringstream s;
    int np=getState().size();
    s<<"#Nhist: log(posterior) acceptance_ratio: ";
    for(int i=0;i<np-1;i++)s<<"param("<<i<<") ";
    s<<"param("<<np-1<<")";
    return s.str();
};

string MH_chain::getStateString(){
    ostringstream s;
    int np=getState().size();
    s<<Nhist<<": "<<current_lpost<<" "<<acceptance_ratio[get_state_idx()]<<": ";
    vector<double> pars=getState().get_params_vector();
    for(int i=0;i<np-1;i++)s<<pars[i]<<" ";
    s<<pars[np-1];
    return s.str();
};

void MH_chain::dumpChain(ostream &os,int Nburn,int ievery){
    if(Nsize==0)return;
    int np=states[0].size();
    os<<"#Ninit="<<Ninit<<", Nburn="<<Nburn<<"\n";
    os<<"#eval: log(posterior) log(likelihood) acceptance_ratio prop_type: ";
    for(int i=0;i<np-1;i++)os<<"param("<<i<<") ";
    os<<"param("<<np-1<<")"<<endl;
    if(Nburn+Ninit<0)Nburn=-Ninit;
    for(int i=Nburn;i<Nhist;i+=ievery){
      int idx=Ninit+i;
      if(i>=0)idx=get_state_idx(i);
      os<<i<<" "<<lposts[idx]<<" "<<llikes[idx]<<" "<<acceptance_ratio[idx]<<" "<<types[idx]<<": ";
      vector<double>pars=states[idx].get_params_vector();
      //cout<<i<<"=i<states.size()="<<states.size()<<"?"<<endl;//debug
      //cout<<"state:"<<states[i].show()<<endl;
      for(int j=0;j<np-1;j++)os<<pars[j]<<" ";
      os<<pars[np-1]<<endl;
    }
};
  
string MH_chain::show(){
    ostringstream s;
    s<<"MH_chain(id="<<id<<",every="<<add_every_N<<",invtemp="<<invtemp<<",size="<<Nsize<<",N="<<Nhist<<")\n";
    return s.str();
};

string MH_chain::status(){
    ostringstream s;
    //s<<"chain(id="<<id<<", N="<<Nsize<<", T="<<1.0/invtemp<<"):"<<lposts[Nsize-1]<<", "<<llikes[Nsize-1]<<" : "<<this->getState().get_string();
    s<<"chain(id="<<id<<", N="<<Nsize<<", T="<<1.0/invtemp<<"):"<<current_lpost<<", "<<current_llike<<" : "<<this->getState().get_string();
    return s.str();
  };



// A parallel tempering set of markov (or non-Markovian) chain
// May add "burn-in" distinction later.

parallel_tempering_chains::parallel_tempering_chains(int Ntemps,int Tmax,double swap_rate,int add_every_N):Ntemps(Ntemps),swap_rate(swap_rate),add_every_N(add_every_N){
    props.resize(Ntemps);
    directions.resize(Ntemps,0);
    instances.resize(Ntemps,-1);
    instance_starts.resize(Ntemps,0);
    directions.resize(Ntemps,0);
    ups.resize(Ntemps,0);
    downs.resize(Ntemps,0);
    log_eratio_up.resize(Ntemps-1,0);
    log_eratio_down.resize(Ntemps-1,0);
    tryrate.resize(Ntemps-1,0);
    swaprate.resize(Ntemps-1,0);
    up_frac.resize(Ntemps,0);
    swap_accept_count.assign(Ntemps-1,0);
    swap_count.assign(Ntemps-1,0);
    temps.assign(Ntemps,0);
    Nsize=0;
    //make geometrically spaced temperature set.
    double tratio=exp(log(Tmax)/(Ntemps-1));
    temps[0]=1;
    for(int i=1;i<Ntemps;i++)temps[i]=temps[i-1]*tratio;
    Ninit=0;
    do_evolve_temps=false;
    max_reboot_rate=0;
    test_reboot_every=10000;
    reboot_grace=0;
    reboot_thresh=100;
    reboot_thermal_thresh=0;
    reboot_aggression=0;
    maxswapsperstep=1+2*swap_rate*Ntemps;
};

void parallel_tempering_chains::initialize( probability_function *log_likelihood, sampleable_probability_function *log_prior,int n){
  Ninit=n;
  dim=log_prior->getDim();
  for(int i=0;i<Ntemps;i++){
    MH_chain c(log_likelihood, log_prior,-30,add_every_N);
    chains.push_back(c);
  }
  //#pragma omp parallel for schedule (guided, 1)  ///try big chunks first, then specialize
#pragma omp parallel for schedule (dynamic, 1) ///take one pass/thread at a time until done.
  for(int i=0;i<Ntemps;i++){
    cout<<"PTchain: initializing chain "<<i<<endl;
    chains[i].initialize(n);
    // chains[i].resetTemp(1/temps[i]);
    chains[i].invtemp=1/temps[i];
    instances[i]=i;
    instance_starts[i]=chains[i].size();
    //cout<<"initialized chain "<<i<<" at "<<&chains[i]<<endl;
    //ichains[i]=i;
    if(i==0)directions[i]=-1;
    else if(i==Ntemps-1)directions[i]=1;
    else directions[i]=0;
    ups[i]=0;
    downs[i]=0;
    Nsize=c0().size(); 
  };
};

void parallel_tempering_chains::set_proposal(proposal_distribution &proposal){
    //cout<<"("<<this<<")ptchain::set_proposal(proposal="<<&proposal<<")"<<endl;
    for(int i=0;i<Ntemps;i++){
      props[i]=proposal.clone();
      //cout<<"cloning proposal: props["<<i<<"]="<<props[i]<<endl;
      //cout<<"                    chain=("<<&chains[i]<<")"<<chains[i].show()<<endl;
      if(props[i]->support_mixing()){
	props[i]->set_chain(this);
	//cout<<"Supporting chain mixing in proposal distribution."<<endl;
}
      else
	props[i]->set_chain(&chains[i]); //*** This seems to set the last chain to all props...***
      //Other possible actions, such as a burn-in chain, or a combination of chains would be possible here.
    }
};

void parallel_tempering_chains::step(){
  int iswaps[maxswapsperstep];
  double x;

  //diagnostics and steering: set up
  const int ireport=10000;//should make this user adjustable.
  static int icount=0;
  static vector< int > swapcounts;
  static vector< int > trycounts;
  if(icount==0){
    trycounts.clear();
    trycounts.resize(Ntemps,0);
    swapcounts.clear();
    swapcounts.resize(Ntemps,0);
  }
  
  //Determine which chains to try for swaps
  for(int i=0;i<maxswapsperstep;i++){
    iswaps[i]=-2;
    if(Ntemps>1&&get_uniform()<(Ntemps-1)*swap_rate/maxswapsperstep){//do swap 
      iswaps[i]=int(get_uniform()*(Ntemps-1));
      //cout<<"trying "<<iswaps[i]<<endl;
      for(int j=0;j<i;j++)//Don't allow adjacent swaps in same step;
	if(iswaps[j]==iswaps[i]||iswaps[j]+1==iswaps[i])iswaps[i]=-2;
    }
  }
  //for(int j=0;j<maxswapsperstep;j++)cout<<"\t"<<iswaps[j];
  //cout<<endl;
  //Now perform the swap trials:
  for(int j=0;j<maxswapsperstep;j++)if(iswaps[j]>=0){
      bool accept=true;
      trycounts[iswaps[j]]++;
      //cout<<"iswaps["<<j<<"]="<<iswaps[j]<<endl;
    //do swap 
    //This algorithm assumes swap_rate<1/Ntemps, which isn't always true.
    //It maybe should be that Nswaps (per step) = int(swap_rate*Ntemps*get_uniform) (possibly/2 as well to get perchain rate)
    //(above, if Ntemps==1,we avoid calling RNG for equivalent behavior to single chain)
    //pick a chain
    int i=iswaps[j];
    //diagnostic records first
    if(i>0){
      if(directions[i]>0)ups[i]++;
      if(directions[i]<0)downs[i]++;
    }
    //cout<<"LogLikes:"<<chains[i].getLogLike()<<" "<<chains[i+1].getLogLike()<<endl;
    //cout<<"invtemps:"<<chains[i].invtemp<<" "<<chains[i+1].invtemp<<endl;
    double log_hastings_ratio=-(chains[i+1].invtemp-chains[i].invtemp)*(chains[i+1].getLogLike()-chains[i].getLogLike());//Follows from (21) of LittenbergEA09.
    if(log_hastings_ratio<0){
      x=get_uniform(); //pick a number
      accept=(log(x)<log_hastings_ratio);
    }
    //cout<<i<<" "<<log_hastings_ratio<<" -> "<<(accept?"Swap":"----")<<endl;
    if(accept){
      //we swap states and leave temp fixed.
      state sA=chains[i].getState();
      double llikeA=chains[i].getLogLike();
      //double lpostA=chains[i].getLogPost();  //This doesn't work since posterior depends on temperature.
      state sB=chains[i+1].getState();
      //double lpostB=chains[i+1].getLogPost();
      double llikeB=chains[i+1].getLogLike();
      double dirB=directions[i+1];
      //chains[i+1].add_state(sA,llikeA,lpostA);
      chains[i+1].add_state(sA,llikeA);  //Allow the chain to compute posterior itself.
      //chains[i].add_state(sB,llikeB,lpostB);
      chains[i].add_state(sB,llikeB);
      int instanceB=instances[i+1];
      instances[i+1]=instances[i];
      instances[i]=instanceB;
      directions[i+1]=directions[i];
      directions[i]=dirB;
      if(i==0)directions[i]=1;
      if(i+1==Ntemps-1)directions[i+1]=-1;
      swap_accept_count[i]++;
      swapcounts[iswaps[j]]++;
      if(do_evolve_temps){
	pry_temps(i,evolve_temp_rate);
      }
    } else { //Conceptually this seems necessary.  It was missing before 11-16-2014.  Didn't notice any problems though, before, though.
      state sA=chains[i].getState();
      double llikeA=chains[i].getLogLike();
      double lpostA=chains[i].getLogPost();  //This does work because temperature doesn't change.
      state sB=chains[i+1].getState();
      double llikeB=chains[i+1].getLogLike();
      double lpostB=chains[i+1].getLogPost();
      chains[i].add_state(sA,llikeA,lpostA);  //For each chain, this is is just a copy of the current state
      chains[i+1].add_state(sB,llikeB,lpostB);//because we tried and failed with a new candidate.
    }	
    swap_count[i]++;
  }
  
  //Either guided or dynamic scheduling seems to work about the same.
  //#pragma omp parallel for schedule (guided, 1)  ///try big chunks first, then specialize 
#pragma omp parallel for schedule (dynamic, 1) ///take one pass/thread at a time until done. 
  for(int i=0;i<Ntemps;i++){
    //NOTE: for thread-independent results with omp and DE-cross-breeding temp chains,
    //we need to restrict the de history for each chain to the part before any updates here. 
    chains[i].history_freeze();
    //cout<<"Calling ("<<this<<")ptchain::step() for chain="<<&chains[i]<<" with prop="<<props[i]<<endl;
    //cout<<"Calling step() for chain "<<i<<endl;
    bool skip_this=false;;
    for(int j=0;j<maxswapsperstep;j++)
      if(i==iswaps[j]||i==iswaps[j]+1)skip_this=true;//skip additional steps from chains that were already tested for swapping.
    if(skip_this)continue;
    chains[i].step(*props[i]);
  }
  
  for(int i=0;i<Ntemps;i++)chains[i].history_thaw();
  Nsize=c0().size();
  //cout<<status()<<endl;
  
  //diagnostics and steering:
  icount++;
  if(icount>=ireport){
    for(int i=0;i<Ntemps-1;i++){
      tryrate[i]=trycounts[i]/(double)icount;
      swaprate[i]=swapcounts[i]/(double)icount;
      //Compute upside log_evidence ratio
      log_eratio_up[i]=   log_evidence_ratio(i  ,i+1,ireport,add_every_N);
      log_eratio_down[i]=-log_evidence_ratio(i+1,i  ,ireport,add_every_N);
    }
    //Compute up/down fracs (and reset count)
    for(int i=0;i<Ntemps;i++){
      if(i==0)up_frac[i]=1;      
      else if(i==Ntemps-1)up_frac[i]=0;      
      else up_frac[i]=ups[i]/(double)(ups[i]+downs[i]);
      ups[i]=downs[i]=0;
    }
    icount=0;
  }
  if((Nsize-Ninit)%test_reboot_every==0){
    //perhaps reboot an underperforming chain.
    //it is convenient to do this at the same time as reporting though this may need to change
    int rcount=0;
    if(max_reboot_rate>0){
      cout<<"step "<<Nsize<<":max_reboot_rate="<<max_reboot_rate<<"  test_reboot_every="<<test_reboot_every<<endl;
      double fac=max_reboot_rate*test_reboot_every;
      int nreboot=0;
      for(int i=0;i<Ntemps;i++){
	fac*=0.5;
	if(fac>get_uniform())nreboot++;
      }
      if(nreboot>Ntemps)nreboot=Ntemps;
      cout<<"Nreboot="<<nreboot<<endl;
      vector<double>maxes(Ntemps-1,-1e100);
      double max=-1e100;
      //first we get the maximum
      for(int i=0;i<Ntemps-1;i++){
	if(chains[i].getLogPost()>chains[i+1].getLogPost()){
	  instance_starts[instances[i]]=(instance_starts[instances[i]]+Nsize)/2;//give an extension on grace period to this instance for improvement
	}
	if(chains[i].getLogPost()>max)max=chains[i].getLogPost();
	maxes[i]=max;
	
      }
      int ntry=0;
      for(int i=Ntemps-1;i>0;i--){
	//We consider rebooting the hottest chains first (only rebooting nreboot per turn at most) 
	//out criterion for killing is that the posterior is a billion times less than the
	//max likelihood of any colder chain.  In this case meaningful exchange between the chains may be
	//considered unlikely.
	if(rcount==nreboot)break;//reached the max-rate limit.
	double thresh=reboot_thresh+reboot_thermal_thresh*chains[i].invTemp();
	double age=Nsize-instance_starts[instances[i]];
	double cage=chains[i].size();
	double val=maxes[i-1]-chains[i].getLogPost();
	//double blindly=reboot_aggression*(nreboot-ntry)/(double)(nreboot+i);
	double blindly=reboot_aggression*(nreboot-ntry*i/(double)Ntemps)/(double)(nreboot+i);
	double agrace=reboot_grace;
	if(reboot_graduate)agrace*=2*(1-i/(double)Ntemps);//Allow linearly longer development for instances that are at colder temp levels
	double cgrace=reboot_grace;
	cout<<i<<"["<<1/chains[i].invTemp()<<"]: ("<<age<<">"<<agrace<<", "<<cage<<">"<<cgrace<<"?) "<<chains[i-1].getLogPost()<<"(or max) - "<<chains[i].getLogPost()<<"="<<val<<" > "<<thresh<<"? aggression="<<blindly<<endl;
	if(val>thresh||(blindly>0&&blindly>get_uniform())){//second part;  If there are no gaps detected yet, then we are going to start killing off randomly...
	  ntry++;
	  //if(chains[i-1].getLogPost()-chains[i].getLogPost()>reboot_thresh){
	  //Now check if this instance and this chain have surpassed the grace period.
	  //This is to prevent the instance from being repeatedly reset before it reaches some basic quasi-equilibrium.
	  //And to allow the chain to mature long enough for a meaningful proposal distribution.
	  if(age>agrace&&cage>cgrace){
	    cout<<i<<": ";	    
	    chains[i].reboot();     
	    instance_starts[instances[i]]=Nsize;
	    for(int j=i;j<Ntemps-1;j++){//swap chain back to the lowest temp
	      //cout<<chains[j].status()<<"\n"<<chains[j+1].status()<<endl;
	      double tmptemp=chains[j+1].invTemp();
	      chains[j+1].resetTemp(chains[j].invTemp());
	      chains[j].resetTemp(tmptemp);
	      swap(chains[j+1],chains[j]);
	      swap(instances[j+1],instances[j]);
	      //cout<<"changed to:\n"<<chains[j].status()<<"\n"<<chains[j+1].status()<<endl;
	    }
	    rcount++;
	  } else {
	    cout<<"*immature*"<<endl;
	  }
	}
      }
      
    }
  }
};


//a scheme for evolving the temperatures to encourage mixing:
void parallel_tempering_chains::pry_temps(int ipry, double rate){
  vector< double > splits(Ntemps-1);
  for(int i=0;i<Ntemps-1;i++)
    splits[i]=chains[i].invTemp()-chains[i+1].invTemp();
  //dTold=1-chains[Ntemps-1].invTemp()
  //dTnew=1-chains[Ntemps-1].invTemp()+rate*splits[ipry]
  double norm=1+rate*splits[ipry]/(1-chains[Ntemps-1].invTemp());
  splits[ipry]*=1.0+rate;
  double invtemp=1;
  for(int i=1;i<Ntemps-1;i++){
    invtemp-=splits[i-1]/norm;
    //if(i==ipry)cout<<"*";
    //cout<<i<<"changing invtemp from "<<chains[i].invTemp();
    chains[i].resetTemp(invtemp);
    //cout<<"  --->   "<<chains[i].invTemp()<<endl;;
  }
}
  
  
//This function computes the evidence ratio between chains a two different temps;
double parallel_tempering_chains::log_evidence_ratio(int ia,int ib,int ilen,int every){
  double size=chains[ib].size();
  if(ilen<size)ilen=size;
  double amb=chains[ia].invTemp()-chains[ib].invTemp();
  //to avoid possible overflow we offset by max(lnL*amb) before the sum;
  double xmax=-1e100;
  for(int i=size-ilen;i<size;i+=every){
    double x=chains[ib].getLogLike(i)*amb;
    if(x>xmax)xmax=x;
  }
  double sum=0;
  for(int i=size-ilen;i<size;i+=every){
    double x=chains[ib].getLogLike(i)*amb-xmax;
    if(x>-35)sum+=exp(x);
  }
  double result=log(sum/ceil(ilen/(double)every))+xmax;
  cout<<"log_eratio: amb="<<amb<<" xmax="<<xmax<<" --> "<<result<<endl;
  return result;
};


  ///reference to zero-temerature chain.
  //MH_chain & c0(){return chains[0];};
  //state getState(int elem=-1){return c0().getState(elem);};
  //double getLogPost(int elem=-1){return c0().getLogPost(elem);};
  //void dumpChain(ostream &os,int Nburn=0,int ievery=1){  dumpChain(0,os,Nburn,ievery);}
  //void dumpChain(int ichain,ostream &os,int Nburn=0,int ievery=1){
  //  chains[ichain].dumpChain(os,Nburn,ievery);
  //};

void parallel_tempering_chains::dumpTempStats(ostream &os){
    if(Nsize==0)return;
    os<<"#T0 up_frac0 up-swap_ratio0-1"<<endl;
    for(int i=0;i<Ntemps;i++){
      double up_frac;
      if(i==0)up_frac=1;      
      else if(i==Ntemps-1)up_frac=0;      
      else up_frac=ups[i]/(double)(ups[i]+downs[i]);
      os<<temps[i]<<" "<<up_frac<<" ";
      //cout<<"i="<<i<<" ups="<<ups[i]<<" downs="<<downs[i]<<endl;
      if(i<Ntemps-1)os<<swap_accept_count[i]/(double)swap_count[i]<<": ";
      //else os<<0;
      os<<endl;	     
    }
    os<<"\n"<<endl;
};

string parallel_tempering_chains::show(){
    ostringstream s;
    s<<"parallel_tempering_chains(id="<<id<<"Ntemps="<<Ntemps<<"size="<<Nsize<<")\n";
    return s.str();
};

string parallel_tempering_chains::status(){
    ostringstream s;
    s<<"chain(id="<<id<<", Ntemps="<<Ntemps<<"):\n";
    for(int i=0;i<Ntemps;i++){
      s<<"instance "<<instances[i]<<"["<<Nsize-instance_starts[instances[i]]<<"]("<<(directions[i]>0?"+":"-")<<" , "<<up_frac[i]<<"):                     "<<chains[i].status()<<"\n";
      //cout<<swaprate.size()<<" "<<tryrate.size()<<" "<< log_eratio_down.size()<<" "<< log_eratio_up.size()<<endl;
      if(i<Ntemps-1)s<<"-><-("<<swaprate[i]<<" of "<<tryrate[i]<<"): log eratio:("<<log_eratio_down[i]<<","<<log_eratio_up[i]<<")"<<endl;
    }
    return s.str();
};
 
  
    
