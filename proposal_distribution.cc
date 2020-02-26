///Proposal distirbutions for MCMC
///
///state objects are tied to a stateSpace object which specifies to domain.
///boundary objects allow specification of the domain bounraies in each dimension.
///Written by John G Baker - NASA-GSFC (2013-2015)

#include "proposal_distribution.hh"
#include <algorithm>

using namespace std;

// Static data
int proposal_distribution::idcount=0;

//************** PROPOSAL DISTRIBUTION classes ******************************

proposal_distribution_set* proposal_distribution_set::clone()const{//need a deep clone...
  proposal_distribution_set* result;
  result=new proposal_distribution_set(*this);
  for(int i=0;i<proposals.size();i++){
    result->proposals[i]=proposals[i]->clone();
    //cout<<"proposal_distribution_set::clone(): cloning->new prop="<<result->proposals[i]<<endl;
  }
  return result;
};

proposal_distribution_set involution_proposal_set(const stateSpace &space,double adapt_rate){
  vector<proposal_distribution*> props;
  vector<stateSpaceInvolution > involutions=space.get_potentialSyms();
  vector<double> shares(involutions.size(),1.0);
  for(auto involution : involutions)props.push_back(new involution_proposal(involution));
  return proposal_distribution_set(props,shares,adapt_rate);
};

//A little utility for normalizing the shares and presetting the bin limits.
void proposal_distribution_set::reset_bins(){
  int Nsize=shares.size();
  double Tfac=0;
  if(Tpow>0){
    if(not ch){
      cout<<"proposal_distribution_set::reset_bins(): Thermal scaling requires that chain must be set for proposal."<<endl;
      Tfac=0;
    } else
      Tfac=1-pow(ch->invTemp(),Tpow);
  }
  double sum=0;
  for(int i=0;i<Nsize;i++)sum+=shares[i];//make sure the portions total one
  for(int i=0;i<Nsize;i++){
    shares[i]/=sum;
    if(i==0)bin_max[0]=shares[0];
    else {
      bin_max[i]=bin_max[i-1]+shares[i];
      if(Tpow>0)bin_max[i]+=(hot_shares[i]-shares[i])*Tfac;
    }
  }
};

proposal_distribution_set::proposal_distribution_set(vector<proposal_distribution*> &props,vector<double> &shares_,double adapt_rate,double Tpow, vector<double> hot_shares_,bool take_pointers):shares(shares_),adapt_rate(adapt_rate),Tpow(Tpow),hot_shares(hot_shares_),own_pointers(take_pointers){
  //if Tpow>0 do temperature-based weighting
  //shares[i](T) = shares[i]*(1/T**Tpow) + hot_shares[i]*(1-1/T**Tpow)
  //so shares[i](T=1) = shares[i]
  //and shares[i](1/T=0) = hot_shares[i]
  //First some checks
  if(props.size()!=shares.size()){
    cout<<"proposal_distribution_set(constructor): Array sizes mismatched.\n";
    exit(1);
  }
  Nsize=shares.size();
  if(Tpow>0){
    double sum=0;
    for(int i=0;i<hot_shares.size();i++)sum+=hot_shares[i];//make sure the portions total one
    if(hot_shares.size()!=Nsize or sum<=0){
      cout<<"proposal_distirubtion_set::With Tpow>0 need to provide hot_shares with sum>0"<<endl;
      hot_shares=shares;
    } else for(int i=0;i<Nsize;i++)hot_shares[i]/=sum;
  }
  bin_max.resize(Nsize);
  reset_bins();
  for(int i=0;i<Nsize;i++)proposals.push_back(props[i]);//no copy here.
  last_type=0;
  last_dist=0;
  //Set up stuff for evolving bins
  last_accepted.resize(Nsize,true);
  adapt_count=0;
  adapt_every=10*Nsize; //A somewhat arbitrary update frequency
  
};


  ///For proposals which draw from a chain, we need to know which chain
//void set_chain(chain *c){for(int i=0;i<Nsize;i++)proposals[i]->set_chain(c);};

///Randomly select from proposals i in 0..n and draw.
///Sets type() value at i+10*proposals[i].type() 
//state proposal_distribution_set::draw(state &s,Random &rng){
state proposal_distribution_set::draw(state &s,chain * caller){
  //cout<<"in prop_dist_set::draw(this="<<this<<")"<<endl;
  Random & rng=*(caller->getPRNG());
  int count=0;
  while(true){
    double x;
    if(Nsize>1){
      x=rng.Next();
      //rngout<<x<<" 3"<<endl;
      //cout<<" :"<<x<<" 3 rng="<<&rng<<endl;
    }
    else x=0;//this should make a trival (set of 1) proposal_dist the same as the underlying dist even up to RNG draws.
    for(int i=0;i<Nsize;i++){
      if(proposals[i]->is_ready()&&x<bin_max[i]){
	//state out= proposals[i]->draw(s,rng);
	state out= proposals[i]->draw(s,caller);
	log_hastings=proposals[i]->log_hastings_ratio();
	//cout<<"Applying prop "<<proposals[i]->show();
	last_type=i+10*proposals[i]->type();//We assume no need for >10 prop classes in the set
	last_dist=i;
	return out;
      }
    }
    //cout<<"proposal_distribution_set::draw:Proposals all not ready."<<endl;
    count++;
    if(count>100){//something is wrong.
      cout<<"propsal_distribution_set::draw: Hmmm... Seems that (nearly?) none of the proposals are ready;\n";
      exit(1);
    }
  }
};

//For the adaptive option, we steer the mixing fraction to opt
void proposal_distribution_set::accept(){
  proposal_distribution::accept();
  //Here is the idea:
  //If two accepts in a row reduce weight scaling
  //and if two rejects in a row reduce weight scaling
  //to favor mid-range acceptance rate
  if(adapt_rate==0){//Don't adapt here, but possibly adapt a sub_distribution
    proposals[last_dist]->accept();
    return; 
  };
  if(last_accepted[last_dist])//successive accepts
    shares[last_dist]*=1-adapt_rate*0.25;
  last_accepted[last_dist]=true;
  adapt_count++;
  if(adapt_count>=adapt_every)reset_bins();
  proposals[last_dist]->accept();
};

void proposal_distribution_set::reject(){
  proposal_distribution::reject();
  //Here is the idea:
  //If two accepts in a row reduce weight scaling
  //and if two rejects in a row reduce weight scaling
  //to favor mid-range acceptance rate
  if(adapt_rate==0){//Don't adapt here, but possibly adapt a sub_distribution
    proposals[last_dist]->reject();
    return;
  };
  if(!last_accepted[last_dist])//successive rejects
    shares[last_dist]*=1-adapt_rate*0.25;
  last_accepted[last_dist]=false;
  adapt_count++;
  if(adapt_count>=adapt_every)reset_bins();
  proposals[last_dist]->reject();
};

void proposal_distribution_set::checkpoint(string path){
  //save basic data 
  ostringstream ss;
  ss<<path;
  string dir=ss.str();
  
  if(adapt_rate>0){
    mkdir(dir.data(),ACCESSPERMS);
    ss<<"proposal="<<id<<".cp";
    ofstream os;
    openWrite(os,ss.str());
    writeDoubleVector(os,shares);
    writeDoubleVector(os,bin_max);
    vector<int>last_accepted_int(Nsize);
    for(int i=0;i<Nsize;i++)last_accepted_int[i]=last_accepted[i];//convert bool to int
    writeIntVector(os,last_accepted_int);
    writeInt(os,adapt_count);
    os.close();
  }
  for(auto proposal: proposals)proposal->checkpoint(path);
};

void proposal_distribution_set::restart(string path){
  //save basic data 
  ostringstream ss;
  ss<<path;
  string dir=ss.str();

  if(adapt_rate>0){
    mkdir(dir.data(),ACCESSPERMS);
    ss<<"proposal="<<id<<".cp";
    ifstream is;
    openRead(is,ss.str());
    readDoubleVector(is,shares);
    readDoubleVector(is,bin_max);
    vector<int>last_accepted_int(Nsize);
    readIntVector(is,last_accepted_int);
    for(int i=0;i<Nsize;i++)last_accepted[i]=last_accepted_int[i];//convert bool to int
    readInt(is,adapt_count);
    is.close();
  }
  for(auto proposal: proposals)proposal->restart(path);
};

string proposal_distribution_set::report(int style){//For status reporting on adaptive
  ostringstream ss;
  if(style==0){
    ss<<proposal_distribution::report(style)<<":";
    ss<<"["<<proposals[0]->report(style);
    for(int i=1;i<Nsize;i++){
      ss<<",";
      string rep=proposals[i]->report(style);
      if(rep!="") ss<<proposals[i]->report(style);
    }
    ss<<"]";
  } else if(style==1){
    string rep=proposals[0]->report(style);
    ss<<"shares=["<<bin_max[0];
    if(rep!="")ss<<":"<<rep;
    for(int i=1;i<Nsize;i++){
      //ss<<","<<shares[i];
      ss<<","<<bin_max[i]-bin_max[i-1];
      string rep=proposals[i]->report(style);
      if(rep!="") ss<<":"<<rep;
    }
    ss<<"]";
  }
  return ss.str();
  
};

string proposal_distribution_set::show(){
  ostringstream s;
  s<<"ChooseFrom(";
  double last=0;
  for(uint i=0;i<Nsize;i++){
    s<<"  "<<(bin_max[i]-last)*100.<<"% : "<<proposals[i]->show()<<"\n";
    last=bin_max[i];
  }
  s<<")\n";
  return s.str();
};

    
//DifferentialEvolution
//Based mainly on (ter Braak and Vrugt 08, Stat Comput (2008) 18: 435–446)
//Also SampsonEA2011 ArXiV:1105.2088
//The basic algorithm involves choosing a random state from the past history and proposing a jump in that
// direction. terBraakEA also add a "small" random jump 
//Comments: SampsonEA comment that "It can be shown that this approach is asymptotically Markovian in the limit as one uses the full past history of the chain."  terBraak don't show this, but state "we conjecture that DE- MCZ is ergodic and converges to a stationary distribution with [the right] pdf."  The basis for this is by general reference to the Adaptive Metropolis-Hastings paper (HaarioEA 2001).  


  
//SOME INTERNAL FUNCTIONS
//state differential_evolution::draw_standard(state &s, Random &rng){
state differential_evolution::draw_standard(state &s, chain *caller){
  Random & rng=*(caller->getPRNG());
  //get gamma
  ///ie gamma weights the size of the jump. It is suggested 2.38/sqrt(2*d)
  //cout<<"DE step from s=("<<s.get_string()<<")"<<endl;
  //if(have_chain)cout<<"chain=("<<ch->show()<<")"<<endl;
  double gamma=1.68/sqrt(dim)/reduce_gamma_fac;//Precalc sqrt(d) for speeed?
  double xgamma=rng.Next();
  //rngout<<xgamma<<" 4"<<endl;
  //cout<<" :"<<xgamma<<" 4 rng="<<&rng<<endl;
  if(xgamma<gamma_one_frac)gamma=1;
  //get pieces of proposal-state expression terBraak08 eq 2
  //state s1=draw_from_chain(rng);
  //state s2=draw_from_chain(rng);
  state s1=draw_from_chain(caller);
  state s2=draw_from_chain(caller);
  //cout<<"s1="<<s1.show()<<endl;
  //cout<<"s2="<<s2.show()<<endl;
  gaussian_dist_product edist(nullptr,dim);
  //cout<<"edist:"<<edist.show();
  state e=edist.drawSample(rng);
  //if(caller->get_id()==1)cout<<"e:"<<e.show();
  /*state prop=e.scalar_mult(b_small);
  //cout<<"x=("<<s.get_string()<<")"<<endl;
  //cout<<"e=("<<prop.get_string()<<")"<<endl;
  prop=prop.add(s1.scalar_mult(gamma));
  //cout<<"s1=("<<s1.get_string()<<")"<<endl;
  //cout<<"s2=("<<s2.get_string()<<")"<<endl;
  prop=prop.add(s2.scalar_mult(-gamma));
  //cout<<"e+(s1-s2)*(gamma="<<gamma<<")=("<<prop.get_string()<<")"<<endl;
  prop=prop.add(s);
  //cout<<"x*=("<<prop.get_string()<<")"<<endl;
  */    
  state prop=s;
  //cout<<"prop:"<<prop.show();
  //cout<<"x=("<<s.get_string()<<")"<<endl;
  //cout<<"e=("<<prop.get_string()<<")"<<endl;
  prop.add(e.scalar_mult(b_small));
  prop=prop.add(s1.scalar_mult(gamma));
  prop=prop.add(s2.scalar_mult(-gamma));
  //#pragma omp critical(gleamout)
    
  //may be faster to do this with vectors instead of "states"
  log_hastings=0;
  last_type=0;
  return prop;
};

//state differential_evolution::draw_snooker(state &s, Random & rng){
state differential_evolution::draw_snooker(state &s, chain *caller){
  Random & rng=*(caller->getPRNG());
  //get gamma
  ///i.e. gamma weights the size of the jump, draw from U[1.2,2.2] as in terBraak08
  double xgamma=rng.Next();
  //rngout<<xgamma<<" 5"<<endl;
  //cout<<" :"<<xgamma<<" 5 rng="<<&rng<<endl;
  double gamma=(1.2+xgamma)/reduce_gamma_fac;
  ///get pieces of proposal-state expression terBraak08 eq 3
  //cout<<"Snooker step from s=("<<s.get_string()<<")"<<endl;
  double smznorm2=0;
  state minusz=s,smz=s;
  int isafe=0;
  while(smznorm2==0){//we assure that s!=z to realize perform the required projection meaningfully; wasn't specified in terBraak08. This is a real possibility since there are many repeated states in the chain.  Alternative would be to repropose the current state, which is silly, or we could add a small random piece as in the "standard" draw (That possibility is mentioned in terBraak).
    //if(isafe>1)cout<<"*********************** isafe="<<isafe<<endl; 
    //state z=draw_from_chain(rng);
    state z=draw_from_chain(caller);
    //#pragma omp critical(gleamout)
    //cout<<"draw_snooker("<<caller->get_id()<<"): z=("<<z.get_string()<<")"<<endl;
    //cout<<"z=("<<z.get_string()<<")"<<endl;
    minusz=z.scalar_mult(-1);
    //cout<<"-z=("<<minusz.get_string()<<")"<<endl;
    smz=s.add(minusz);
    //cout<<"s-z=("<<smz.get_string()<<")"<<endl;
    smznorm2=smz.innerprod(smz);
    //cout<<"|s-z|^2="<<smznorm2<<endl;
    isafe++;
    if(isafe>1000){
      cout<<"differential_evolution::draw_snooker: We seem to be stuck in an infinite loop.  Bailing out!"<<endl;
      exit(1);
    }
  }
  //#pragma omp critical(gleamout)
  //cout<<"draw_snooker("<<caller->get_id()<<"): smz=("<<smz.get_string()<<")"<<endl;
  //state s1=draw_from_chain(rng);
  //state s2=draw_from_chain(rng);
  state s1=draw_from_chain(caller);
  state s2=draw_from_chain(caller);
  //cout<<"s1=("<<s1.get_string()<<")"<<endl;
  //cout<<"s2=("<<s2.get_string()<<")"<<endl;
  state ds12=s1.scalar_mult(gamma);
  ds12=ds12.add(s2.scalar_mult(-gamma));
  //cout<<"(s1-s2)*(gamma="<<gamma<<")=("<<ds12.get_string()<<")"<<endl;
  state prop=s;
  prop=prop.add(smz.scalar_mult(ds12.innerprod(smz)/smznorm2));
  state pmz=prop.add(minusz);
  //terBraak08 Eq 4.
  log_hastings=(log(pmz.innerprod(pmz))-log(smznorm2))*(dim-1)/2.0;
  last_type=1;
  //#pragma omp critical(gleamout)
  //cout<<"draw_snooker("<<caller->get_id()<<"): x*=("<<prop.get_string()<<")"<<endl;
    
  return prop;
};

state  differential_evolution::draw_from_chain(chain *caller){
  Random & rng=*(caller->getPRNG());
  if(!is_ready()){
    if(have_chain)cout<<"size="<<ch->size()<<" < "<<get_min_start_size()<<"=minsize"<<endl;
    else cout<<"No chain set!"<<endl;
    cout<<"differential_evolution:draw_from_chain: Chain is not ready. Verify readiness with is_ready() before drawing.\n";
    exit(1);
  }
  int nchains=ch->multiplicity();
  if(nchains==1||!do_support_mixing){
    int index=draw_i_from_chain(caller,ch);
    return ch->getState(index,true);//grab the state with raw indexing, not nominal indexing (that includes unsaved states and no inits)
  }

  //The rest of this is to implement *mixing* of history info from parallel chains.

  //For the case of a chain set we must first choose which chain to draw from.
  //We weight the chains based on their relative temeratures and a set of likelihoods drawn from each chain
  //The motivation for this is to provide that a draws from the typical region of the (non-caller) chain
  //do not overwhelm those of the same region of the caller chain.
  double k[nchains+1];
  int index;
  //const int Nmean=10,Nmedian=10,bstyle=false,beststyle=false;
  const int Nmean=10,Nmedian=10,bstyle=false;
  const double pmix=temperature_mixing_factor;//encourage more mixing
  double l0max=-1e100;
  double l0min=1e100;
  double l0[Nmean];
  //if(bstyle||beststyle)
  if(bstyle)
    for(int i=0;i<Nmean;i++){
      index=draw_i_from_chain(caller,caller);
      double dl=caller->getLogLike(index,true);
      if(isfinite(dl)){
	  if(dl>l0max)l0max=dl;
	  if(dl<l0min)l0min=dl;
	  l0[i]=dl;
	} else i--;
      //cout<<"dll="<<dll<<" <= "<<llmax<<endl;
    }
  
  double beta=caller->invTemp();
  k[0]=0;
  int ithis=0;
  chain *ci=0;
  for(int i=0;i<nchains;i++){
    //if(!bstyle||beststyle)
    ci=ch->subchain(i);
    if(!bstyle){
      l0max=-1e100;
      l0min=1e100;
      for(int j=0;j<Nmean;j++){
	index=draw_i_from_chain(caller,ci);
	double dl=caller->getLogLike(index,true);
	if(isfinite(dl)){
	  if(dl>l0max)l0max=dl;
	  if(dl<l0min)l0min=dl;
	  l0[j]=dl;
	} else j--;
	//cout<<"dl="<<dl<<" <= "<<l0max<<endl;
      }
    }
    double alpha=ci->invTemp(),amb=alpha-beta;
    if(!bstyle)amb=-amb;
    if(amb==0)ithis=i;
    double sum=0;
    double l0scale=amb<0?l0min:l0max;//scale by exp(llmax) or exp(llmin) to avoid overflow
    for(int ii=0;ii<Nmean;ii++)sum+=exp((l0[ii]-l0scale)*amb);
    double ll0=log(sum/Nmean)+l0scale*amb;
    double ll;
    vector<double> l(Nmedian);
    for(int j=0;j<Nmedian;j++){
      index=draw_i_from_chain(caller,ci);
      l[j]=ci->getLogLike(index,true);
    }
    sort(l.begin(),l.end());
    ll=l[(int)Nmedian/2];
    double lk=ll0-ll*amb;
    if(!bstyle)lk=-lk;
    //if(caller->size()%100==0)cout<<" "<<i<<":"<<ll<<","<<ll0/amb<<" "<<ll*amb<<","<<ll0<<" ("<<l0scale<<") "<<amb<<" -> "<<lk;
    lk/=pmix;
    //if(lk>-abs(amb))lk=-abs(amb);
    if(lk>0)lk=0;
    //if(caller->size()%100==0)cout<<" ("<<lk<<")"<<endl;
    k[i+1]=k[i]+exp(lk);
  }
  //cout<<"ithis="<<ithis<<" norm="<<k[nchains]<<endl;
  //for(int i=0;i<nchains;i++)cout<<i<<":"<<int(k[i]*100/k[nchains])<<" ";
  //cout<<endl;
  int ipick=ithis;
  //#pragma omp critical(gleamout)
  {
    //cout<<ithis<<"("<<caller->get_id()<<"):";
    //for(int i=0;i<nchains;i++)if((k[i+1]-k[i])/k[nchains]>0.01)cout<<i<<" ";//list all chains contributing >1%
    double xrnd=rng.Next()*k[nchains];
    for(int i=0;i<nchains;i++){
      if(xrnd<=k[i+1]){
	//cout<<"differential_evolution::draw_from_chain:: Selected chain "<<i<<" of "<<nchains<<endl;
	ipick=i;
	break;
      }
    }
    ci=ch->subchain(ipick);
    index=draw_i_from_chain(caller,ci);
  }
  //cout<<" -> "<<ipick<<" state "<<index<<endl;
  if(ipick<0){
    cout<<"differential_evolution::draw_from_chain:Something went wrong!"<<endl;
    exit(-1);
  }

  //diagnostics:
  const int ireport=500000;
  //static int icount=0;  //these moved to static class data.
  //static vector< vector< int > > mixcounts;
  //static vector< int > trycounts;

  /*  //commented out to facillitate testing of checkpointing
#pragma omp critical
  //diagnostics/reporting on mixing
  {
    if(icount==0){
      trycounts.resize(nchains,0);
      mixcounts.resize(nchains);
      for(int i=0;i<nchains;i++)
	mixcounts[i].resize(nchains,0);
    }
    icount++;
    trycounts[ithis]++;
    mixcounts[ithis][ipick]++;
    if(icount==ireport){
      cout<<"differential_evolution::mixing report: (>0.5 percent):"<<endl;
      for(int i=0;i<nchains;i++){
	int tries=trycounts[i];
	cout<<" "<<i<<":";
	for(int j=0;j<nchains;j++){
	  int picks=mixcounts[i][j];
	  if(picks*200>tries)
	    cout<<"\t"<<j<<":"<<(picks*100)/tries;
	}
	cout<<endl;
      }
      icount=0;//restart the count;
    }
  }
  */
  
  return ci->getState(index,true);//grab the state with raw indexing, not nominal indexing (that includes unsaved states and no inits)
};
  
//draw a sample from a specified chain
int  differential_evolution::draw_i_from_chain(chain *caller,chain *c){
  Random & rng=*(caller->getPRNG());
  if(!is_ready()){
    if(have_chain)cout<<"size="<<c->size()<<" < "<<get_min_start_size()<<"=minsize"<<endl;
    else cout<<"No chain set!"<<endl;
    cout<<"differential_evolution:draw_i_from_chain: Chain is not ready. Verify readiness with is_ready() before drawing.\n";
    exit(1);
  }
  //cout<<"drawing from chain: (this="<<this<<",ch="<<ch<<") "<<ch->show()<<endl;
  int size=c->size();
  int start=0;
  int mins=get_min_start_size();
  int minc=get_min_cut_size();
  if((size-minc)*(1-ignore_frac)>mins)start=(size-minc)*ignore_frac;

  //If unlikely_alpha>0, we reduce the probability of drawing samples from excessively unlikely
  //regions of the chain history.  As a starting place we take the maximum (log) posterior value
  //so far, then subtract the space dimension.  Above this value we take everything. Below this
  //we accept some, we probability p=exp(alpha*(lpost-lpost0)))
  double lpost0=ch->getMAPlpost()-ch->getDim();
  while(true){
    double xrnd=rng.Next();
    int index = start+(size-start)*xrnd;
    double lpost=ch->getLogPost(index,true);
    if(unlikely_alpha>0 and lpost0>lpost){
      double p=exp(unlikely_alpha*(lpost-lpost0));
      xrnd=rng.Next();
      //cout<<"lp0,lp,p,x:"<<lpost0<<" "<<lpost<<" "<<p<<" "<<xrnd<<endl;
      if(xrnd<p)return index;
    } else return index;
  }
  //return index;
};


differential_evolution::differential_evolution(double snooker, double gamma_one_frac,double b_small,double ignore_frac,double unlikely_alpha):gamma_one_frac(gamma_one_frac),b_small(b_small),ignore_frac(ignore_frac),snooker(snooker),unlikely_alpha(unlikely_alpha){
  reduce_gamma_fac=1;
  have_chain=false;
  do_support_mixing=false;
  temperature_mixing_factor=1;
};


//state differential_evolution::draw(state &s,Random &rng){
state differential_evolution::draw(state &s,chain *caller){
  Random & rng=*(caller->getPRNG());
  //cout<<"in DE::draw(this="<<this<<")"<<endl;
  double x=rng.Next(); //pick a number
  //rngout<<x<<" 7"<<endl;
  //#pragma omp critical
  //cout<<" :"<<x<<" 7 rng="<<&rng<<endl;
  //if(snooker>x)return draw_snooker(s,rng);
  if(snooker>x)return draw_snooker(s,caller);
  //return draw_standard(s,rng);
  return draw_standard(s,caller);
};

string differential_evolution::show(){
  ostringstream s;
  s<<"DifferentialEvolution(";
  s<<"snooker="<<snooker<<", gamma_one_frac="<<gamma_one_frac<<", b_small="<<b_small<<", ignore_frac="<<ignore_frac<<")\n";
  return s.str();
};

