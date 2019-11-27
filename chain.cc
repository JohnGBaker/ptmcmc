///Chains for MCMC
///
///The base class chain, is not useful, but MH_chain and parallel_tempering_chain should be.
///John G Baker - NASA-GSFC (2013-2014)
#include <algorithm>
#include "chain.hh"
#include "proposal_distribution.hh"
#ifdef USE_MPI
#include <mpi.h>
#else
#define MPI_Allgather(a,b,c,d,e,f,g)
#define MPI_Barrier(a)
#define MPI_DOUBLE
#define MPI_COMM_WORLD
#endif

bool chain_verbose=false;

// Static data
int chain::idcount=0;

//** CHAIN Classes *******

//Main chain base class

void chain::checkpoint(string path){
  //save basic data 
  ostringstream ss;
  ss<<path<<"chain"<<id<<"-cp/";
  string dir=ss.str();
  
  if(outputAllowed()){
    mkdir(dir.data(),ACCESSPERMS);
    ss<<"chain.cp";
    ofstream os;
    openWrite(os,ss.str());
    writeInt(os, id);
    writeInt(os, Nsize);
    writeInt(os, Ninit); //Maybe move this history stuff to a subclass
    writeInt(os, Nearliest);
    writeInt(os, Nfrozen);
    writeInt(os, dim);
    os.close();
    //write RNG
    rng->SetInstanceDirectory(dir.data());
    rng->CopyInstanceSeedToDisk();
  }
};

void chain::restart(string path){
  //restore basic data;
  ostringstream ss;
  ss<<path<<"chain"<<id<<"-cp/";
  string dir=ss.str();
  ss<<"chain.cp";
  ifstream is;
  openRead(is,ss.str());
  readInt(is, id);
  readInt(is, Nsize);
  cout<<"id "<<id<<":Nsize="<<Nsize<<endl;
  readInt(is, Ninit); //Maybe move this history stuff to a subclass
  readInt(is, Nearliest);
  readInt(is, Nfrozen);
  readInt(is, dim);
  is.close();
  //write RNG
  rng.reset(new MotherOfAll(rng->Next()));
  rng->SetInstanceDirectory(dir.data());
  rng->CopyInstanceSeedFromDisk();
};

void chain::inNsigma(int Nsigma,vector<int> & indicies,int nburn){
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

//General chain analysis:

///A routine for processing chain history to estimate autocorrelation of
///windowed chain segments
///
///For some feature function \f$f({\bf x})\f$ over the state space, we will
///compute the correlation length \f$\rho\f$ over various segments 
///\f$\mathcal{S}\f$of the chain history data. Given some \f$f\f$, 
///the correltation length is defined as:
///\f[
/// \rho =  1 + 2*\sum_k^{n_{\mathrm{lag}}} \rho(\tau_i)
///\f]
///where \f$ \rho(\tau_k) \f$ is the autocorrelation with lag \f$\tau_k\f$
///computed by
///\f[
//    \rho(\tau) = \frac{\sum_{i\in\mathcal{S}}( f(x_i)-\bar f_{\mathcal{S},\tau})( f(x_{i-\tau})-\bar f_{\mathcal{S},\tau})}{\sum_{i\in\mathcal{S}}( f(x_i)-\bar f_{\mathcal{S},\tau})^2}.
///\f]
///We define the average symmetrically over the nominal data segment \f$\mathcal{S}\f$ and the lagged segment, specifically
///\f[
  //   \bar f_{\mathcal{S},\tau}=\sum_{i\in\mathcal{S}}(f(x_i)+f(x_(i-\tau__)/2/N_{\mathcal{S}}
///\f]
//                    Sum[ ( f(i)-avg[iwin,lag] )*( f(i-lag)-avg[iwin,lag] ) ]
  // covar[iwin,lag] = ---------------------------------------------------------
  //                                           count
//                Sum[covar[iwin,lag] + (avg-avg[iwin,lag])^2,{iwin in set}]
//    \rho(\tau) \= ----------------------------------------------------------
//                  Sum[covar[iwin,0] + (avg-avg[iwin,0])^2,{iwin in set}]
///
///is basednecessary to perform autocorrelation analysis are computed block-wise over a set of window subdomains \f$\mathcal{W}_i\f$ in the step history.  
void chain::compute_autocovar_windows(bool (*feature)(const state &, double & value),vector< vector<double> >&covar,vector< vector<double> >&means,vector< vector <int> >&counts,vector<int>&outwindows,vector<int>&outlags,int width,int nevery,int burn_windows, bool loglag, int max_lag, double dlag){
  //inputs:
  //  feature         function which returns a feature value from a state.
  //  width           (>1)width in steps of each window
  //  nevery          (>=1) sampling rate in steps for the correlation analysis
  //  burn_windows    (>=1) number of initial windows to skip
  //  loglag          flag to use logarithmic lag spacing
  //  dlag            (>=1.01) factor for logarithmic spacing of lag windows
  //  max_lag         (<=burn_windows) maxiumum lag, in number of window widths
  //outputs:
  //  covar           [Nwin][Nlag] lag covariance covar[iwin][0] is variance
  //  denoms          [Nwin][Nlag] vector array of partial denomenators
  //  means           [Nwin][Nlag] Array of per-window feature means. 
  //  counts          [Nwin][Nlag] Number of samples included.
  //  outwindows      [Nwin+1] window start index labels for the output rows
  //  outlags         [Nlag] lag-size labels for the output columns 
  //
  // Then to compute rho(lag) for each window:
  //
  //                Sum[covar[iwin,lag] + (avg-avg[iwin,lag])^2,{iwin in set}]
  //    rho(lag) = ----------------------------------------------------------
  //                  Sum[covar[iwin,0] + (avg-avg[iwin,0])^2,{iwin in set}]
  //
  //                    Sum[ ( f(i)-avg[iwin,lag] )*( f(i-lag)-avg[iwin,lag] ) ]
  // covar[iwin,lag] = ---------------------------------------------------------
  //                                           count
  //
  //   avg[iwin,lag] = Sum[f(i)+f(i-lag)]/2/count
  //
  // where in the last lines, the sum and avg is over all the points in iwin,
  // and, for simplicity, we have left out weighting by count[iwin,lag] in the
  // expression for rho.  It is not obvious, but happens to be true, that the
  // particular form of mean given in the last line allows the covariances to
  // be simply combined as in the first line.
  //
  // Notes:
  //   -length of averages will be Nwin+max_lag, with first "window" segment
  //    mean at averages[max_lag], and the earlier ones corresponding to the
  //    buffer windows
  //   -Each window variance is w.r.t its avg in averages[max_lag+iwin] 
  //   -if (loglag==true) lags are approximately logarithmically spaced 
  //   -for meaningful results need max_lag>=burn_windows
  //   -last entry for outwindows be next value, for use in sums
  //   -windows are aligned to end at latest step
  //   -everything will be done in units of nevery
  //   -(*feature) should generally return true; if false then the feature is
  //    understood to apply only to a subdomain of the state space and such
  //    points are not included in in nums or denoms
  //   -this is why denoms may depend on lag
  //   -it will probably be necessary to replace these "feature" functions with
  //    some kind of class objects, probably to be defined in states.hh
  
  if(width<=1)width=2;
  if(nevery<1)nevery=1;
  int swidth=int(width/nevery);
  width=swidth*nevery;
  if(burn_windows<1)burn_windows=1;
  if(max_lag==0 or max_lag>burn_windows)max_lag=burn_windows;
  if(dlag<1.01)dlag=1.01;

  //determine output structure
  int ncount=getStep();
  int Nwin=ncount/width - burn_windows;
  if(Nwin<0)Nwin=0;
  int Nlag;
  //if(loglag){
    //logarithmic
    //Nlag=log(max_lag*swidth)/log(dlag);
  //} else {
    //linear
    //Nlag=swidth*burn_windows-1;
  //}
  //if(Nlag<0)Nlag=0;
  
  //set up output grid
  outwindows.resize(Nwin+1);
  int istart=getStep()-Nwin*width;
  //cout<<"istart,width,swidth:"<<istart<<" "<<width<<" "<<swidth<<endl;
  for(int i=0;i<=Nwin;i++)outwindows[i]=istart+i*width;
  if(loglag){
    //logarithmic
    outlags.resize(0);
    outlags.push_back(0);
    double fac=1;
    int idx=1;
    while(idx<max_lag*swidth){
      outlags.push_back(nevery*idx);
      //cout<<"lag="<<nevery*idx<<endl;
      int lastidx=idx;
      while(lastidx==idx){
	fac*=dlag;
	idx=int(fac);
      }
    }
    Nlag=outlags.size();
  } else {
    //linear
    Nlag=swidth*burn_windows+1;
    outlags.resize(Nlag);
    for(int i=0;i<Nlag;i++)outlags[i]=nevery*i;
  }
  covar.assign(Nwin,vector<double>(Nlag,0));
  means.assign(Nwin,vector<double>(Nlag,0));
  counts.assign(Nwin,vector<int>(Nlag,0));
  //cout<<"ncount,Nwin,Bwin,Nlag:"<<ncount<<" "<<Nwin<<" "<<burn_windows<<" "<<Nlag<<endl;

  //populate output vectors
  for(int k=0;k<Nwin;k++){
    //cout<<"k="<<k<<endl;
    double xsum=0;
    double xxsum=0;
    //We hold on to the zero-lag results
    vector<double>f(swidth,0);
    vector<bool>ok(swidth,false);
    for(int i=0;i<swidth;i++){
      int idx=outwindows[k]+i*nevery;
      //cout<<"idx="<<idx<<endl;
      //state s=getState(idx);
      //double fi;
      //cout<<"state="<<s.show()<<endl;      
      ok[i]=(*feature)(getState(idx),f[i]);
      //bool okay;
      //okay=(*feature)(s,fi);
      //cout<<"okay="<<okay<<endl;
      //ok[i]=okay;
      //cout<<"fi="<<fi<<endl;
      //f[i]=fi;
      //cout<<"..."<<endl;
      if(ok[i]){
	xsum+=f[i];
	xxsum+=f[i]*f[i];
	counts[k][0]++;
      }
    }
    means[k][0]=xsum/counts[k][0];
    covar[k][0]=xxsum/counts[k][0]-means[k][0]*means[k][0];
    for(int j=1;j<Nlag;j++){
      int ilag=outlags[j];
      xsum=0;
      xxsum=0;
      for(int i=0;i<swidth;i++){
	double fi;
	int idx=outwindows[k]+i*nevery-ilag;
	//cout<<"idx="<<idx<<endl;
	//cout<<"state="<<getState(idx).show()<<endl;      
	bool oklag=(*feature)(getState(idx),fi);
	if(oklag and ok[i]){
	  xsum+=(fi+f[i]);
	  xxsum+=fi*f[i];
	  counts[k][j]++;
	}
      }
      means[k][j]=xsum/counts[k][j]/2; //this is the avg of mean and lagged-mean
      covar[k][j]=xxsum/counts[k][j]-means[k][j]*means[k][j];//this equals covar wrt above mean
    }
  }
  //cout<<"autocorr: Nwin,ncount,width,burn ="<<Nwin<<", "<<ncount<<", "<<width<<", "<<burn_windows<<endl;
  //for(int k=0;k<Nwin;k++){
  //  cout<<"window k=:"<<k<<endl;
  //  for(int j=1;j<Nlag;j++)cout<<" "<<j<<" "<<counts[k][j]<<" "<<means[k][j]<<" "<<covar[k][j]<<endl;
  //}
}

//Estimate effective number of samples for some feature of the state samples
void chain::compute_effective_samples(vector<bool (*)(const state &,double & value)>&features, double &effSampSize, int &best_nwin, int width,int nevery,int burn_windows, bool loglag, int max_lag, double dlag){
  //inputs:
  //  features        functions which return feature values from a state.
  //  width           (>1)width in steps of each window
  //  nevery          (>=1) sampling rate in steps for the correlation analysis
  //  burn_windows    (>=1) number of initial windows to skip
  //  dlag            (>=1.1) factor for logarithmic spacing of lag windows
  //  max_lag         (<=burn_windows) maxiumum lag, in number of window widths
  //outputs:
  //  effSampSize     the estimate for effective sample size     
  //  best_nwin       the number of windows which optimized effSampSize
  //
  //  This routine uses compute_autocovar_windows to compute autocorrelation
  //  lenghts for the state feature functions provided.  Then, downsampling
  //  by this length, computes the effective number of samples for each
  //  feature, taking the minimum of the set.  The routine does this repeatedly,
  //  considering a range of possible sample sizes, beginning at the end of the
  //  chain and working backward.  Generally, one expects to find an optimal
  //  effective length which maximizes the effective sample size as correlation
  //  lengths should be longer when earlier, more immature parts of the chain
  //  are included in the calculation.
  //
  // Notes:
  //   -the autocorrlength over best_nwin is ac_len=width*best_nwin/effSampSize
  //   -See also notes for compute_autocovar_windows
    
  
  //controls
  double oversmall_aclen_fac=3.0;

  //cout<<"Enter compute_effective_samples"<<endl;
  int nf=features.size();
  //cout<<"nf="<<nf<<endl;
  vector< vector< vector<double> > > covar(nf);
  vector< vector< vector<double> > > means(nf);
  vector< vector< vector< int > > >counts(nf);
  //vector<double>esses(nf);
  double essi;
  vector<double> esses(nf),best_esses(nf);
  vector<double> best_means(nf);
  vector<int>windows;
  vector<int>lags;
  for(int i=0;i<nf;i++)
    compute_autocovar_windows(features[i],covar[i],means[i],counts[i],windows,lags,width,nevery,burn_windows, loglag, max_lag, dlag);
  int Nwin=windows.size()-1;
  int Nlag=lags.size();
  //cout<<"ESS: Nwin="<<Nwin<<" Nlag="<<Nlag<<endl;

  int lmin=0;
  double ess_max=0;
  int nwin_max=0;
  //Loop over options for how much of the chain is used for the computation
  for(int nwin=1;nwin<=Nwin;nwin++){
    //Compute autocorr length
    //We compute an naive autocorrelation length
    // ac_len =  1 + 2*sum( corr[i] )
    double ess=1e100;
    vector<double> feature_means(nf);
    for(int ifeat=0;ifeat<nf;ifeat++){
      //Compute sample mean.
      double sum=0;
      for(int i=Nwin-nwin;i<Nwin;i++)sum+=means[ifeat][i][0];
      double mean=sum/nwin;
      int last_lag=0;
      double ac_len=1.0;
      double lastcorr=1;
      double dacl=0;
      for(int ilag=1;ilag<Nlag;ilag++){
	int lag=lags[ilag];
	//compute the correlation for each lag
	double num=0;
	double denom=0;
	for(int iwin=Nwin-nwin;iwin<Nwin;iwin++){
	  double dmean=mean-means[ifeat][iwin][ilag];
	  double dmean0=mean-means[ifeat][iwin][0];
	  double cov=covar[ifeat][iwin][ilag]+dmean*dmean;
	  double var=covar[ifeat][iwin][0]+dmean0*dmean0;
	  int c=counts[ifeat][iwin][ilag];
	  num   += cov*c;
	  denom += var*c;
	  //cout<<nwin<<" "<<ifeat<<" "<<ilag<<" "<<iwin<<": count,mean,dmean,num,den="<<c<<", "<<mean<<", "<<dmean<<", "<<num<<", "<<denom<<endl;
	}
	//cout<<"num,denom:"<<num<<" "<<denom<<endl;
	double corr=num/denom;
	if(lastcorr<0 and corr<0){
	  //keep only "initally positive sequence" (IPS)
	  ac_len-=dacl;
	  break;
	}
	lastcorr=corr;
	dacl=2.0*(lag-last_lag)*corr;
	ac_len+=dacl;
	//cout<<"nwin,lag,corr,acsum:"<<nwin<<" "<<lag<<" "<<corr<<" "<<ac_len<<endl;
	last_lag=lag;
      }
      //cout<<"baselen="<<nwin*width<<"  aclen="<<ac_len<<endl;
      //compute effective sample size estimate
      //esses[ifeat]=nwin*width/ac_len;
      essi=nwin*width/ac_len;
      //cout<<"nwin,lag,aclen,effss:"<<nwin<<" "<<last_lag<<" "<<ac_len<<" "<<essi<<endl;
      if(ac_len<nevery){
	//Ignore as spurious any
	cout<<"aclen<nevery!: "<<ac_len<<" < "<<nevery<<endl;
        //esses[ifeat]=0;
	//We don't really trust this result, and so instead set ess supposing aclen=nevery*oversmall_aclen_fac
	essi=nwin*width/oversmall_aclen_fac/nevery;
      }
      // if(esses[ifeat]<ess){
      //ess=esses[ifeat];
      if(essi<ess){
	ess=essi;
      }
      esses[ifeat]=essi;
      feature_means[ifeat]=mean;
      //cout<<"nwin,ifeat="<<nwin<<","<<ifeat<<":  -> "<<essi<<endl;
    }
    if(ess>ess_max){
      ess_max=ess;
      nwin_max=nwin;
      best_esses=esses;
      best_means=feature_means;
    }
  }
  best_nwin=nwin_max;
  effSampSize=ess_max;

  //dump info
  if(reporting){ 
    cout<<"len="<<nwin_max<<"*"<<width<<": ";
    for(auto ess : best_esses)cout<<" "<<ess;
    cout<<endl;
    cout<<"     means = ";
    for(auto val : best_means)cout<<" "<<val;
    cout<<endl;
  }
  //For testing, we dump everything and quit
  static int icount=0;
  icount++;
  //cout<<"COUNT="<<icount<<endl;
  if(false and not loglag){
    ofstream os("ess_test.dat");
    for(int iwin=Nwin-best_nwin-burn_windows;iwin<Nwin;iwin++){
      for(int i=0;i<int(width/nevery);i++){
	int idx=windows[iwin]+i*nevery;
	double fi;
	if((*features[0])(getState(idx),fi)){
	  os<<idx<<" "<<fi<<endl;
	}
      }
    }
    cout<<"aclen="<<best_nwin*width/ess_max<<endl;
    cout<<"range="<<best_nwin*width<<endl;
    cout<<"ess="<<ess_max<<endl;
    cout<<"burnfrac="<<burn_windows/(1.0*best_nwin+burn_windows)<<endl;
    cout<<"Quitting for test!"<<endl;
    //exit(0);
  }
};

//Report effective samples
//Compute effective samples for the vector of features provided.
//Report the best case effective sample size for the minimum over all features
//allowing the length of the late part of the chain under consideration to vary
//to optimize the ESS.
//Returns (ess,length)
pair<double,int> chain::report_effective_samples(vector< bool (*)(const state &,double & value) > & features,int width,int nevery){
  double ess;
  int nwin;
  int burn=2;
  vector<bool (*)(const state &,double & value)> onefeature(1);

  int i=1;

  //static int ic=0;
  //ic++;
  //int icstop=200;
  //if(ic>icstop)for(auto feature:features){
  //  onefeature[0]=feature;
  //  compute_effective_samples(onefeature, ess, nwin, width, nevery, burn, false);
  //  cout<<"Par "<<i<<": ess="<<ess<<"  useful chain length is: "<<width*nwin<<" autocorrlen="<<width*nwin/ess<<endl;
  //  i++;
  //}
  compute_effective_samples(features, ess, nwin, width, nevery, burn, true,0,1.1);
  if(reporting)
    cout<<"Over "<<features.size()<<" pars: ess="<<ess<<"  useful chain length is: "<<width*nwin<<" autocorrlen="<<width*nwin/ess<<endl;
  //if(ic>icstop)exit(0);

  return make_pair(ess,width*nwin);
}

//Report effective samples
//This is simplified interface producing effective sample estimates for each parameter
pair<double,int>  chain::report_effective_samples(int imax,int width, int every){
  //int width=40000;
  while(width<getStep()*0.05)width*=2;
  //int every=100;
  if(imax<0)imax=dim;
  if(imax>dim)imax=dim;
  vector<bool (*)(const state &,double & value)> features;
  //We have to make an explicit function for each parameter, so there seems
  //to be no way to make this pretty with a loop
  if(imax>0){
    auto feature = [](const state &s,double &val) { val=s.get_param(0);return true;};
    features.push_back(feature);
  }
  if(imax>1){
    auto feature = [](const state &s,double &val) { val=s.get_param(1);return true;};
    features.push_back(feature);
  }
  if(imax>2){
    auto feature = [](const state &s,double &val) { val=s.get_param(2);return true;};
    features.push_back(feature);
  }
  if(imax>3){
    auto feature = [](const state &s,double &val) { val=s.get_param(3);return true;};
    features.push_back(feature);
  }
  if(imax>4){
    auto feature = [](const state &s,double &val) { val=s.get_param(4);return true;};
    features.push_back(feature);
  }
  if(imax>5){
    auto feature = [](const state &s,double &val) { val=s.get_param(5);return true;};
    features.push_back(feature);
  }
  if(imax>6){
    auto feature = [](const state &s,double &val) { val=s.get_param(6);return true;};
    features.push_back(feature);
  }
  if(imax>7){
    auto feature = [](const state &s,double &val) { val=s.get_param(7);return true;};
    features.push_back(feature);
  }
  if(imax>8){
    auto feature = [](const state &s,double &val) { val=s.get_param(8);return true;};
    features.push_back(feature);
  }
  if(imax>9){
    auto feature = [](const state &s,double &val) { val=s.get_param(9);return true;};
    features.push_back(feature);
  }
  if(imax>10){
    auto feature = [](const state &s,double &val) { val=s.get_param(10);return true;};
    features.push_back(feature);
  }
  if(imax>11){
    auto feature = [](const state &s,double &val) { val=s.get_param(11);return true;};
    features.push_back(feature);
  }
  if(imax>12){
    auto feature = [](const state &s,double &val) { val=s.get_param(12);return true;};
    features.push_back(feature);
  }
  if(imax>13){
    auto feature = [](const state &s,double &val) { val=s.get_param(13);return true;};
    features.push_back(feature);
  }
  if(imax>14){
    auto feature = [](const state &s,double &val) { val=s.get_param(14);return true;};
    features.push_back(feature);
  }
  if(imax>15){
    auto feature = [](const state &s,double &val) { val=s.get_param(15);return true;};
    features.push_back(feature);
  }
  if(imax>16){
    auto feature = [](const state &s,double &val) { val=s.get_param(16);return true;};
    features.push_back(feature);
  }
  if(imax>17){
    auto feature = [](const state &s,double &val) { val=s.get_param(17);return true;};
    features.push_back(feature);
  }
  if(imax>18){
    auto feature = [](const state &s,double &val) { val=s.get_param(18);return true;};
    features.push_back(feature);
  }
  if(imax>19){
    auto feature = [](const state &s,double &val) { val=s.get_param(19);return true;};
    features.push_back(feature);
  }
  if(imax>20){
    cout<<"chain::report_effective_samples(): Currently only supports the first 20 params, by default."<<endl;
  }
  
  return report_effective_samples(features,width,every);
};

// A markov (or non-Markovian) chain based on some variant of the Metropolis-Hastings algorithm
// May add "burn-in" distinction later.
MH_chain::MH_chain(probability_function * log_likelihood, const sampleable_probability_function *log_prior,double minPrior,int add_every_N):
  llikelihood(log_likelihood),lprior(log_prior),minPrior(minPrior),add_every_N(add_every_N){
  Nsize=0;Nhist=0;Nzero=0;invtemp=1;Ntries=1;Naccept=1;last_type=-1;
  dim=log_prior->getDim();
  default_prop_set=false;
  //cout<<"creating chain at this="<<this->show()<<" with lprior="<<lprior->show()<<endl;//debug
  Ninit=0;
};

void MH_chain::checkpoint(string path){
  chain::checkpoint(path);
  ostringstream ss;
  ss<<path<<"chain"<<id<<"-cp/";
  ss<<"MHchain.cp";
  ofstream os;
  openWrite(os,ss.str());
  //save basic data 
  //The philosopy is that we don't need to save anything set by setup...
  writeInt(os, Ntries);
  writeInt(os, Naccept);
  writeInt(os, last_type);
  //add_every_N;
  writeIntVector(os,types);
  writeInt(os,states.size());for(int i=0;i<states.size();i++)writeString(os,states[i].save_string());
  writeDoubleVector(os,lposts);
  writeDoubleVector(os,llikes);
  //cout<<"current_state before:"<<current_state.show()<<endl;
  writeString(os,current_state.save_string());
  writeDouble(os,current_lpost);
  writeDouble(os,current_llike);
  writeDoubleVector(os,acceptance_ratio);
  writeDoubleVector(os,invtemps);
  //probability_function *llikelihood;
  //const sampleable_probability_function *lprior;
  //double minPrior;
  //proposal_distribution *default_prop;
  //bool default_prop_set;
  writeInt(os,Nhist);
  writeInt(os,Nzero);
  writeDouble(os,invtemp);
};

void MH_chain::restart(string path){
  state protostate=lprior->drawSample(*rng);//We draw a sample as seed state *before* reinstating the rng;  if the current rng were ever needed again this would fail
  chain::restart(path);
  ostringstream ss;
  ss<<path<<"chain"<<id<<"-cp/";
  ss<<"MHchain.cp";
  ifstream os; 
  openRead(os,ss.str());
  //save basic data 
  //The philosopy is that we don't need to save anything set by setup...
  readInt(os, Ntries);
  readInt(os, Naccept);
  readInt(os, last_type);
  //add_every_N;
  readIntVector(os,types);
  //The following block is to restore the states vector
  //We build the states from an example (protostate) which includes the right stateSpace (which isn't stored) already set up...
  int n;readInt(os,n);states.resize(n,protostate);for(int i=0;i<n;i++){string s;readString(os,s);states[i].restore_string(s);};
  readDoubleVector(os,lposts);
  readDoubleVector(os,llikes);
  //cout<<"current_state before restore:"<<current_state.show()<<endl;
  string s;
  readString(os,s);
  current_state=protostate;
  current_state.restore_string(s);
  //cout<<"current_state after restore:"<<current_state.show()<<endl;
  readDouble(os,current_lpost);
  readDouble(os,current_llike);
  readDoubleVector(os,acceptance_ratio);
  readDoubleVector(os,invtemps);
  //probability_function *llikelihood;
  //const sampleable_probability_function *lprior;
  //double minPrior;
  //proposal_distribution *default_prop;
  //bool default_prop_set;
  readInt(os,Nhist);
  readInt(os,Nzero);
  readDouble(os,invtemp);
};

void MH_chain::reserve(int nmore){//If you know how long you are going to run, it can be more efficient to reserve up front, rather than resizing ever step
    states.reserve(Nsize+nmore);
    lposts.reserve(Nsize+nmore);
    llikes.reserve(Nsize+nmore);
    acceptance_ratio.reserve(Nsize+nmore);
    invtemps.reserve(Nsize+nmore);
  };

void MH_chain::initialize(uint n, string initialization_file){
//Initialize chain from an file in the format of a chain output file
//Generally we presume that the input file is consistent with a unit-temperature
//chain.  If this chain's temp isnot one, then we mix in prior samples as well
//so that the initial distribution is not too narrow.  Clearly at unit-temp
//we want all samples from the file, and at zero inverse temp we want all 
//samples from the prior.  We linearly interpolate the intervening mix.
//We assume the order is not interesting. We read first from the file, then
//supplement from the prior.

  if(Nhist>0){
    cout<<"MH_chain::initialize: Cannot re-initialize."<<endl;
    exit(1);
  }
  Ninit=n;
  int n_file=n*invtemp;
  
  ifstream inifile(initialization_file);
  string line;
  const stateSpace *space=lprior->get_space();
  vector<double> pars(space->size());
  //This is a rather clunky/slow way to count the relevant lines
  int count=0;
  while(getline(inifile,line)){
    if(line.compare(0,1,"#")==0)continue;
    count++;
  }
  int nlines=count;
  int startline=count*0.75;//probably make this fraction a parameter;
  inifile.clear();
  inifile.seekg(0, ios::beg);    
  count=0;
  while(getline(inifile,line)){//skip first portion of file
    if(line.compare(0,1,"#")==0)continue;
    count++;
    if(count>=startline-1)break;
  }
  int icnt=0;
  cout<<"Reading initial samples from file: "<<initialization_file<<endl;
  while(getline(inifile,line)){
    istringstream line_ss(line);
    int num;
    double llike,lpost,acc;
    string typ;
    //cout<<line<<endl;
    if(line.compare(0,1,"#")==0)continue;
    count++;
    line_ss>>num;
    if(num<0)continue;
    if(not rng->Next()>(nlines-count)/(double)(n-1-icnt))continue;       // draw with appropriate probability from the remaining lines
    //if we make it this far then we draw the line as a state
    line_ss>>lpost>>llike>>acc>>typ;
    for(auto &par:pars)line_ss>>par;
    state s=state(space,pars);
    double slike=-1e100;
    double sprior=-1e100;
    //cout<<"state="<<s.show()<<endl;
    if(not s.invalid()){
      //cout<<"=valid"<<endl;
      if( (sprior=lprior->evaluate_log(s))>-1e9){
	//Note: we need not trust the llike/lpost from the file
	if( (slike=llikelihood->evaluate_log(s))>-1e9){
	  //cout<<"ADDED state llike="<<slike<<endl;
	  Nhist=0;
	  add_state(s,slike,sprior+slike);
	  icnt++;
	}else{
	  //cout<<"rejected state llike="<<slike<<endl;
	}
      }else{
	//cout<<"rejected state lprior="<<sprior<<endl;
      }
    }else{
      //cout<<"rejected state as invalid"<<endl;
    }
  }
  //cout<<"Ninit="<<Ninit<<endl;
  //Ninit=icnt;//Should be the same unless file was short.
  //Now (if needed) add points from the prior
  for(uint i=0;i<Ninit-icnt;i++){
    state s=lprior->drawSample(*rng);
    int icnt=0,icntmax=1000;
    //while(s.invalid()&&icnt<2*icntmax){
    double slike;
    while(s.invalid() or (slike=llikelihood->evaluate_log(s))<-1e100){
      icnt++;
      if(icnt>=icntmax)
	cout<<"MH_chain::initialize: Having trouble drawing a valid state.  Latest state:"<<s.show()<<"...was invalid in space:"<<s.getSpace()->show()<<endl;
      s=lprior->drawSample(*rng);
    }
    //cout <<"starting with Nsize="<<Nsize<<endl;//debug
    Nhist=0;//As long as Nhist remains zero we will add the state regardless of add_every_N
    add_state(s);
  }
  Nhist=0;
}

void MH_chain::initialize(uint n){
  if(Nhist>0){
    cout<<"MH_chain::initialize: Cannot re-initialize."<<endl;
    exit(1);
  }
  Ninit=n;
  for(uint i=0;i<n;i++){
    state s=lprior->drawSample(*rng);
    int icnt=0,icntmax=1000;
    //while(s.invalid()&&icnt<2*icntmax){
    double slike;
    while(s.invalid() or (slike=llikelihood->evaluate_log(s))<-1e100){
      //cout<<"invalid state: like="<<slike<<" pars:"<<s.get_string()<<endl;
      icnt++;
      if(icnt>=icntmax)
	cout<<"MH_chain::initialize: Having trouble drawing a valid state.  Latest state:"<<s.show()<<"...was invalid in space:"<<s.getSpace()->show()<<endl;
      s=lprior->drawSample(*rng);
    }
    //cout <<"starting with Nsize="<<Nsize<<endl;//debug
    Nhist=0;//As long as Nhist remains zero we will add the state regardless of add_every_N
    add_state(s);
  }
  Nhist=0;//Don't count these.
};

void MH_chain::reboot(){
  Nzero=Nhist;
  Nsize=0;Nhist=0;Ntries=1;Naccept=1;last_type=-1;
  cout<<"Rebooting chain (id="<<id<<")"<<endl;
  Nfrozen=-1;
  types.clear();
  states.clear();
  lposts.clear();
  llikes.clear();
  acceptance_ratio.clear();
  invtemps.clear(); 
  initialize(Ninit);//not set up to initialize from chain file in this case.
};

//If we should decide we no longer need to hold on to part of the chain then:
//This is an initial draft implementation not yet tested or enabled
/*
void MH_chain::forget(int imin){
  int icut=get_state_idx(imin);
  if(icut<=0)return;
  int ncut=icut;
  Nzero=imin;
  //Nsize=0;
  //Nhist=0;
  //Ntries=1;
  //Naccept=1;
  //last_type=-1;
  cout<<"Forgetting early part of chain (id="<<id<<")"<<endl;
  types.erase(types.begin(),types.begin()+ncut);
  states.erase(states.begin(),states.begin()+ncut);
  lposts.erase(lposts.begin(),lposts.begin()+ncut);
  llikes.erase(llikes.begin(),llikes.begin()+ncut);
  acceptance_ratio.erase(acceptance_ratio.begin(),acceptance_ratio.begin()+ncut);
  invtemps.erase(invtemps.begin(),invtemps.begin()+ncut);
  Ninit=0;
};
*/

void MH_chain::add_state(state newstate,double log_like,double log_post){
  if(newstate.invalid()){
    cout<<"MH_chain::add_state: Adding an invalid state!  State is:"<<newstate.get_string()<<endl;
    exit(1);
  }
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
    newstate.enforce();
    double newlike,newlpost,newlprior=lprior->evaluate_log(newstate);
    //cout<<"MH_chain::step: newlprior="<<newlprior<<endl;
    if((!newstate.invalid())&&((!current_lpost>-1e200&&newlprior>-1e200)||newlprior-oldlprior>minPrior)){//Avoid spurious likelihood calls where prior effectively vanishes. But try anyway if current_lpost is huge. 
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
    //if(!(current_lpost>-1e200))  cout<<"   log_hastings_ratio="<<log_hastings_ratio<<endl;
    bool accept=true;
    if(newstate.invalid())accept=false;
    //cout<<Nhist<<"("<<invtemp<<"): ("<<newlike<<","<<newlpost<<")vs.("<<oldlpost<<")->"<<log_hastings_ratio<<endl;//debug
    if(accept and log_hastings_ratio<0){
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
      prop.accept();
      add_state(newstate,newlike,newlpost);
      //chainsteps.push_back(Nhist-Ninit);
    }
    else {
      prop.reject();
      //cout<<"        rejecting"<<endl;//debug
      add_state(current_state,current_llike,current_lpost);
      //cout<<"Nhist="<<Nhist<<"        rejected"<<endl;
    }
    // cout<<"incremented NHist="<<Nhist<<endl;
};

double MH_chain::expectation(double (*test_func)(state s),int Nburn){
  //should probably add a check that Nburn>Nzero
  double sum=0;
  for(int i=Nburn;i<Nhist;i+=add_every_N)sum+=test_func(states[get_state_idx(i)]);
  return sum/int((Nhist-Nburn)/add_every_N);
};

double MH_chain::variance(double (*test_func)(state s),double fmean,int Nburn){
  //should probably add a check that Nburn>Nzero
    double sum=0;
    for(int i=Nburn;i<Nhist;i+=add_every_N){
      double diff=test_func(states[get_state_idx(i)])-fmean;
      sum+=diff*diff;
    }
    return sum/int((Nhist-Nburn)/add_every_N);
};

int MH_chain::get_state_idx(int i){
  //should probably add a check that Nburn>Nzero
  if(i<0||i>=Nhist)i=Nhist-1;
  int irequest=Ninit+(i-Nzero)/add_every_N;
  if(irequest<0){
    cout<<"MH_chain::get_state_idx: Requested data for i("<<i<<")="<<irequest<<" but data before i=0 is not available!!!"<<endl;
    //irequest=0;
  }
  return irequest;
}

int MH_chain::getStep(){
  return Nhist;
};

state MH_chain::getState(int elem,bool raw_indexing){//defaults (-1,false)  
  //cout<<"MH_chain::getState()"<<endl;
  if(elem<0||(raw_indexing&&elem>=Nsize)||(!raw_indexing&&elem>=Nhist)){
    if(chain_verbose)cout<<"returning current state: elem="<<elem<<" Nhist="<<Nhist<<endl;
    return current_state;//Current state by default
  } else if (raw_indexing) { 
    //cout<<"returning states["<<elem<<"]"<<endl;
    return states[elem];
  } else {
    if(chain_verbose)cout<<"returning states["<<get_state_idx(elem)<<"]"<<endl;
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
    //for(int i=0;i<np-1;i++)os<<"param("<<i<<") ";
    //os<<"param("<<np-1<<")"<<endl;
    const stateSpace *sp=states[0].getSpace();
    for(int i=0;i<np;i++)os<<sp->get_name(i)<<" ";
    os<<endl;
    if(Nburn+Ninit<0)Nburn=-Ninit;
    for(int i=Nburn;i<Nhist;i+=ievery){
      int idx=Ninit+i;
      if(i>=0)idx=get_state_idx(i);
      //cout<<"i="<<i<<" "<<idx<<"=idx<states.size()="<<states.size()<<"?"<<endl;//debug
      os<<i<<" "<<lposts[idx]<<" "<<llikes[idx]<<" "<<acceptance_ratio[idx]<<" "<<types[idx]<<": ";
      vector<double>pars=states[idx].get_params_vector();
      //cout<<"state:"<<states[i].show()<<endl;
      for(int j=0;j<np-1;j++)os<<pars[j]<<" ";
      os<<pars[np-1];
      os<<" "<<invtemp;
      os<<endl;
    }
};
  
string MH_chain::show(bool verbose){
    ostringstream s;
    s<<"MH_chain(id="<<id<<",every="<<add_every_N<<",invtemp="<<invtemp<<",size="<<Nsize<<",N="<<Nhist<<")\n";
    if(verbose)s<<current_state.getSpace()->show()<<endl;
    return s.str();
};

string MH_chain::status(){
    ostringstream s;
    //s<<"chain(id="<<id<<", N="<<Nsize<<", T="<<1.0/invtemp<<"):"<<lposts[Nsize-1]<<", "<<llikes[Nsize-1]<<" : "<<this->getState().get_string();
    s<<"chain(id="<<id<<", N="<<Nsize<<", T="<<1.0/invtemp<<"):"<<current_lpost<<", "<<current_llike<<" : "<<this->getState().get_string();
    return s.str();
};

string MH_chain::report_prop(int style){
  if(default_prop_set)
    return default_prop->report(style);
  else return "NoProp";
};




// A parallel tempering set of markov (or non-Markovian) chain
// May add "burn-in" distinction later.

parallel_tempering_chains::parallel_tempering_chains(int Ntemps,double Tmax,double swap_rate,int add_every_N,bool do_evid_in,bool verbose_evid,double dpriormin):Ntemps(Ntemps),swap_rate(swap_rate),add_every_N(add_every_N),do_evid(do_evid_in),verbose_evid(verbose_evid),dpriormin(dpriormin){
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
    evidence_count=0;
    evidence_records_dim=0;
    best_evidence_stderr=1e100;
    use_mpi=false;
    nproc=1;
    myproc=0;
#ifdef USE_MPI
    use_mpi=true;
    MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    if(myproc==0)cout<<"constructing ptc: nproc="<<nproc<<endl;
    if(do_evid and nproc>1){
      do_evid=false;
      if(myproc==0)cout<<"parallel_tempering_chains(constructor): Warning evidence computation not implemented for MPI."<<endl;
    }
    if(nproc==0)use_mpi=false;
#endif
    istatsbin=-1;
};

void parallel_tempering_chains::checkpoint(string path){
  chain::checkpoint(path);
  if(myproc==0){ //MPI only the head-proc writes the base-chain data.  
    ostringstream ss;
    ss<<path<<"chain"<<id<<"-cp/";
    ss<<"PTchain.cp";
    ofstream os;
    openWrite(os,ss.str());
    writeIntVector(os, directions);
    writeIntVector(os, ups);
    writeIntVector(os, downs);
    writeIntVector(os, instances);
    writeIntVector(os, instance_starts);
    writeIntVector(os, swap_accept_count);
    writeIntVector(os, swap_count);
    writeDoubleVector(os, temps);
    writeDoubleVector(os, log_eratio_up);
    writeDoubleVector(os, log_eratio_down);
    writeDoubleVector(os, tryrate);
    writeDoubleVector(os, swaprate);
    writeDoubleVector(os, up_frac);
    int n=total_evidence_records.size();
    writeInt(os,n);
    for(int i=0;i<n;i++)writeDoubleVector(os, total_evidence_records[i]);
    writeInt(os, evidence_count);
    writeDouble(os, best_evidence_stderr);
  }
  //MPI Each proc writes its sub-chain's data in separate files
  cout<<"proc "<<myproc<<": About to dump subchain checkpoint data."<<endl;
  MPI_Barrier(MPI_COMM_WORLD);
  for(int i:mychains)chains[i].checkpoint(path);
  cout<<"proc "<<myproc<<": Dumped subchain checkpoint data."<<endl;
};

void parallel_tempering_chains::restart(string path){
  chain::restart(path);
  ostringstream ss;
  ss<<path<<"chain"<<id<<"-cp/";
  ss<<"PTchain.cp";
  ifstream os;
  //MPI Each chain will need to read the common info.  In principle, should be able to have restart with a different number of processors than before.  No info about MPI should need to be saved.
  //MPI Need to know by here how chains are distributed across MPI processes
  //preMPI:for(int i=0;i<Ntemps;i++)chains[i].restart(path);
  for(int i:mychains)chains[i].restart(path);
  //MPI Is there a better easy way to read/broadcast? All procs read same file here
  openRead(os,ss.str());
  readIntVector(os, directions);
  readIntVector(os, ups);
  readIntVector(os, downs);
  readIntVector(os, instances);
  readIntVector(os, instance_starts);
  readIntVector(os, swap_accept_count);
  readIntVector(os, swap_count);
  readDoubleVector(os, temps);
  readDoubleVector(os, log_eratio_up);
  readDoubleVector(os, log_eratio_down);
  readDoubleVector(os, tryrate);
  readDoubleVector(os, swaprate);
  readDoubleVector(os, up_frac);
  int n;
  readInt(os,n);
  evidence_records_dim=n;
  total_evidence_records.resize(n);
  for(int i=0;i<n;i++)readDoubleVector(os, total_evidence_records[i]);
  readInt(os, evidence_count);
  readDouble(os, best_evidence_stderr);
};

void parallel_tempering_chains::initialize( probability_function *log_likelihood, const sampleable_probability_function *log_prior,int n,string initialization_file){
  Ninit=n;
  dim=log_prior->getDim();

  //MPI This is the starting place where MPI distribution is determined
  //Need a global indexing of chains, and maps to and from the local indices.
  mychains.clear();//probably don't need this
  is_my_chain.clear();//probably don't need this
  interproc_unpack_index.clear();//probably don't need this
#ifdef USE_MPI
  //use_mpi=true;
  //MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
  //MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  is_my_chain.resize(Ntemps,false);
  //We distribute the replicas to processes in cycles.  There is probably no advantage to near-temperature locality and if there is a difference
  //in typical likehood comp times it is likely temperature dependent.
  //Note that it makes no sense to use more procs*threads than the number of temps.
  for(int i=myproc;i<Ntemps;i+=nproc){
    mychains.push_back(i);
    is_my_chain[i]=true;
  }
  interproc_stride=(Ntemps-1)/nproc+1;
  for(int i=0;i<Ntemps;i++){
    int p=i%nproc;
    interproc_unpack_index.push_back(p*interproc_stride+i/nproc);
    if(p==myproc){
      cout<<"proc "<<myproc<<": index="<<p*interproc_stride+i/nproc<<"  i="<<i<<endl;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);  
  if(myproc!=0)this->reporting=false;//turn off output except for head node
#else
  //use_mpi=false;
  //myproc=0;
  //nproc=1;
  is_my_chain.resize(Ntemps,true);
  mychains.resize(Ntemps);
  for(int i=0;i<Ntemps;i++)mychains[i]=i;
  interproc_stride=1;
#endif
  
  //MPI For thread-safe set-up, to avoid some issues are with the global indexing of chains, and the initialization of chain RNGs, we instantiate all chains, though most will not be used.
  for(int i=0;i<Ntemps;i++){
    MH_chain c(log_likelihood, log_prior,dpriormin,add_every_N);
    chains.push_back(c);
  }
  //#pragma omp parallel for schedule (guided, 1)  ///try big chunks first, then specialize
#pragma omp parallel for schedule (dynamic, 1) ///take one pass/thread at a time until done.
  for(int iloc=0;iloc<mychains.size();iloc++){
    int i=mychains[iloc];
    //MPI loops like this need to be only over the local chains
    //MPI probably temps[i] is global, and need local-to-global index map
    //MPI temps[global_temp_index[i]]
    //MPI Ntemps in these loops should be NLocalChains or such
    int ith=omp_get_thread_num();
    ostringstream oss;oss<<"PTchain: initializing chain "<<i<<" on thread "<<ith<<endl;
#pragma opm critical    
    cout<<oss.str()<<endl;
    chains[i].invtemp=1/temps[i];
    if(initialization_file!="")chains[i].initialize(n,initialization_file);
    else chains[i].initialize(n);
  }
  for(int i=0;i<Ntemps;i++){
    //chains[i].invtemp=1/temps[i];
    //MPI instances[] instance_starts[] directions[] ups[] downs[] all global?
    instances[i]=i;
    //MPI If rebooting is implemented with MPI then we may need to set instance_starts in a meaningful way for non-local replicas
    instance_starts[i]=chains[i].size();
    //cout<<"initialized chain "<<i<<" at "<<&chains[i]<<endl;
    //ichains[i]=i;
    if(i==0)directions[i]=-1;
    else if(i==Ntemps-1)directions[i]=1;
    else directions[i]=0;
    ups[i]=0;
    downs[i]=0;
  };
  //Note istatsbin sets pace for various analysis beyond just evidence
  istatsbin=10000*(add_every_N/10+1);//make this user adjustable?
  if(myproc==0)cout<<"Stats bin size = "<<istatsbin<<endl;
  
  //MPI Calls to c0 should be avoided if possible. Where any instance of the chain will suffice, can be chains[0], the first local chain.
  //MPI FIXME?
  Nsize=c0().size(); 
};

void parallel_tempering_chains::set_proposal(proposal_distribution &proposal){
  //cout<<"("<<this<<")ptchain::set_proposal(proposal="<<&proposal<<")"<<endl;
  for(int i=0;i<Ntemps;i++){
    //MPI: strictly loop over local chains
    //MPI: props[i] should probably be local as well? Where is this vector defined
    props[i]=proposal.clone();
    //cout<<"cloning proposal: props["<<i<<"]="<<props[i]<<endl;
    //cout<<"                    chain=("<<&chains[i]<<")"<<chains[i].show()<<endl;
    if(props[i]->support_mixing()){
      props[i]->set_chain(this);
      if(myproc==0)cout<<"Supporting chain mixing in proposal distribution."<<endl;
    }
    else
      props[i]->set_chain(&chains[i]); //*** This seems to set the last chain to all props...***
    //Other possible actions, such as a burn-in chain, or a combination of chains would be possible here.
  }
};

bool parallel_tempering_chains::outputAllowed()const{
  //only allow output if base proc
  return myproc==0;
};

void parallel_tempering_chains::step(){
  int iswaps[maxswapsperstep];
  double x;

  //diagnostics and steering: set up
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
  //MPI all global loop
  for(int i=0;i<maxswapsperstep;i++){
    iswaps[i]=-2;
    double x=get_uniform();
    if(Ntemps>1&&x<(Ntemps-1)*swap_rate/maxswapsperstep){//do swap 
      x=get_uniform();
      iswaps[i]=int(x*(Ntemps-1));
      //cout<<"trying "<<iswaps[i]<<endl;
      for(int j=0;j<i;j++)//Don't allow adjacent swaps in same step;
	if(iswaps[j]==iswaps[i]||iswaps[j]+1==iswaps[i])iswaps[i]=-2;
    }
  }
  //for(int j=0;j<maxswapsperstep;j++)cout<<"\t"<<iswaps[j];
  //cout<<endl;
  //Now perform the swap trials:
  //MPI This loop may be one of the most subtle to parallelize
  //MPI We certainly want most to go through at a global level, but it includes
  //MPI references to specific chains.  A likely way to handle this is to first
  //MPI these actions globally, then execute them locally. This may mean:
  //MPI 1. Globally communicate essential chain info
  //MPI 2. Plan the actions
  //MPI 3. Execute the actions
  //MPI 4. Globally update info (if needed)
  //MPI: The next declarations establish global data in preparation for the loop
  vector<state>states=gather_states();
  vector<double>loglikes=gather_llikes();
  vector<double>invtemps=gather_invtemps();
  for(int j=0;j<maxswapsperstep;j++){
    if(iswaps[j]>=0){
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
      //cout<<"Trying swap between chains "<<i<<" and "<<i+1<<endl;
      //cout<<"  logLikes:"<<chains[i].getLogLike()<<" "<<chains[i+1].getLogLike()<<endl;
      //cout<<"  invtemps:"<<chains[i].invtemp<<" "<<chains[i+1].invtemp<<endl;
      //MPI: probably bcast global info about all chains logLike values at start
      //MPI: global allTempLogLikes
      //MPI: global allInvTemps
      //preMPI double lla=chains[i].getLogLike();if(!(lla>-1e200))lla=-1e200;
      double lla=loglikes[i];if(!(lla>-1e200))lla=-1e200;
      //preMPI double llb=chains[i+1].getLogLike();if(!(llb>-1e200))llb=-1e200;
      double llb=loglikes[i+1];if(!(llb>-1e200))llb=-1e200;
      //preMPI double log_hastings_ratio=-(chains[i+1].invtemp-chains[i].invtemp)*(llb-lla);//Follows from (21) of LittenbergEA09.
      double log_hastings_ratio=-(invtemps[i+1]-invtemps[i])*(llb-lla);//Follows from (21) of LittenbergEA09.
      if(log_hastings_ratio<0){
	x=get_uniform(); //pick a number
	accept=(log(x)<log_hastings_ratio);
      }
      //cout<<"lla, llb: "<<lla<<", "<<llb<<endl;
      //cout<<i<<" "<<log_hastings_ratio<<" -> "<<(accept?"Swap":"----")<<endl;
      if(accept){
	//we swap states and leave temp fixed.
	//MPI: start with global allTempStates
	//state sA=chains[i].getState();
	//MPI: start with global allTempStates
	//double llikeA=chains[i].getLogLike();
	//double lpostA=chains[i].getLogPost();  //This doesn't work since posterior depends on temperature.
	//state sB=chains[i+1].getState();
	//double llikeB=chains[i+1].getLogLike();
	//double llikeB=loglikes(i+1);
	//chains[i+1].add_state(sA,llikeA,lpostA);
	//MPI only perform these adds on local chains
	//MPI This should work fine as long as there are no double exchanges in
	//MPI one pass through the loop.  That should already be assured, since
	//MPI we don't allow adjacent swaps in the same step...
	//preMPI chains[i+1].add_state(sA,llikeA);  //Allow the chain to compute posterior itself.
	//preMPI chains[i].add_state(sB,llikeB);
	if(is_my_chain[i+1])
	  chains[i+1].add_state(states[i],loglikes[i]);//Allow the chain to compute posterior itself.
	if(is_my_chain[i])
	  chains[i].add_state(states[i+1],loglikes[i+1]);
	swap(states[i],states[i+1]);
	swap(loglikes[i],loglikes[i+1]);
	//swap(logposts[i],logposts[i+1]);
	//MPI instances etc should be global
	swap(directions[i],directions[i+1]);
	swap(instances[i],instances[i+1]);
	if(i==0)directions[i]=1;
	if(i+1==Ntemps-1)directions[i+1]=-1;
	swap_accept_count[i]++;
	swapcounts[iswaps[j]]++;
	if(do_evolve_temps){
	  if(0){
	    pry_temps(i,evolve_temp_rate);
	    invtemps=gather_invtemps();
	  } else {	  
	    //MPI Eventually we move this out of the loop.  We then need instead to add to a list of accepted swaps for collective pry_temps
	    //MPI pry_temps will be called globally
	    //MPI as currently implemented we do pry_temps individually when each swap is performed
	    //MPI This may be inefficient since each time requires global temperature information and each time that information must
	    //MPI be adjusted.  Since pry_temps resets all current temps and thus all current posteriors this is currently order dependent
	    //MPI The order dependence probably only affects the chain history of what temp/post is stored with swap steps, and nothing else 
	    //MPI It is probably more logical to update all temps at the same time after all swaps are completed. This will also make MPI
	    //MPI much easier to implement.  Proposal, change pry_temps to a function which acts globally on the full list iswaps
	    //MPI and call that function after the swaps are all done.
	    vector<int> ipry(1,i);
	    vector<double>logposts=gather_lposts();
	    pry_temps(ipry,evolve_temp_rate,invtemps,logposts);
	    //invtemps=gather_invtemps();
	  }
	}
      } else { //Conceptually this seems necessary.  It was missing before 11-16-2014.  Didn't notice any problems though, before, though.
	state sA=chains[i].getState();
	//preMPI
	//double llikeA=chains[i].getLogLike();
	//double lpostA=chains[i].getLogPost();  //This does work because temperature doesn't change here.
	//state sB=chains[i+1].getState();
	//double llikeB=chains[i+1].getLogLike();
	//double lpostB=chains[i+1].getLogPost();
	//chains[i].add_state(sA,llikeA,lpostA);  //For each chain, this is is just a copy of the current state
	//chains[i+1].add_state(sB,llikeB,lpostB);//because we tried and failed with a new candidate.
	if(is_my_chain[i])
	  chains[i].add_state(states[i],loglikes[i],chains[i].getLogPost());
	if(is_my_chain[i+1])
	  chains[i+1].add_state(states[i+1],loglikes[i+1],chains[i+1].getLogPost());	
      }	
      swap_count[i]++;
    }
  }

  //Here we perform the standard (non-swap) step for the remaining chains.
  //Either guided or dynamic scheduling seems to work about the same.
  //#pragma omp parallel for schedule (guided, 1)  
  ///take one pass/thread at a time until done.
#pragma omp parallel for schedule (dynamic, 1) 
  for(int iloc=0;iloc<mychains.size();iloc++){
    int i=mychains[iloc];
    //PreMPI for(int i=0;i<Ntemps;i++){
    //MPI: This loop should only be over local chains.  Any DE-cross-breeding may be problematic with MPI
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

  //MPI: Might need a Barrier here for DE-cross-breeding...
  //MPI: This loop should only be over local chains.
  //PreMPI for(int i=0;i<Ntemps;i++){
  for(int i:mychains){
    chains[i].history_thaw();
  }
  //MPI Not clear whether/what this Nsize is needed for. If anything.
  //MPI FIXME
  Nsize=c0().size();
  //cout<<status()<<endl;
  
  //diagnostics and steering:
  icount++;

  if(icount>=istatsbin){ 
    for(int i=0;i<Ntemps-1;i++){
      tryrate[i]=trycounts[i]/(double)icount;
      swaprate[i]=swapcounts[i]/(double)icount;
    }
    if(do_evid){
      //invtemps=gather_invtemps();//Maybe don't need to redo this if careful
      //MPI: log_evidence_ratio will need several bits of chain info to be globally collected here
      double evidence=0;
      for(int i=0;i<Ntemps-1;i++){
	//Compute upside log_evidence ratio
	if(nproc==1){
	  log_eratio_up[i]=   log_evidence_ratio(i  ,i+1,istatsbin,1);
	  log_eratio_down[i]=-log_evidence_ratio(i+1,i  ,istatsbin,1);
	  evidence+=(log_eratio_up[i]+log_eratio_down[i])/2.0;
	} else evidence+=1; //MPI not yet implemented with MPI
      }
      //MPI: For this we need the temps of all chains That can be precommunicated now, or globally maintained.
      //preMPI double dE=(log_eratio_up[Ntemps-2]+log_eratio_down[Ntemps-2])/2.0/(chains[Ntemps-2].invTemp()/chains[Ntemps-1].invTemp()-1);    // Note, if beta 1 is small, r~1 anyway then the result will be about ~ beta1
      double dE=(log_eratio_up[Ntemps-2]+log_eratio_down[Ntemps-2])/2.0/(invtemps[Ntemps-2]/invtemps[Ntemps-1]-1);    // Note, if beta 1 is small, r~1 anyway then the result will be about ~ beta1
      evidence+=dE;
      //evidence*=1.0/(1-chains[Ntemps-1].invTemp());
      //evidence+=evidence*chains[Ntemps-1].invTemp();
      cout<<"Total log-evidence: "<<evidence<<endl;
      //MPI: Nsize
      //cout<<" Nsize="<<Nsize<<endl;
      //Save records of evidence:
      evidence_count++;
      int Ndim=log2(evidence_count);
      if(Ndim>evidence_records_dim){//Add a column for a new larger scale of evidence averages
	total_evidence_records.push_back(vector<double>());
	evidence_records_dim++;
      }
      if(evidence_records_dim>0){//Note that we begin saving once we reach the *second* epoch at each scale.
	total_evidence_records[0].push_back(evidence);
	cout<<"total_evidence_records[0]["<<evidence_count-2<<"]="<<evidence<<endl;	
      }
      for(int i=1;i<evidence_records_dim;i++){
	//We store evidence records averaged on various periods
	//the period here is 2^i, so we store a new sample when
	//then next lower bit in the count rolls over...
	if(evidence_count % (1<<i) == 0 ){
	  cout<<"i="<<i<<" ec="<<evidence_count<<" 1<<(i)="<<(1<<i)<<" mod="<<(evidence_count % (1<<i))<<endl;
	  //we average the last two entries at the next shorter period
	  int iend=total_evidence_records[i-1].size()-1;
	  double new_avg=(total_evidence_records[i-1][iend-1]+total_evidence_records[i-1][iend])/2.0;
	  total_evidence_records[i].push_back(new_avg);
	  //cout<<"averaging ev["<<i-1<<"]["<<iend-1<<"] and ev["<<i-1<<"]["<<iend<<"] to get ev["<<i<<"]["<<total_evidence_records[i].size()-1<<"]"<<endl;
	  //cout<<"ie averaging "<<total_evidence_records[i-1][iend-1]<<" and "<<total_evidence_records[i-1][iend]<<" to get "<<total_evidence_records[i].back()<<endl;
	}
      }
      //report evidence records:
      if(Ndim>0){
	cout<<"total_log_evs:"<<endl;
	int ndisplaymax=-1;
	if(verbose_evid)ndisplaymax=20;
	int ntot=total_evidence_records[0].size();      
	int ndisplay=ntot;
	if(ndisplay>ndisplaymax)ndisplay=ndisplaymax;
	for(int i=ntot-ndisplay;i<ntot;i++){
	  /*
	    for(int j=0;j<(int)log2(i+2);j++){
	    int ind=(i+2)/(1<<j)-2;
	    cout<<total_evidence_records[j][ind]<<"\t";
	    }
	    for(int j=(int)log2(i+2);j<total_evidence_records.size();j++)cout<<NAN<<"\t";
	    cout<<endl;
	  */
	  for(int j=0;j<total_evidence_records.size();j++){
	    //int ind=(i+2)/(1<<j)-2; //i-> ntot-1-(ntot-1-i)*(1<<j)
	    int ind=(ntot+1)/(1<<j)+(i-ntot)-1;
	    if(ind>=0)
	      cout<<total_evidence_records[j][ind]<<"\t";
	    else
	      cout<<"      ---      "<<"\t";
	  }
	  cout<<endl;
	}
      }
      //report recent evidence stdevs:
      cout<<"recent ev analysis:"<<endl;
      Ndim=total_evidence_records.size();
      for(int j=0;j<Ndim-1;j++){
	int Nvar=2*(Ndim-j)+1;
	int N=total_evidence_records[j].size();
	if(N>=Nvar){//compute and report variance
	  double sum1=0,sum2=0;
	  for(int i=N-Nvar;i<N;i++){
	    double ev=total_evidence_records[j][i];
	    sum1+=ev;
	    sum2+=ev*ev;
	  }
	  double mean=sum1/Nvar;
	  double variance=(sum2-sum1*mean)/(Nvar-1);
	  double stderr=sqrt(variance/Nvar);
	  cout<<j<<": N="<<Nvar<<" <ev>="<<sum1/Nvar<<" sigma="<<sqrt(variance)<<" StdErr="<<stderr<<endl;
	  if(stderr<best_evidence_stderr)best_evidence_stderr=stderr;
	}
      }
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

  //***** Reboot *****//
  //MPI: For initial implementation probably should disable Rebooting with MPI until we know it is needed.
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
      //MPI maxes needs to be globally communicated for this to work
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
	//max posterior of any colder chain.  In this case meaningful exchange between the chains may be
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


///A scheme for evolving the temperatures to encourage mixing:
///
///This function is called each time there is a replica exchange.  It functions in two ways.  First, each time the states of
///Two temperatures are exchanged, we push those temperatures slightly apart relative to the others, while maintaining the
///same min and max temps.  The effect, by itself is to move the temps around until the exchange rates are equal for all temperature
///steps.  The rate argument controls the rate at which we drive toward this equilibrium.
///
///Secondly, if the evolve_temp_lpost_cut is engaged, then we similarly push apart any chains for which the current posterior values are
///out of order by more than the cut value.  The expected order is that hotter chains have larger posteriors as elaborated below.  Since the
///posterior values jump around stochasitically, it makes sense to allow some tolerance of "disordered" posterior values. Pushing the
///temps apart reduces the contibution of the likelihood for the hotter chain and thus reduces its posterior value toward the cut value.
///
/// The motivation for enforcing posterior ordering is to prevent too much exchange between chains with extremely different likelihoods
/// so that various replicas are pushed toward independently finding their ways toward high-likelihood regions, rather than perhaps learing
/// the route from other replicas.  The possiblity of learning from other replicas is characteristic of differential evolution proposal
/// distributions, which is an advantage when the chains are providing decent sampling of the high-likelihood region. The trouble is that
/// when exceptionally low-likelihood replicas are exchanging with high-likelihood replicas, the low-likelihood replicas are more likely to
/// "learn" then path to that same high-likelihood region, which ends up with too many replicas, and reduces the pool of high-temp replicas
/// searching for alternative high-likelihood areas.  This hypothesis is how I interpret the observed tendency of runs with evolving temperatures
/// to yield many replicas exploring a nearly disjoint subset of the high-posterior region and few or none in other regions, which the selection
/// of which regions are explored dependent on the randomization of the run. That is parameter space ranges of the various replicas are strongly
/// correlated with each other, yielding good mixing in only part of the parameter space.
///
///Thermal ordering of posteriors:
///  The annealed evidence or annealed marginal likelihood is the expectation value of the thermally weighted likelihood
///  As is standard for thermal integration, the evidence ratio between nearby inverse temperatures \beta
///  (Z(\beta_b)/Z(\beta_a) ~ (\beta_b-\beta_a) <ln(L)>
///  Another way to write this is d lnZ / d\beta ~ <ln(L)>
///  Where the expectation value of ln L is computed at either temp a or b (or more accurately, the average)
///  For b colder than a (\beta_b-\beta_a)>0 with <ln L> generally <0, we see that the relevant evidence ratio is always negative
///  meaning that Z(\beta) should always decrease with increasing \beta.
///  On average then, we expect the annealed evidence, and therefore also typical values of the posterior, to *increase* with hotter temperature. 
///  Thus we expect, for a mature set of chains, that the posterior values are typically ordered from largest to smallest.
///  If that is far from the case, something is wrong; then is may be desirable to keep the hotter chain, with the excessively small
///  posterior at a hotter temperature to allow it to anneal and to keep it from exchanging with the colder chain, which may lead to less
///  independence of the set of cold instances with DE as we may end of "greasing" the path to that particular cold state.
///
///Cut scaling:
///  We set that the cut limit is proportional to the inverse temperature of the lower chain. This means
///  that smaller mis-orderings are more significant for hotter chains.  This makes some sense if we think
///  we are essentially setting a cut on the likelihood difference allowed between mis-ordered chain pairs.
///  That makes sense also if we want to "protect" the colder chain's better likelihood from jumping up to
///  the hotter chain. Default is that cut=0.
///
/// The latest version pries several gaps at once.
void parallel_tempering_chains::pry_temps(const vector<int> &ipry, const double rate, vector<double> &all_invtemps, const vector<double> &all_lposts ){
  // Arguments:
  // ipry          in      vector with lower index for all accepted swaps
  // rate          in      rate factor controlling the strength of thermal driving 
  // all_invtemps  in/out  vector with full set of replica inverse temperatures
  // all_lposts    in      vector with log posterior for all replicas, used with posterior ordering enforcement
  vector< double > splits(Ntemps-1);
  for(int i=0;i<Ntemps-1;i++){
    //preMPI splits[i]=chains[i].invTemp()-chains[i+1].invTemp();
    splits[i]=all_invtemps[i]-all_invtemps[i+1];
    if(evolve_temp_lpost_cut>=0 &&
       //Previously, these pries to loosely enforce posterior ordering were done every time a swap was accepted
       //Now we are moving to doing pries in batches.  To approximately recover the same effect we raise the
       //order enforcement to the power of the number of swaps/pries in the batch.
       //preMPI chains[i].current_lpost-chains[i+1].current_lpost>evolve_temp_lpost_cut*chains[i].invTemp()){
       all_lposts[i]-all_lposts[i+1]>evolve_temp_lpost_cut*all_invtemps[i]){
      //cout<<"splits["<<i<<"]="<<splits[i]<<"-->";
      splits[i]*=pow(1.0+rate,ipry.size());
      //cout<<splits[i]<<endl;
    }
  }
  for(int i:ipry)splits[i]*=1.0+rate;
  double sum=0;
  for(int i=0;i<Ntemps-1;i++)sum+=splits[i];
  
  //double norm=1+rate*splits[ipry]/(1-chains[Ntemps-1].invTemp());
  double norm=sum/(1-all_invtemps[Ntemps-1]);
  double invtemp=1;
  for(int i=1;i<Ntemps-1;i++){
    invtemp-=splits[i-1]/norm;
    //MPI reset temps can only be done for local chains, changes temp and posterior
    //MPI this, thereby messes with the global temps and posteriors
    //chains[i].resetTemp(invtemp);
    all_invtemps[i]=invtemp;
    if(is_my_chain[i])
      chains[i].resetTemp(invtemp);
  }
}
void parallel_tempering_chains::pry_temps(int ipry, double rate){
  vector< double > splits(Ntemps-1);
  double sum=0;
  for(int i=0;i<Ntemps-1;i++){
    splits[i]=chains[i].invTemp()-chains[i+1].invTemp();
    if(evolve_temp_lpost_cut>=0 &&
       chains[i].current_lpost-chains[i+1].current_lpost>evolve_temp_lpost_cut*chains[i].invTemp()){
      //cout<<"splits["<<i<<"]="<<splits[i]<<"-->";
      splits[i]*=1.0+rate;
      //cout<<splits[i]<<endl;
    }
    if(i==ipry)splits[i]*=1.0+rate;
    sum+=splits[i];
  }
  double norm=sum/(1-chains[Ntemps-1].invTemp());
  //double norm=1+rate*splits[ipry]/(1-chains[Ntemps-1].invTemp());
  //splits[ipry]*=1.0+rate;
  double invtemp=1;
  for(int i=1;i<Ntemps-1;i++){
    invtemp-=splits[i-1]/norm;
    //if(i==ipry)cout<<"*";
    //cout<<i<<"changing invtemp from "<<chains[i].invTemp();
    chains[i].resetTemp(invtemp);
    //cout<<"  --->   "<<chains[i].invTemp()<<endl;;
  }
}
  
//MPI parallelism supporting functions
  //interproc_stride: Maximum number of chains per proc.  Fixed at start
  //interproc_unpack_index[i]: Index, within the master package for chain i.

//Collect values for invtemps from all chain
vector<state> parallel_tempering_chains::gather_states(){
  vector<state> states(Ntemps);
  if(not use_mpi){
    for(int i=0;i<Ntemps;i++){
      states[i]=chains[i].getState();
      //if(myproc==0)cout<<"state "<<i<<": "<<states[i].get_string()<<endl;
    }
  } else {
    //This one is a little trickier than the others
    //MPI doesn't directly support passing around class instances
    //We need to also know what the size of each states data is
    //We assume the state-space is the same
    //We need to know the size of each state (number of pars), should not evolve.
    //For transdimensional (not yet supported) this should be the max size.
    const stateSpace *sp=chains[mychains[0]].getState().getSpace();
    int statesize=sp->size();
    int stride=interproc_stride*statesize;
    double sendbuf[stride];
    int recvtotalcount=stride*nproc;
    double recvbuf[recvtotalcount];
    for(int iloc=0;iloc<mychains.size();iloc++){
      state st=chains[mychains[iloc]].getState();
      for(int j=0;j<statesize;j++)
	sendbuf[iloc*statesize+j]=st.get_param(j);
    }
    for(int j=mychains.size()*statesize;j<stride;j++)sendbuf[j]=0;//pad with zeros, probl. unnec.
    MPI_Allgather(&sendbuf, stride, MPI_DOUBLE, &recvbuf, stride, MPI_DOUBLE,MPI_COMM_WORLD);
    for(int i=0;i<Ntemps;i++){
      vector<double> params(statesize);
      for(int j=0;j<statesize;j++)params[j]=recvbuf[interproc_unpack_index[i]*statesize+j];  
      state st=state(sp,params);
      states[i]=st;
      //if(myproc==0)cout<<"state "<<i<<": "<<states[i].get_string()<<endl;
    }
  }
  return states;
}

//Collect values for invtemps from all chains
vector<double> parallel_tempering_chains::gather_invtemps(){
  vector<double> invtemps(Ntemps);
  if(not use_mpi){
    for(int i=0;i<Ntemps;i++)
      invtemps[i]=chains[i].invTemp();
  } else {
    double sendbuf[interproc_stride];
    int recvcount=interproc_stride*nproc;
    double recvbuf[recvcount];
    for(int iloc=0;iloc<mychains.size();iloc++)sendbuf[iloc]=chains[mychains[iloc]].invTemp();
    for(int iloc=mychains.size();iloc<interproc_stride;iloc++)sendbuf[iloc]=0;//pad with zeros, maybe unnec.
    MPI_Allgather(&sendbuf, interproc_stride, MPI_DOUBLE, &recvbuf, interproc_stride, MPI_DOUBLE,MPI_COMM_WORLD);
    for(int i=0;i<Ntemps;i++)
      invtemps[i]=recvbuf[interproc_unpack_index[i]];
  }
  return invtemps;
}

//Collect values for current_lpost from all chains
vector<double> parallel_tempering_chains::gather_lposts(){
  vector<double> lposts(Ntemps);
  if(not use_mpi){
    for(int i=0;i<Ntemps;i++)
      lposts[i]=chains[i].current_lpost;
  } else {
    double sendbuf[interproc_stride];
    int recvcount=interproc_stride*nproc;
    double recvbuf[recvcount];
    for(int iloc=0;iloc<mychains.size();iloc++)sendbuf[iloc]=chains[mychains[iloc]].current_lpost;
    for(int iloc=mychains.size();iloc<interproc_stride;iloc++)sendbuf[iloc]=0;//pad with zeros, maybe unnec.
    MPI_Allgather(&sendbuf, interproc_stride, MPI_DOUBLE, &recvbuf, interproc_stride, MPI_DOUBLE,MPI_COMM_WORLD);
    for(int i=0;i<Ntemps;i++)
      lposts[i]=recvbuf[interproc_unpack_index[i]];  
  }
  return lposts;
}

//Collect values for log like from all chains
vector<double> parallel_tempering_chains::gather_llikes(){
  vector<double> llikes(Ntemps);
  if(not use_mpi){
    for(int i=0;i<Ntemps;i++)
      llikes[i]=chains[i].getLogLike();
  } else {
    double sendbuf[interproc_stride];
    int recvcount=interproc_stride*nproc;
    double recvbuf[recvcount];
    for(int iloc=0;iloc<mychains.size();iloc++)sendbuf[iloc]=chains[mychains[iloc]].getLogLike();
    for(int iloc=mychains.size();iloc<interproc_stride;iloc++)sendbuf[iloc]=0;//pad with zeros, maybe unnec.
    MPI_Allgather(&sendbuf, interproc_stride, MPI_DOUBLE, &recvbuf, interproc_stride, MPI_DOUBLE,MPI_COMM_WORLD);
    for(int i=0;i<Ntemps;i++)
      llikes[i]=recvbuf[interproc_unpack_index[i]];    
  }
  return llikes;
}

///This function computes the evidence ratio between chains a two different temps;
///
///  As is standard for thermal integration, the evidence ratio between nearby inverse temperatures \beta
///  (Z(\beta_b)/Z(\beta_a) ~ (\beta_b-\beta_a) <ln(L)>
///  Another way to write this is d lnZ / d\beta ~ <ln(L)>
///  For small differences between a and b=a+\epsilon
///  The approximation:
///  (Z(\beta_b)/Z(\beta_a) ~ (\beta_b-\beta_a) ( <ln(L)>a + <ln(L)>b ) / 2
///  is good to order \epsilon^2.  Here < . >a means that we take the mean over the samples at temp a.
///  This function computes the log of the one-sided evidence ratio
double parallel_tempering_chains::log_evidence_ratio(int ia,int ib,int ilen,int every){
  //MPI this function needs several bits of global chain info:
  //MPI global chain sizes, temps, istarts, loglikes
  //MPI these are not changed within this func.  Propose to add this global info as function arguments
  double size=chains[ib].get_state_idx(chains[ib].Nhist);
  double istart=chains[ib].get_state_idx(chains[ib].Nhist-ilen);
  double amb=chains[ia].invTemp()-chains[ib].invTemp();
  //to avoid possible overflow we offset by max(lnL*amb) before the sum;
  /*
  double xmax=-1e100;
  for(int i=size-ilen;i<size;i+=every){
    double x=chains[ib].getLogLike(i)*amb;
    if(x>xmax)xmax=x;
    }*/
  double sum=0;
  double count=0;
  for(int i=istart;i<size;i+=every){
    double x=chains[ib].getLogLike(i,true)*amb;
    //double x=chains[ib].getLogLike(i)*amb-xmax;
    sum+=x;
    count++;
    //if(x>-35)sum+=exp(x);
  }
  double result=sum/count;
  //double result=log(sum/ceil(ilen/(double)every))+xmax;
  //cout<<"log_eratio: amb="<<amb<<" --> "<<result<<endl;
  //cout<<"log_eratio: amb="<<amb<<" xmax="<<xmax<<" --> "<<result<<endl;
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

//MPI Info is global, call only from main proc, if still needed.
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

//MPI: This routine will require communication or coordination among procs.
//MPI: One possibility is to do some kind of MPI_Barrier to force procs to take turns in order
string parallel_tempering_chains::show(bool verbose){
    ostringstream s;
    s<<"parallel_tempering_chains(id="<<id<<",Ntemps="<<Ntemps<<",size="<<Nsize<<")\n";
    if(verbose)for(int i=0;i<Ntemps;i++)s<<i<<":"<<chains[i].show(verbose)<<endl;
    return s.str();
};

//MPI: This routine will require communication or coordination among procs.
//MPI: One possibility is to do some kind of MPI_Barrier to force procs to take turns in order
string parallel_tempering_chains::status(){
  ostringstream s;
  s<<"chain(id="<<id<<", Ntemps="<<Ntemps<<"):\n";
  if(not use_mpi){
    for(int i=0;i<Ntemps;i++){
      int istart=0;if(max_reboot_rate>0)istart=Nsize-instance_starts[instances[i]];
      s<<"instance \033[1;31m"<<instances[i]<<"\033[0m["<<istart<<"]";
      //<<"("<<(directions[i]>0?"+":"-")<<" , "<<up_frac[i]<<")"
      if(i<Ntemps-1)s<<"("<<swaprate[i]<<" of "<<tryrate[i]<<"):";
      else s<<"                  ";
      s<<":                     "<<chains[i].status()<<"\n";
      //cout<<swaprate.size()<<" "<<tryrate.size()<<" "<< log_eratio_down.size()<<" "<< log_eratio_up.size()<<endl;
      if(do_evid and verbose_evid and i<Ntemps-1)s<<" log eratio:("<<log_eratio_down[i]<<","<<log_eratio_up[i]<<")"<<endl;    
    }
  } else {
    int maxwid=2000;
    char allstrings[interproc_stride*maxwid*nproc];
    char mystrings[maxwid*interproc_stride];
    for(int ii=0;ii<interproc_stride;ii++){
      ostringstream ss;
      if(ii<mychains.size()){
        int i=mychains[ii];
	int istart=0;if(max_reboot_rate>0)istart=Nsize-instance_starts[instances[i]];
	ss<<"instance \033[1;31m"<<instances[i]<<"\033[0m["<<istart<<"]";
	//<<"("<<(directions[i]>0?"+":"-")<<" , "<<up_frac[i]<<")"
	if(i<Ntemps-1)ss<<"("<<swaprate[i]<<" of "<<tryrate[i]<<"):";
	else ss<<"                  ";
	ss<<":                     "<<chains[i].status()<<"\n";
        if(do_evid and verbose_evid and i<Ntemps-1)ss<<" log eratio:("<<log_eratio_down[i]<<","<<log_eratio_up[i]<<")"<<endl;
      }
      strncpy(mystrings+ii*maxwid,ss.str().c_str(),maxwid-1);
      //if(ii<mychains.size())cout<<"proc "<<myproc<<" setting up temp "<<mychains[ii]<<":\n"<<(mystrings+ii*maxwid)<<endl;
    } 
    MPI_Allgather(mystrings, interproc_stride*maxwid, MPI_CHAR, allstrings, interproc_stride*maxwid, MPI_CHAR,MPI_COMM_WORLD);
    for(int i=0;i<Ntemps;i++){
      s<<(allstrings+interproc_unpack_index[i]*maxwid);
      //cout<<"proc "<<myproc<<" appending temp "<<i<<":\n"<<(allstrings+i*maxwid)<<endl;
    }
  }
  if(do_evid)s<<"Best evidence stderr="<<best_evidence_stderr<<endl; 
  return s.str();
};
 
string parallel_tempering_chains::report_prop(int style){//Not yet MPI aware
  ostringstream result("");
  for(auto i: mychains)
    result<<" T="<<temps[i]<<"; "<<props[i]->report(style)<<endl;//"; "<<chains[i].show();
  return result.str();
};


  
    
