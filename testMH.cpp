#include <valarray>
#include <iostream>
#include <fstream>
#include "bayesian.hh"
#include "chain.hh"
#include "probability_function.hh"
#include "proposal_distribution.hh"
#include "sines.hh"
#include <string>
#include <sstream>

using namespace std;

//Control parameters
const int Ndim=2;
int Npeaki[]={2,2};
int Nchain=6;
double nburn_frac=0.100;
int Nstep; //Number of steps attemped in each chain
double peakbias=log(2.0);
double height=64.0;
int Nprop_set=4;


//Globals for expectation evaluation functions
sines *llike;
valarray<int> ref_peak(Ndim);//use to specify peak with global functions below.
int ref_dir;//use to specify coordinate in estimating peak.
//Function to estimate the integrated likelihood of each peak
double for_peak_likelihood(state s){
  valarray<int>peak=llike->whichPeak(s);
  bool eq=true;
  for(int i=0;i<Ndim;i++)eq=eq&&(peak[i]==ref_peak[i]);
  return eq?1.0:0.0;
};
//Function to estimate the marginalized position of each peak
double for_peak_position(state s){
  valarray<int>peak=llike->whichPeak(s);
  valarray<double>params=s.get_params();
  bool eq=true;
  for(int i=0;i<Ndim;i++)eq=eq&&(peak[i]==ref_peak[i]);
  return (eq?1.0:0.0)*params[ref_dir];
};

//***************************************************************************************8
//main test program
int main(int argc, char*argv[]){
  double seed;
  string outname;
  int proposal_option;
  if(argc!=5||!(istringstream(argv[1])>>seed)){
    cout<<"argc="<<argc<<endl;
    if(argc>1)cout<<"argv[1]="<<argv[1]<<endl;
    cout<<"Usage testMH seed N-steps proposal_option outname\n(seed) should bin in [0,1)."<<endl;
    return -1;
  }
  istringstream(argv[2])>>Nstep;
  int nburn=Nstep*nburn_frac;
  istringstream(argv[3])>>proposal_option;
  outname=string(argv[4]);;
  cout<<"seed="<<seed<<endl;
  ProbabilityDist::setSeed(seed);
  
  //Set up the parameter space
  stateSpace space(Ndim);
  string names[Ndim];  
  for(int i=0;i<Ndim;i++){
    ostringstream ss("p");ss<<i;
    names[i]=ss.str();
  }
  space.set_names(names);  
  cout<<"Parameter space:\n"<<space.show()<<endl;
  
  valarray<double> mins,maxs,x0s,sigmas;
  valarray<int>ks(Npeaki,Ndim);
  //set the prior
  mins.resize(Ndim,0.0);
  maxs.resize(Ndim,1.0);
  uniform_dist_product prior(&space,mins,maxs);
  sigmas.resize(Ndim,1.0);
  x0s.resize(Ndim,0.5);
  //gaussian_dist_product prior(x0s,sigmas);

  //set the likelihood
  llike=new sines(&space,height,ks,mins,maxs,peakbias);

  //Set the proposal distribution
  int Ninit = 1;
  valarray<double> zeros=x0s*0.;
  proposal_distribution* test_prop;
  switch(proposal_option){
  case 0:  //Draw from prior distribution   
    cout<<"Selected draw-from-prior proposal option"<<endl;
    test_prop=new draw_from_dist(prior);
    break;
  case 1:  //gaussian   
    cout<<"Selected Gaussian proposal option"<<endl;
    test_prop=new gaussian_prop(sigmas/8.);
    break;
  case 2:  {  //range of gaussians
    cout<<"Selected set of Gaussian proposals option"<<endl;
    vector<proposal_distribution*> gaussN(Nprop_set);
    vector<double>shares(Nprop_set);
    double fac=1;
    for(int i=0;i<Nprop_set;i++){
      fac*=2;
      gaussN[i]=new gaussian_prop(sigmas/fac);
      shares[i]=fac;
      cout<<"  sigma="<<sigmas[0]/fac<<", weight="<<fac<<endl;
    }
    test_prop=new proposal_distribution_set(gaussN,shares);
    break;
  }
  case 3:{
    cout<<"Selected differential evolution proposal option"<<endl;
    //differential_evolution(bool snooker=false, double gamma_one_frac=0.1,double b_small=0.0001,double ignore_frac=0.3):snooker(snooker),gamma_one_frac(gamma_one_frac),b_small(b_small),ignore_frac(ignore_frac)
    //test_prop=new differential_evolution();
    test_prop=new differential_evolution(0.0,0.3,0.0001,0.0);
    Ninit=1000*Ndim;//Need a hug pop of samples to avoid getting stuck in a peak unless occasionally drawing from prior.
    break;
  }
  case 4:{
    cout<<"Selected differential evolution with snooker updates proposal option"<<endl;
    //differential_evolution(bool snooker=false, double gamma_one_frac=0.1,double b_small=0.0001,double ignore_frac=0.3):snooker(snooker),gamma_one_frac(gamma_one_frac),b_small(b_small),ignore_frac(ignore_frac)
    test_prop=new differential_evolution(0.1,0.3,0.0);
    Ninit=1000*Ndim;
    break;
  }
  case 5:{
    cout<<"Selected differential evolution proposal with prior draws option"<<endl;
    //differential_evolution(bool snooker=false, double gamma_one_frac=0.1,double b_small=0.0001,double ignore_frac=0.3):snooker(snooker),gamma_one_frac(gamma_one_frac),b_small(b_small),ignore_frac(ignore_frac)
    //test_prop=new differential_evolution();
    vector<proposal_distribution*>props(2);
    vector<double>shares(2);
    props[0]=new draw_from_dist(prior);
    shares[0]=0.1;
    props[1]=new differential_evolution(0.0,0.3,0.0001,0.0);
    shares[1]=0.9;
    test_prop=new proposal_distribution_set(props,shares);    
    Ninit=100*Ndim;
    break;
  }
  case 6:{
    cout<<"Selected differential evolution (with snooker updates) proposal with prior draws option"<<endl;
    //differential_evolution(bool snooker=false, double gamma_one_frac=0.1,double b_small=0.0001,double ignore_frac=0.3):snooker(snooker),gamma_one_frac(gamma_one_frac),b_small(b_small),ignore_frac(ignore_frac)
    //test_prop=new differential_evolution();
    vector<proposal_distribution*>props(2);
    vector<double>shares(2);
    props[0]=new draw_from_dist(prior);
    shares[0]=0.1;
    props[1]=new differential_evolution(0.1,0.3,0.0001,0.0);
    shares[1]=0.9;
    test_prop=new proposal_distribution_set(props,shares);    
    Ninit=100*Ndim;
    break;
  }
  default:
    cout<<"Unrecognized value: proposal_option="<<proposal_option<<endl;
    exit(1);
  }

  //*************************************************
  //Prepare analysis variables 
  int ncount=1;
  for(int i=0;i<Ndim;i++)ncount*=ks[i];
  cout<<"ncount="<<ncount<<endl;//ncount=number of peaks
  valarray<valarray<double> > P_est(Nchain),P_var(Nchain),est_peak_width(Nchain);
  valarray<valarray<valarray<double> > > est_peak_loc(Nchain), peak_loc_var(Nchain);
  valarray<double> P_thy(ncount), net_prob(0.0,ncount), net_prob_var(0.0,ncount), net_peak_loc_rvar(ncount),est_peak_width_mean(0.0,ncount);
  valarray<valarray<double> > peak_loc(ncount), net_peak_loc(ncount), net_peak_loc_var(ncount);
  valarray<valarray<int> > iis(ncount);
  valarray<int> pcount(ncount);
  double probsum=0,possum=0;
  
  //Ndimensional loop over peaks to prep for analysis
  bool done=false;
  valarray<int> ii(0,Ndim);
  int icount=0;
  while(not done){
    iis[icount].resize(Ndim);
    peak_loc[icount].resize(Ndim);
    net_peak_loc[icount].resize(Ndim);
    net_peak_loc_var[icount].resize(Ndim,0.0);
    iis[icount]=ii;
    peak_loc[icount]=(llike->peak_location(ii));
    probsum+=(P_thy[icount]=exp(llike->nontrivial_step_func(ii)));
    //loop increment and test
    done=true;icount++;
    for(int j=Ndim-1;j>=0;j--){
      if(ii[j]<ks[j]-1){ii[j]++;done=false;break;
      } else ii[j]=0;
    }
  }
  //Normalize
  P_thy/=probsum;

  //Prepare for chain output
  ostringstream ss("");
  ss<<"testMH_"<<outname<<".out";
  ofstream out(ss.str().c_str());
  ss.str("");ss<<"testMH_"<<outname<<"_eval.out";
  ofstream oute(ss.str().c_str());
  
  //*************************************************
  //Loop over Nchains
  for(int ic=0;ic<Nchain;ic++){

    //Create the Chain
    MH_chain c(llike,&prior);
    test_prop->set_chain(&c);
    proposal_distribution* test_prop2=test_prop->clone();
    //Initialize
    c.initialize(Ninit);
    //cout<<"Initialized Chain:"<<endl;c.dumpChain(cout,-Ninit);

    for(int i=0;i<Nstep;i++){
      //cout<<"step="<<i<<endl;
      c.step(*test_prop2);
    }
    //if(ic==0)out<<c.getStateStringLabels()<<endl;//write header
    c.dumpChain(out,nburn);
    out<<"\n"<<endl;
  
    ///Perform analysis
    probsum=0;possum=0;
    P_est[ic].resize(ncount);
    est_peak_loc[ic].resize(ncount);
    est_peak_width[ic].resize(ncount);
    if(ic==0)oute<<" x_true  y_true:  x_est  y_est:  P_true  P_est[ic]:  dr    dP: peak_width"<<endl;
    oute<<"chain "<<ic<<endl;
    for(int i=0;i<ncount;i++){
      valarray<int> ii=iis[i];
      cout<<"measuring peak "<<i<<" at (";for(int j=0;j<Ndim-1;j++)cout<<ii[j]<<",";cout<<ii[Ndim-1]<<")"<<endl;
      ref_peak=ii;
      P_est[ic][i]=c.expectation(for_peak_likelihood,nburn);
      est_peak_loc[ic][i].resize(Ndim);
      double varval=0;
      for(int j=0;j<Ndim;j++){
        ref_dir=j;
	double val=c.expectation(for_peak_position,nburn);
        net_peak_loc[i][j]+=val/Nchain;
	val/=P_est[ic][i];
	varval+=c.variance(for_peak_position,val,nburn);
	est_peak_loc[ic][i][j]=val;
      }
      est_peak_width[ic][i]=varval/P_est[ic][i];
      net_prob[i]+=P_est[ic][i]/Nchain;
      double rsum=0;
      for(int j=0;j<Ndim;j++)oute<<" "<<peak_loc[i][j];
      oute<<": ";
      for(int j=0;j<Ndim;j++){
        oute<<" "<<est_peak_loc[ic][i][j];
        double val=est_peak_loc[ic][i][j]-peak_loc[i][j];
        val=val*val;
        net_peak_loc_var[i][j]+=val/Nchain;
        rsum+=val;
      }
      possum+=rsum;
      net_peak_loc_rvar[i]+=rsum/Nchain;
      if(P_est[ic][i]>0){
	est_peak_width_mean[i]+=sqrt(est_peak_width[ic][i]);
	pcount[i]++;
      }
      double val=P_est[ic][i]-P_thy[i];
      oute<<": "<<P_thy[i]<<" "<<P_est[ic][i]<<": "<<sqrt(rsum)<<" "<<val<<": "<<sqrt(est_peak_width[ic][i])<<endl;
      val=val*val;
      probsum+=val;
      net_prob_var[i]+=val/Nchain;
    }
    oute<<"#N="<<Nstep<<"resid:"<<sqrt(possum/ncount)<<" "<<sqrt(probsum/ncount)<<endl;    
  }

  oute.close();
  out.close();
  
  //*************************************************
  //All-chain analysis
  //estimates for variance w.r.t. sample mean.
  valarray<double>net_peak_loc_rvar_est(0.0,ncount);
  valarray<double>est_peak_width_var(0.0,ncount);
  valarray<double>net_prob_var_est(0.0,ncount);
  for(int i=0;i<ncount;i++){
    int rvar_est_cnt=0;
    est_peak_width_mean[i]/=pcount[i];
    for(int j=0;j<Ndim;j++){
      net_peak_loc[i][j]/=net_prob[i];
      for(int ic=0;ic<Nchain;ic++){
        double val=est_peak_loc[ic][i][j]-net_peak_loc[i][j];
        double wval=sqrt(est_peak_width[ic][i])-est_peak_width_mean[i];
	if(!isnan(val)){
	  net_peak_loc_rvar_est[i]+=val*val;
	  est_peak_width_var[i]+=wval*wval;
	  rvar_est_cnt++;
	}
      }
    }
    for(int ic=0;ic<Nchain;ic++){
      double val=P_est[ic][i]-net_prob[i];
      net_prob_var_est[i]+=val*val;
    }
    net_prob_var_est[i]/=(Nchain-1);
    net_peak_loc_rvar_est[i]/=(rvar_est_cnt-1);
    est_peak_width_var[i]/=(rvar_est_cnt-1);
  }

  //Dump summary info
  ss.str("");ss<<"testMH_"<<outname<<"_evalsum.out";
  out.open(ss.str().c_str());
  out<<"#Nchain="<<Nchain<<" Nstep="<<Nstep<<" prop="<<proposal_option<<endl;
  out<<"x_true y_true: x_est y+est: chains_r_var: width chains_width_var: P_true P_est: chains_P_var:: r_err P_err"<<endl;
  if(proposal_option==2)out<<"."<<Nprop_set;
  out<<endl;
  probsum=0;possum=0;
  double probsum2=0,possum2=0,widsum2=0;;
  for(int i=0;i<ncount;i++){
    for(int j=0;j<Ndim;j++)out<<" "<<peak_loc[i][j];
    out<<": ";
    double dr=0;
    for(int j=0;j<Ndim;j++){
      out<<" "<<net_peak_loc[i][j];
      double val=net_peak_loc[i][j]-peak_loc[i][j];
      dr+=val*val;
    }
    out<<": "<<sqrt(net_peak_loc_rvar_est[i]);
    out<<": "<<est_peak_width_mean[i]<<" "<<sqrt(est_peak_width_var[i]);
    out<<": "<<P_thy[i]<<" "<<net_prob[i]<<": "<<sqrt(net_prob_var_est[i]);
    out<<":: "<<sqrt(dr)<<" "<<sqrt(net_prob_var[i])<<endl;
    probsum+=net_prob_var[i];
    possum+=dr;
    probsum2+=net_prob_var_est[i];
    possum2+=net_peak_loc_rvar_est[i];
    widsum2+=est_peak_width_var[i];
  }
  out<<"peakRMS: Rerr,Perr,Rvar,Pvar,width_var: "<<sqrt(possum/ncount)<<" "<<sqrt(probsum/ncount)<<": "<<sqrt(possum2/ncount)<<" "<<sqrt(probsum2/ncount)<<" "<<sqrt(widsum2/ncount)<<endl;
  out.close();
}
