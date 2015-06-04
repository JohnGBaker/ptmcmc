#include <valarray>
#include <cmath>
#include "MCMC.hh"

///Idea:
/// For more vigorous testing.  It would be nice to have a model where the peaks do not all have the same shape,
/// and in particular to have some that have smaller catchment regions than others.  For "sines" the catchment regions are identical.
/// A concrete way to do this would be to write the likelihood as a sum of Gaussians, in a similar grid where the directional "sigmas"

///A simple multimodal test distribution
///The surface includes an array of peaks.  Each peak is the same size and shape. If step_scale!=0 then the block-like region around each peak is shifted in probability in a diagonal stair-step fashion with the specified (logarithmic) step-size.
class sines: public probability_function{
  int Ndim;
  valarray<int> ks;
  valarray<double> mins;
  valarray<double> maxs;
  double height;
  ///scale for stepped peak likelihoods
  double step_scale;
 public:
  sines(stateSpace *sp,double height,valarray<int> &ks,valarray<double>&mins,valarray<double>&maxs,double stepscale=0):height(height),ks(ks),mins(mins),maxs(maxs),step_scale(stepscale),probability_function(sp){Ndim=ks.size();};
  double evaluate_log(state &s){
    double lprod=0;
    valarray<double>params=s.get_params();
    for(int i=0;i<Ndim;i++){
      double x=(params[i]-mins[i])/(maxs[i]-mins[i]);
      //if(x<mins[i]||x>maxs[i])prod=0;
      //else
      double s=sin(ks[i]*M_PI*x);
      s=s*s;
      lprod+=(s*s-1)*height;
    }
    //cout<<"lprod="<<lprod<<endl;
    valarray<int> idxs(Ndim);
    idxs=whichPeak(s);
    //cout<<"  idxs="<<idxs[0]<<" "<<idxs[1]<<endl;
    return lprod+nontrivial_step_func(idxs);
  };
  valarray<int> whichPeak(const state s){
    int idxs[Ndim];
    valarray<double>params=s.get_params();
    for(int j=0;j<Ndim;j++){
      double x=(params[j]-mins[j])/(maxs[j]-mins[j]);
      idxs[j]=x*ks[j];
    }
    valarray<int> result(idxs,Ndim);
    return result;
  };
  double trivial_step_func(const state s){return 1;};
  double nontrivial_step_func(const valarray<int>idxs){
    int isum=0;
    for(int j=0;j<Ndim;j++)isum+=idxs[j];
    return -isum*step_scale;
  };
  valarray<double> peak_location(const valarray<int> idxs){
    valarray<double>result(Ndim);
    result=mins;
    for(int j=0;j<Ndim;j++)result[j]+=(idxs[j]+0.5)/ks[j];
    return result;;
  };
};

// Prior implementing step levels that are flat, but differing discontinuously in the vicinity of each peak.
// The prior implements the *same* manner of stair-step offseting as is provided in the sines class.
// This allows experiments testing the effect of likelihood vs prior changes between peaks.
class stepped_prior: public sampleable_probability_function{
  int Ndim,Npeaks;
  valarray<double> mins;
  valarray<double> maxs;
  double step_scale;
  double lognorm;
  valarray<double> cuts;
  valarray<valarray<int> > c_idxs;

public:
  stepped_prior(stateSpace *space,valarray<int>&ks,valarray<double>&mins,valarray<double>&maxs,double stepscale=0):ks(ks),mins(mins),maxs(maxs),step_scale(stepscale),sampleable_probability_function(space){
    Ndim=ks.size();
    compute_cuts();
  };
  state drawSample(Random &rng){
    valarray<double>cuts(0,Ndim*Ndim);
    int ic=0;
    double xrand=rnd.Next();
    for(int ic=0;i<Npeaks;i++)
      if(xrand<cuts(ic))break;
    valarray<int>idxs=c_idxs[ic];//Structures/syntax wrong FIXME
    valarray<double>params(Ndim);
    for(int i=0;i<Ndim;i++){
      //inverting idxs(params) as in this:
      //double x=(params[j]-mins[j])/(maxs[j]-mins[j]);
      //idxs[j]=x*ks[j];
      double pwid=(maxs[i]-mins[i])/(double)ks[i];
      double pmin=mins[i]+idxs(i)*pwid;
      params[i]=pmin+rnd.Next()*pwid;
    }
    return state(space,params);
  };
  
  double evaluate(state &s){return exp(evaluate_log(s));};
  double evaluate_log(state &s){
    idxs=whichPeak(s);
    return nontrivial_step_function(idxs)-lognorm;
  };
  double nontrivial_step_func(const valarray<int>idxs){
    int isum=0;
    for(int j=0;j<Ndim;j++)isum+=idxs[j];
    return -isum*step_scale;
  };
  string show();
  protected:
  valarray<int> whichPeak(const state s){
    int idxs[Ndim];
    valarray<double>params=s.get_params();
    for(int j=0;j<Ndim;j++){
      double x=(params[j]-mins[j])/(maxs[j]-mins[j]);
      idxs[j]=x*ks[j];
    }
    valarray<int> result(idxs,Ndim);
    return result;
  };
  protected:
  void comp_cuts(){
    valarray<int> idxs(Ndim);
    valarray<double> cuts(0,Npeaks);
    for(auto&& i:idxs)i=0;//initialize 
    bool going=true;
    int idim=0,ic=0;
    double sum=0;
    while(idim<Ndim){//This is a hyperdimensional loop over all peaks.
      if(idim==0){//Do something non-trivial when we are stepping on the leaf
	double val=exp(nontrivial_step_func(idxs));
	sum+=log(val);
	cuts[ic]=sum;
      }	
      c_idxs[ic]=idxs;
      idxs[idim]++;
      if(idxs[idim]>=ks[idim]){//done with this level, roll up to next dim
	idxs[idim]=0;
	idim++;
      } else {                 //not done at this level, return to looping at leaf level
	idim=0;
      }
    }
    //Now normalize the cuts:
    for(auto&& c:cuts)c/=sum;
    lognorm=log(sum);
    valarray<int> result(idxs,Ndim);
    return result;
};

