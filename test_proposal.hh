///John Baker NASA-GSFC (2019)
///
///This file provides a class for managing testing of proposal distributions.
/// FIXME: SO FAR THIS IS JUST A SKETCH OF A PROPOSED IDEA

#include "states.hh"
#include "probability_function.hh"
#include "proposal_distribution.hh"
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include "Mrpt.h"
#include <algorithm>

using namespace std;

///Class for managing testing of proposal distributions.
///
///This is to allow unit testing to maintain existing proposals, and most
///crucially to allow a systematic way of testing planned user-defined proposals.
///Tests are based on verifying that the image of a set of samples that have been
///drawn from some target distribution and then transformed by a MH step [wrt the
///target distribution] on each point of the sample set using the proposal
///distribution does still match the target distribution.  The match can be
///measured by the Kullback-Leibler (KL) divergence.
///
///A sampleable_probability_function can be provided as the target distribution.
///Eventually plan for the option of auto-generating of an appropriate target
///distribution.
///
///  Functioning version of proposal testing. This is class which defines a
/// test of a proposal_distribution. The test applies the proposal multiple
/// times and then compares the resulting distribution against the original
/// sample distirbution and also against an independently redrawn sample.
/// The first test is based on the KLdivergence among the sample distributions.
/// Since KLdiv is asymmetric, this is tried both ways.  To quantify the result
/// the same test is applied to multiple re-draws of the original sample 
/// distribution.  The test is considered to pass if the transformed sample
/// KLdiv value does not exceed some percentile of the re-draw comparison
/// values, stated explicitly in the output.  Since the KLdiv test is evidently
/// prone to a fair amount of noise a second "fake" KL test is performed which
/// is computed from only the mean and variance of each distribution.  This
/// test statistic reduces to the KL test for sufficiently large Gaussian
/// distributions, but it is much faster and less noisy, thus more sensitive
/// to variance and mean differences, though it would be insensitive to 
/// differences which don't show up in the mean and variance.
class test_proposal {
  shared_ptr<Random> rng;
  proposal_distribution *proposal=nullptr;
  sampleable_probability_function *target_dist=nullptr;
  bool loop;
  chain ch;
  string path;
public:
  //test_proposal(proposal_distribution &proposal, shared_ptr<Random> rng){cout<<"The idea is to provide a ready-made test distribution, perhaps a gaussian mixture, perhaps augmented by gaussians located at proposal-step images of the initial gaussian centers.  This is not yet implemented."<<endl;exit(-1);};
  test_proposal(proposal_distribution &proposalref, sampleable_probability_function &target, bool loop=false, string path_="", vector<int> index=vector<int>()):
    rng(new MotherOfAll(ProbabilityDist::getPRNG()->Next())),
    proposal(&proposalref),
    target_dist(&target),
    loop(loop),
    path(path_)
  {
    //Note: if proposal needs a chain, we may need to make one up with samples from the target dist.
    //configure output name
    if(path!=""){
      //cout<<"path.back()='"<<path.back()<<"'"<<endl;
      //cout<<(path.back()=='/'?"match":"non-match")<<endl;
      if(path.back()=='/')path=path+"test_prop";
      if(index.size()>0)path=path+"_"+to_string(index[0]);
      if(index.size()>1)for(int i: vector<int>(index.begin()+1,index.end()))path+="-"+to_string(i);
      cout<<"path='"<<path<<"'"<<endl;
    }
			    
    //If proposal is a proposal_set and index is provided, then we access only the sub proposal
    while(index.size()>0){
      proposal_distribution_set *set=dynamic_cast<proposal_distribution_set*>(proposal);
      if(set){
	proposal=set->members()[index[0]];
	vector<int> subindex(index.begin()+1,index.end());
	index=subindex;
      }else break;
    }
  };
  
  test_proposal(){};//Trivial constructor for development and testing...
  vector<state> sample_target_dist(int nsamp, double maxfail=1e100){
    //make a set of samples from the target distribution
    vector<state> samples(nsamp);
    bool done=false;
    for( auto &sample : samples){
      int failcount=0;
      done=false;
      while(not (done or failcount>maxfail)){
	sample=target_dist->drawSample(*rng);
	sample.enforce();
	if(not sample.invalid())done=true;
	else failcount++;
	//if(sample.get_params_vector()==copy.get_params_vector())done=true;
	//else cout<<"rejected state "<<copy.get_string()<<endl;
	//No we are using rejection sampling to reject any part of the
	//target distribution outside of the stateSpace domain
      }
      if(not done){
	cout<<"test_proposal::sample_target_dist: Struggling to find draw valid samples. Bailing out."<<endl;
	return vector<state>(0);//
      }
    }
    return samples;
  };
  void write_samples(vector<state> samps, string tag){
    if(path!=""){
      string file=path+"_"+tag+".dat";
      cout<<"Will write to "<<file<<endl;
      ofstream out(file);
      for(auto s : samps)out<<s.get_string()<<endl;
    }
  };
  vector<state> transform_samples(vector<state> &samples, int & Naccept,double hastings_err=0){
    //Apply the proposal via Metropolis-Hastings step, to each sample and return result.
    //Number of acceptances goes in Naccept
    //set hastings_err!=0 to force a nontrivial test (even with a good proposal)
    Naccept=0;
    size_t N=samples.size();
    vector<state> transformed(N);
    
    for(size_t i;i<N;i++){
      //cout<<i<<endl;
      state &oldstate=samples[i];
      double oldlpdf=target_dist->evaluate_log(oldstate);
      oldstate.enforce();//this shouldn't be needed
      if(oldstate.invalid())cout<<"enforcement failure for oldstate "<<oldstate.get_string()<<endl;
      state newstate=proposal->draw(oldstate,&ch); 
      newstate.enforce();
      double newlpdf=target_dist->evaluate_log(newstate);
      
      //Now the test: 
      double log_hastings_ratio=proposal->log_hastings_ratio();
      log_hastings_ratio+=newlpdf-oldlpdf+hastings_err;
      bool accept=true;
      if(newstate.invalid())accept=false;
      if(accept and log_hastings_ratio<0){
	double x=rng->Next(); //pick a number
	accept=(log(x)<log_hastings_ratio);
      }

      //cout<<"old: "<<oldstate.get_string()<<" --> "<<oldlpdf<<endl;
      //cout<<"new: "<<newstate.get_string()<<" --> "<<newlpdf<<endl;
      //cout<<"    "<<(accept?"ACCEPTED    ":    "REJECTED")<<endl;
      if(accept){
	if(newstate.invalid())cout<<"enforcement failure for newstate "<<newstate.get_string()<<endl;
	Naccept++;
	proposal->accept(); //As with MCMC, this allows some proposals to 'adapt'
	transformed[i]=newstate;
	if(oldstate.dist2(newstate)<1e-16){
	  cout<<"test_proposal: Transformed state not significantly different."<<endl;
	  cout<<"                s="<<oldstate.get_string()<<endl;
	  cout<<"               s'="<<newstate.get_string()<<"  dist2="<<oldstate.dist2(newstate)<<endl;
	  vector<state> v={oldstate};
	  involution_proposal *invprop = dynamic_cast<involution_proposal*>(proposal);
	  if(invprop){
	    stateSpaceInvolution involution=invprop->get_involution();
	    cout<<" Intrinsic test:"<<endl;
	    involution.test_involution(oldstate,-1);
	  }
	}
      }
      else {
	proposal->reject();
	transformed[i]=oldstate;
      }
    }
    return transformed;
  };
  ///This is the core test.
  ///
  ///  1. Construct Nsamp samples from the target distribution.
  ///  2. Apply M-H steps based on the proposal to each sample, through as many cycles
  ///     as needed so the total number of accepted proposals is at least Nsamp*ncyc
  ///  3. Compute the KL divergence between the orginal and transformed set distributions
  ///     -KLdiv is generally asymmetric, but here we are interested in near equivalent distributions
  ///      where it is approximately symmetric.
  ///     -Note: For insight, Gaussians differing in 1 dimensions, yield:
  ///        KL(A||B) = (1/2) [ (mu(A)-mu(B))^2 / sigma(B)^2  + sigma(A)^2/sigma(B)^2 - 1 - log(sigma(A)^2/sigma(B)^2) ]
  ///     -Thus for small differences:
  ///        KL( A[mu+dmu*sigma,sigma*(1+dlnsig)] || B[mu,sigma]) ~ dmu^2 + dlnsig^2
  ///  4. A challenge is knowing what level of difference is significant, and what level is measurable.
  ///     One approach to this is to compare KL measurements of redraws of the original sample with
  ///     those of the transformed set.
  ///  5. Note ncyc will be cut back if sampling is too inefficient
  ///  6. If acceptance rate is < min_accept then will return with failure, otherwise returns true.
  bool test(int Nsamp=10000, double ncyc=5, int ntry=10, double hastings_err=0, double min_accept=0){
    cout<<"\n\nTesting proposal: "<<proposal->show()<<endl;
    cout<<"ncyc="<<ncyc<<endl;
    //If loop==true, then we make a new test_proposal object for the sub_proposal, and loop;
    if(loop){
      proposal_distribution_set *set=dynamic_cast<proposal_distribution_set*>(proposal);
      if(set){
	auto members=set->members();
	int n=members.size();
	cout<<n<<" members in proposal set"<<endl;
	int i=0;
	for(auto member : members){
	  cout<<"Preparing and testing member "<<i+1<<" of "<<n<<endl;
	  string subpath=path;
	  if(path!="")subpath=subpath+"-"+to_string(i);
	  cout<<"subpath='"<<subpath<<"'"<<endl;
	  test_proposal subtest(*member, *target_dist,true, subpath);
	  bool ok=subtest.test(Nsamp,ncyc,ntry,hastings_err,min_accept);
	  if(not ok)return false;
	  i++;
	}
	return true;
      }
    }


    //Draw reference samples
    vector<state> target_samples=sample_target_dist(Nsamp,100);
    if(target_samples.size()<Nsamp)return false;
    write_samples(target_samples,"target");
    
    //Perform intrinsic test if any
    string intrinsic_test_result=proposal->test(target_samples,*rng);
    if(intrinsic_test_result!="")cout<<" Intrinsic test:\n"<<intrinsic_test_result<<endl;

    //Transform by application of the proposal
    int Naccept_cum=0;
    int Ngoal=ncyc*Nsamp;
    vector<state> transformed=target_samples;
    cout<<"Applying transform:"<<endl;
    int icyc=1;
    while(Naccept_cum<Ngoal){
      int Naccept=0;
      vector<state> old=transformed;
      transformed=transform_samples(transformed, Naccept,hastings_err);
      Naccept_cum+=Naccept;
      if(icyc%int(ncyc+1)==0)
	cout<<100.0*Naccept/Nsamp<<"\% accepted,  "<<100.0*Naccept_cum/Ngoal<<" percent done."<<endl;
      if(icyc==2000 and Naccept_cum<Ngoal*0.01){//Not sampling fast enough
	//Reduce goal for number of cycles
	ncyc=ncyc/2.0;
	if(Naccept*1.0/Nsamp<min_accept){
	  cout<<"Acceptance rate too low. Giving up."<<endl;
	  return false;
	}
	cout<<"Sampling poorly.  Reducing to ncyc="<<ncyc<<endl;
	//start over
	Ngoal=ncyc*Nsamp;
	icyc=0;
	Naccept_cum=0;
      }	
      icyc++;
    }
    write_samples(transformed,"transformed");

    vector<double> mean,var;
    var=get_sample_var(target_samples,mean);
    cout<<"target mean:      ";for(auto x:mean)cout<<x<<" ";cout<<endl;
    cout<<"target var:       ";for(auto x:var)cout<<x<<" ";cout<<endl;
    var=get_sample_var(transformed,mean);
    cout<<"transformed mean: ";for(auto x:mean)cout<<x<<" ";cout<<endl;
    cout<<"transformed var:  ";for(auto x:var)cout<<x<<" ";cout<<endl;
    
    //Compute KL divergence
    //First decide whether to use approx NN:
    double dim=target_samples[0].size();
    bool use_approxNN=(Nsamp*pow(dim,0.7)>4500); //estimate of when approx is faster
    cout<<"Computing KL divergences:"<<endl;
    double KLtransformed=KL_divergence(transformed,target_samples,use_approxNN);
    double KLtransformedX=KL_divergence(target_samples,transformed,use_approxNN);
    double fKLtransformed=fake_KL_divergence(transformed,target_samples);
    double fKLtransformedX=fake_KL_divergence(target_samples,transformed);

    //For comparison estimate statistics of redraws of samples (expected value is 0)
    double sum=0;
    int fntry=30*ntry;
    vector<double> KLdiffs(ntry),fKLdiffs(fntry);
    for(int i=0;i<ntry;i++){
      //draw alternative samples;
      vector<state> alt_samples=sample_target_dist(Nsamp);
      KLdiffs[i]=KL_divergence(alt_samples,target_samples,use_approxNN);
      //cout<<"("<<KLdiffs[i]<<","<<fKLdiffs[i]<<")"<<(i<ntry-1?",":"");
      //cout<<KLdiffs[i]<<(i<ntry-1?",":"");
    }
    //cout<<endl;
    cout<<"Computing 'fake' KL divergences:"<<endl;
    for(int i=0;i<fntry;i++){
      //draw alternative samples;
      vector<state> alt_samples=sample_target_dist(Nsamp);
      fKLdiffs[i]=fake_KL_divergence(alt_samples,target_samples);
      //cout<<"("<<KLdiffs[i]<<","<<fKLdiffs[i]<<")"<<(i<fntry-1?",":"");
    }
    //cout<<endl;
    //draw alternative samples;
    vector<state> alt_samples=sample_target_dist(Nsamp);
    double altKLtransformed=KL_divergence(transformed,alt_samples,use_approxNN);
    double altKLtransformedX=KL_divergence(alt_samples,transformed,use_approxNN);
    double altfKLtransformed=fake_KL_divergence(transformed,alt_samples);
    double altfKLtransformedX=fake_KL_divergence(alt_samples,transformed);
    for(auto &diff:KLdiffs)diff=fabs(diff);
    for(auto &diff:fKLdiffs)diff=fabs(diff);
    sort(KLdiffs.begin(),KLdiffs.end());
    sort(fKLdiffs.begin(),fKLdiffs.end());
    int Ncut=sqrt(ntry);
    double KLcut=0;
    double cut_frac=Ncut*1.0/ntry;
    if(ntry>1){
      KLcut=(KLdiffs[ntry-Ncut]+KLdiffs[ntry-Ncut-1])/2.0;
      //cout<<"KLdiffs";for(auto diff:KLdiffs)cout<<diff<<" ";cout<<endl;
      cout<<"\nEstimate that only "<<cut_frac*100<<"\% of KLdiff measurements are likely to exceed "<<KLcut<<" for matching distributions."<<endl;
    }
    cout<<"Transformed KLdiv="<<KLtransformed<<" <-> "<<KLtransformedX<<endl;

    cout<<"Alt-transformed KLdiv="<<altKLtransformed<<" <-> "<<altKLtransformedX<<endl;
    if(altKLtransformed-KLtransformed>fabs(KLcut))cout<<"When 'Transformed' is signficantly less than 'alt-Transformed' it may indicate that the effect of the proposal (after ncyc="<<ncyc<<" applications) is too small to measure at this level."<<endl;
    if(altKLtransformed<KLcut)cout<<"PASS"<<endl;
    else cout<<"FAIL\nKL value is "<<altKLtransformed/KLcut<<" times the stated threshold."<<endl;
    //Report fake KL tests
    Ncut=sqrt(fntry);
    KLcut=0;
    cut_frac=Ncut*1.0/fntry;
    if(fntry>1){
      //cout<<"fKLdiffs";for(auto diff:fKLdiffs)cout<<diff<<" ";cout<<endl;
      KLcut=(fKLdiffs[fntry-Ncut]+fKLdiffs[fntry-Ncut-1])/2.0;
      cout<<"\nEstimate that only "<<cut_frac*100<<"\% of fKLdiff measurements are likely to exceed "<<KLcut<<" for matching distributions."<<endl;
    }
    cout<<"Transformed fKLdiv="<<fKLtransformed<<" <-> "<<fKLtransformedX<<endl;

    cout<<"Alt-transformed fKLdiv="<<altfKLtransformed<<" <-> "<<altfKLtransformedX<<endl;
    if(altfKLtransformed-fKLtransformed>fabs(KLcut))cout<<"When 'Transformed' is signficantly less than 'alt-Transformed' it may indicate that the effect of the proposal (after ncyc="<<ncyc<<" applications) is too small to measure at this level."<<endl;
    if(altfKLtransformed<KLcut)cout<<"PASS"<<endl;
    else cout<<"FAIL\nKL value is "<<altfKLtransformed/KLcut<<" times the stated threshold."<<endl;
    //double result=altKLtransformed;
    //if(result<0)result=0;
    //result+=altfKLtransformed;
    return true;
  };
	
    double KL_divergence(vector<state> &samplesP,vector<state> &samplesQ,bool approx_nn=true){
    ///Interesting to apply to both target samples and transformed samples
    /// Implements eqn 2 of Perez-Cruz 2008 "Kullback-Leibler Divergence Estimation of Continuous Distributions"
    /// with k=1. We infer that the sign is wrong on the sum term there.
    ///Notes:
    /// -We see faster convergence when the Q set has the larger variance (more specifically when all points in P
    ///  are in regions with some Q support) as noted near Fig2 of Perez-Cruz.
    /// -We see faster runs for the approximate method when N>~3000. (On a laptop example eval takes a few 100ms there in 2D).
    /// -With k=2,target_recall=0.99 we get not much difference in biases for approx/exact (diff << stat err for 2D tests)
    /// -More tests varying d<=20 suggests crossing point is about N*d^0.7~4500
    /// -exact method cost seems to scale with n  and  with (1+d)           [near n~1000]
    /// -approx method cost seems to scale with n^1/2 and with (1+d)^.2
    int k=2;double target_recall=0.99;//For internal use of Mrpt;
    bool useMrptP=approx_nn,useMrptQ=approx_nn;
    double result=0;
    double N=samplesP.size();
    //cout<<"KL:about to call all_nnd2"<<endl;
    //vector<double> s1sqs=all_nnd2(samplesP);
    vector<double> s1sqs,r1sqs;
    if(useMrptP)r1sqs=all_nnd2_mrpt(samplesP,samplesP,true,k,target_recall);
    else r1sqs=all_nnd2(samplesP);
    if(useMrptQ)s1sqs=all_nnd2_mrpt(samplesP,samplesQ,false,k,target_recall);
    else {
      s1sqs.resize(N);
      for(int i=0;i<N;i++)s1sqs[i]=one_nnd2(samplesP[i],samplesQ);
    }
    if(useMrptQ and false){//testing
      vector<double>tests1sqs(N);
      for(int i=0;i<N;i++)tests1sqs[i]=one_nnd2(samplesP[i],samplesQ);
      //Copied/adapted from all_nnd2_mrpt::
      Eigen::MatrixXf X=get_X(samplesQ);
      Mrpt mrpt=get_mrpt(X,k);
      for(int i=0;i<N;i+=20){
	cout<<"test s1sqs["<<i<<"]: "<<tests1sqs[i]<<" -> "<<s1sqs[i]<<endl;
	//get i0 byt the non-mrpt method
	int i0;
	double nnd2=one_nnd2(samplesP[i],samplesQ,&i0);	
	cout<<"  "<<i0<<": "<<nnd2<<endl;
	//get i0 and two version of d^2 by mrpt method:
	i0=one_knn(samplesP[i],mrpt,k,0);
	nnd2=samplesP[i].dist2(samplesQ[i0]);
	vector<double> Pveci=samplesP[i].get_params_vector();
	double nnd2_raw=0;
	for(int j=0;j<Pveci.size();j++){
	  double diff=Pveci[j]-X(j,i0);
	  nnd2_raw+=diff*diff;
	}
	cout<<"  "<<i0<<": "<<nnd2_raw<<" -> "<<nnd2<<endl;
      }
    }
    //Here we put a floor on the smallest value of all NN distances
    //based on the kfloorth smallest distance within the P set
    int kfloor=5;
    if(kfloor>0){
      auto dists=r1sqs;
      sort(dists.begin(),dists.end());
      double floor=dists[kfloor+1];
      for(auto&d2:r1sqs)if(d2<floor)d2=floor;//this is pretty inefficient!
      for(auto&d2:s1sqs)if(d2<floor)d2=floor;//this is pretty inefficient!
    }

    for(int i=0;i<N;i++){
      //double r1sq=one_nnd2(samplesP[i],samplesQ);
      result+=-log(r1sqs[i]/s1sqs[i]);
      //cout<<i<<" "<<r1sqs[i]<<" "<<s1sqs[i]<<" -> "<<result<<endl;
    }
    double dim=samplesP[0].size();
    int M=samplesQ.size();
    result *= (0.5*dim)/N;//factor of 1/2 because we use nearest neighbor dist^2
    result += log(M/(N-1.0));
    //cout<<"KL about to return"<<endl;
    return result;

  };
  double one_nnd2(state &s,vector<state> &samples,int *out_i0=nullptr){//Computation of nearest neighbor distance to point, brute force.
    
    size_t N=samples.size();
    double nnd2=-1;
    int i0=-1;
    for(size_t i=0;i<N;i++){
      double dist2=samples[i].dist2(s);
      //double dist2ji=s.dist2(samples[i]);//debug
      //if(fabs((dist2-dist2ji)/dist2)>1e-14)cout<<"distance not symmetric "<<dist2<<"!="<<dist2ji<<endl;
      if(nnd2<0 or nnd2>dist2){
	i0=i;
	nnd2=dist2;
      }
      if(out_i0)*out_i0=i0;
      //cout<<"  "<<i<<": "<<nnd2<<endl;
    }
    return nnd2;
  };
  vector<double> all_nnd2(vector<state> &samples){//Computation of all nearest neighbor distances, brute force.
    size_t N=samples.size();
    vector<int> nni(N);
    vector<double> nnd2(N,-1);
    for(size_t i=0;i<N;i++){
      //cout<<"state i="<<i<<":"<<samples[i].show()<<endl;
      for(size_t j=i+1;j<N;j++){
	//cout<<"  state j="<<j<<":"<<samples[j].show()<<endl;
	double dist2=samples[i].dist2(samples[j]);
	//double dist2ji=samples[j].dist2(samples[i]);//debug
	//if(fabs((dist2-dist2ji)/dist2)>1e-14)cout<<"distance not symmetric "<<dist2<<"!="<<dist2ji<<endl;
	if(nnd2[i]<0 or nnd2[i]>dist2){
	  nni[i]=j;
	  nnd2[i]=dist2;
	}
	if(nnd2[j]<0 or nnd2[j]>dist2){
	  nni[j]=i;
	  nnd2[j]=dist2;
	}
      }
    }
    return nnd2;
  };
  
  double fake_KL_divergence(vector<state> &samplesP,vector<state> &samplesQ){
    //This applies a simplified alternative to the KL divergence (which is difficult to compute accurately from samples).
    //The calculation is based on the means and variances of the two samples and would agree with the KL diverences
    //for large samples of Gaussian distributions.
    Eigen::MatrixXd covP,covQ,invCovQ,covPinvCovQ;
    Eigen::VectorXd meanP,meanQ,dmu;
    covP=get_sample_cov(samplesP,meanP);
    covQ=get_sample_cov(samplesQ,meanQ);
    int dim=covP.rows();
    int nQ=samplesP.size();
    double unbiasing_factor=(nQ-dim-2.0)/(nQ-1.0);//The final factor is to make unbiased for finite nQ
    invCovQ=covQ.inverse()*unbiasing_factor;
    covPinvCovQ=covP*invCovQ;
    dmu=meanP-meanQ;
    double result=0;
    //cout<<"\n\ncovPdiag:\n"<<covP.diagonal()<<endl;
    //cout<<"\n\ncovQdiag:\n"<<covQ.diagonal()<<endl;
    //Eigen::MatrixXd covP2=covP*covP;
    //cout<<"\n\ncovP^2:\n"<<covP*covP<<endl;
    //cout<<"\n\ncovP^2:\n"<<covP2<<endl;
    //cout<<"|covP|^2:"<<(covP*covP).trace()<<endl;
    //cout<<"covPdiag: "<<covP.diagonal().transpose()<<endl;
    //cout<<"|covQ|^2:"<<(covQ*covQ).trace()<<endl;
    //cout<<"covQdiag: "<<covQ.diagonal().transpose()<<endl;
    //cout<<"invCovQdiag: "<<invCovQ.diagonal().transpose()<<endl;
    result += -dim + covPinvCovQ.trace();
    //cout<<"result A:"<<result<<endl;
    result += -log((covPinvCovQ/unbiasing_factor).determinant());  //If nP != nQ then we need another digamma-based term to make unbiased
    //cout<<"result AB:"<<result<<endl;
    result += dmu.transpose()*invCovQ*dmu - (dim + covPinvCovQ.trace())/nQ;
    //cout<<"result ABC:"<<result<<endl;
    
    return 0.5*result;
    //return 0.5*(covPinvCovQ.trace()-dim -log(covPinvCovQ.determinant()) +dmu.transpose()*invCovQ*dmu);    
  };
  
  Eigen::MatrixXf get_X(vector<state> &samples){
    size_t N=samples.size();;
    vector<double> statevec=samples[0].get_params_vector();
    int dim=statevec.size();
    Eigen::MatrixXf X(dim, N);
    Eigen::VectorXf v(dim);

    for(size_t i=0;i<N;i++){
      statevec=samples[i].get_params_vector();
      for(size_t j=0;j<dim;j++)
	X(j,i)=statevec[j];
    }
    return X;
  };
  
  Mrpt get_mrpt(Eigen::MatrixXf &X,int k=8,double recall=0.9){
    Mrpt mrpt(X);
    mrpt.grow_autotune(recall, k);
    return mrpt;
  };
  
  int one_knn(state &s,Mrpt &mrpt,int k=8,int ik=0){//Computation of nearest neighbor distance to point, brute force.
    //ik is index of which kth nearest neighbor to take (not nec same as how many are found)
    //note that when state is s among those used to define mrpt, then we often want to exclude that one, hence ik=1
    vector<double> statevec=s.get_params_vector();
    int dim=statevec.size();
    Eigen::VectorXf v(dim);
    
    for(size_t j=0;j<dim;j++) v(j)=statevec[j];
    
    int nni=-1,index[k];
    for(int i=0;i<=ik;i++)index[i]=-1;
    bool exact=false;
    //cout<<"k-"<<k<<endl;
    if(!exact){
      mrpt.query(v,index);
      nni=index[ik];
    }
    if(nni<0){//approx search failed
      mrpt.exact_knn(v, ik+1, index);
      nni=index[ik];
      //cout<<nni<<" ";
    }
    //cout<<nni<<endl;
    return nni;
  };
  
  vector<double>all_nnd2_mrpt(vector<state> &teststates,vector<state>&samples,bool same_set=false,int k=8,double target_recall=0.9,bool nonzero=true){
    //Computation of all nearest neighbor distances, using mrpt header-library.
    //This version doesn't know the parameter space topology in identifying nearest neighbors,
    //but does use the state.dist2 function for reporting the nearest distance.
    //
    Eigen::MatrixXf X=get_X(samples);
    Mrpt mrpt=get_mrpt(X,k,target_recall);
    size_t N=teststates.size();
    vector<double> nnd2(N);
    int ik=0;
    if(same_set)ik=1;
    if(k<=ik)k=ik+1;
    //cout<<k<<">"<<ik<<endl;
    for(size_t i=0;i<N;i++){
      double d2=0;
      size_t i0=one_knn(teststates[i],mrpt,k,ik);
      //cout<<"i,i0:"<<i<<" "<<i0<<endl;
      nnd2[i]=teststates[i].dist2(samples[i0]);
    }
    if(false and nonzero){//replace zeros with the next smallest value
      double eps=1e100;
      for(auto d2 :nnd2)if(d2<eps and d2>0)eps=d2;
      for(auto &d2 :nnd2)if(d2<eps)d2=eps;
    }
    return nnd2;
  };
  vector<double>all_nnd2_mrpt_v2_orig(vector<state> &samples){
    //Computation of all nearest neighbor distances, using mrpt header-library.
    //This version doesn't know the parameter space topology in identifying nearest neighbors,
    //but does use the state.dist2 function for reporting the nearest distance.
    int k=8;
    Eigen::MatrixXf X=get_X(samples);
    Mrpt mrpt=get_mrpt(X,k);
    size_t N=samples.size();
    vector<double> nnd2(N);
    for(size_t i=0;i<N;i++){
      size_t i0=one_knn(samples[i],mrpt,k,1);
      nnd2[i]=samples[i].dist2(samples[i0]);
    }
    return nnd2;
  };
  vector<double> get_sample_mean(const vector<state>&samples){
    size_t N=samples.size();
    size_t dim=samples[0].size();
    vector<double>sum(dim,0);
    for(auto s:samples)for(int j=0;j<dim;j++)sum[j]+=s.get_param(j);
    for(auto &x :sum)x/=N;
    return sum;
  };
  vector<double> get_sample_var(const vector<state>&samples){
    vector<double>mean;
    return get_sample_var(samples,mean);
  }
  vector<double> get_sample_var(const vector<state>&samples,vector<double>&out_mean){
    size_t N=samples.size();
    size_t dim=samples[0].size();
    out_mean=get_sample_mean(samples);
    vector<double>sum(dim,0);
    for(auto s:samples)for(int j=0;j<dim;j++){
	double diff=s.get_param(j)-out_mean[j];
	sum[j]+=diff*diff;
      }
    for(auto &x :sum)x/=(N-1.0);
    return sum;
  };
  Eigen::MatrixXd get_sample_cov(const vector<state>&samples){
    Eigen::VectorXd mean;
    return get_sample_cov(samples,mean);
  }
  Eigen::MatrixXd get_sample_cov(const vector<state>&samples, Eigen::VectorXd &out_mean){
    size_t N=samples.size();
    size_t dim=samples[0].size();
    Eigen::MatrixXd cov(dim,dim);
    cov = Eigen::MatrixXd::Zero(dim,dim);
    vector<double> mean=get_sample_mean(samples);
    out_mean=Eigen::VectorXd::Zero(dim);
    for(size_t i=0;i<dim;i++)out_mean(i)=mean[i];
    vector<double>sum(dim,0);
    for(auto s:samples)for(int j=0;j<dim;j++){
	double jdiff=s.get_param(j)-mean[j];
	cov(j,j)+=jdiff*jdiff;
	for(int i=j+1;i<dim;i++){
	  double idiff=s.get_param(i)-mean[i];
	  double val=idiff*jdiff;
	  cov(i,j)+=val;
	  cov(j,i)+=val; //Eigen is finicky so cov(j,i)=cov(i,j) doesn't work right!
	  if(cov(i,j)*cov(i,j)>1e50){
	    cout<<i<<","<<j<<":"<<idiff<<"*"<<jdiff<<"="<<val<<"->"<<cov(i,j)<<endl;
	}
	}
      }
    cov/=(N-1.0);
    return cov;
  };
    
};


void mrpt_knn() {
  int n = 10000, d = 200, k = 10;
  double target_recall = 0.9;
  Eigen::MatrixXf X = Eigen::MatrixXf::Random(d, n);
  Eigen::MatrixXf q = Eigen::VectorXf::Random(d);

  Eigen::VectorXi indices(k), indices_exact(k);

  Mrpt::exact_knn(q, X, k, indices_exact.data());
  std::cout << indices_exact.transpose() << std::endl;

  Mrpt mrpt(X);
  mrpt.grow_autotune(target_recall, k);

  mrpt.query(q, indices.data());
  std::cout << indices.transpose() << std::endl;
}

		
