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
#include <algorthm>

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
class test_proposal {
  shared_ptr<Random> rng;
  proposal_distribution &proposal;
  sampleable_probability_function &target;
public:
  //test_proposal(proposal_distribution &proposal, shared_ptr<Random> rng){cout<<"The idea is to provide a ready-made test distribution, perhaps a gaussian mixture, perhaps augmented by gaussians located at proposal-step images of the initial gaussian centers.  This is not yet implemented."<<endl;exit(-1);};
  test_proposal(proposal_distribution &proposal, sampleable_probability_function &target):
    rng(new MotherOfAll(ProbabilityDist::getPRNG()->Next())),
    proposal(proposal),
    target(target)
  {
    //Note: if proposal needs a chain, we may need to make one up with samples from the target dist.
  };
      
  test_proposal(){};//Trivial constructor for development and testing...
  vector<state> sample_target_dist(int nsamp){
    //make a set of samples from the target distribution
    vector<state> samples(nsamp);
    for( auto &sample : samples)sample=target.drawSample();
    return samples;
  };
  vector<state> transform_samples(vector<state> &samples, int & Naccept){
    //Apply the proposal via Metropolis-Hastings step, to each sample and return result.
    //Number of acceptances goes in Naccept
    Naccept=0;
    size_t N=samples.size();
    vector<state> transformed(N);
    
    for(size_t i;i<N;i++){
      const state &oldstate=samples[i];
      double oldlpdf=target->evaluate_log(oldstate);
      state newstate=prop.draw(current_state,nullptr); //If prop needs a chain then this won't work yet.
      double newlpdf=target->evaluate_log(newstate);
      
      //Now the test: 
      double log_hastings_ratio=prop.log_hastings_ratio();
      log_hastings_ratio+=newlpdf-oldlpdf;
      bool accept=true;
      if(newstate.invalid())accept=false;
      if(accept and log_hastings_ratio<0){
	double x=get_uniform(); //pick a number
	accept=(log(x)<log_hastings_ratio);
      }
      
      if(accept){
	Naccept++;
	prop.accept(); //As with MCMC, this allows some proposals to 'adapt'
      }
      else {
	prop.reject();
	add_state(current_state,current_llike,current_lpost);
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
  double test(int Nsamp=10000, double ncyc=5, int ntry=10){
    //Draw reference samples
    vector<state> target_samples=sample_target_dist(Nsamp);

    //Transform by application of the proposal
    int Naccept_cum=0;
    int Ngoal=ncnc*Nsamp;
    vector<state> transformed=target_samples;
    while(Naccept_cum<Ngoal){
      int Naccept=0;
      vector<state> transformed=transform_samples(transformed, Naccept);
      Naccept_cum+=Naccept;
      cout<<"Accept rate: "<<100.0*Naccept/Nsamp<<"\% Naccepted, Ngoal:"<< Naccept_cum<<", "<<Ngoal<<"  Percent done ="<<100.0*Naccept_cum/Ngoal<<endl;
    }

    //Compute KL divergence
    //First decide whether to use approx NN:
    double dim=samplesP[0].size();
    bool use_approxNN=(Nsamp*pow(dim,0.7)>4500); //estimate of when approx is faster
    double KLtransformed=KL_divergence(transformed,target_samples,use_approxNN);

    //For comparison estimate statistics of redraws of samples (expected value is 0)
    double sum=0;
    vector<double> KLdiffs(Ntry);
    for(int i=0;i<Ntry;i++){
      //draw alternative samples;
      vecgort<state> alt_samples=sample_target_dist(Nsamp);
      KLdiffs[i]=KLtransformed(transformed,target_samples,use_approxNN);
      cout<<KLdiff<<",";
    }
    cout<<endl;
    sort(KLdiffs.begin(),KLdiffs.end());
    int Ncut=sqrt(Ntry);
    if(Ntry>1){
      double KLcut=(KLdiffs[Ncut]+KLdiffs[Ncut+1])/2.0;
      double cut_frac=1-Ncut*1.0/Ntry;
      cout<<"Estimate that only "<<cut_frac*100<<"\% of KLdiff measurements are likely to exceed "<<KLcut<<" for matching distributions."<<endl;
    }
    cout<<"Transformed KLdiv="<<KLtransformed<<endl;
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
  
  vector<double>all_nnd2_mrpt(vector<state> &teststates,vector<state>&samples,bool same_set=false,int k=8,double target_recall=0.9){
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
      size_t i0=one_knn(teststates[i],mrpt,k,ik);
      //cout<<"i,i0:"<<i<<" "<<i0<<endl;
      nnd2[i]=teststates[i].dist2(samples[i0]);
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

		
