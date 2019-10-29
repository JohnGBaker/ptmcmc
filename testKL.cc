#include "proposal_distribution.hh"
#include "probability_function.hh"
#include "test_proposal.hh"
#include <time.h>

int main(){
  int npar=3;
  
  //set space
  stateSpace space(npar);
  vector<string> names={"m1","m2","a1","a2","tc","dist","inc","phi","lamb","beta","psi"};
  space.set_names(names);

  //Add involution to space for a test, there are a couple examples:
  stateSpaceInvolution scrunch(space,"scrunch");
  scrunch.register_transformState(
				  [](const state &s,void *object, const vector<double> &randoms){
				    double x=s.get_param(0);
				    double y=s.get_param(1);
				    double r=sqrt(x*x+y*y);
				    double arg=atan2(y,x),ss;
				    state result=s;
				    if(abs(arg)>=M_PI/2)ss=1/2.0;
				    else if(abs(arg)>=M_PI/4)ss=2.0;
				    else ss=1;
				    x=r*cos(arg*ss);
				    y=r*sin(arg*ss);
				    result.set_param(0,x);	      
				    result.set_param(1,y);
				    return result;
				  });
  scrunch.register_jacobian(
			    [](const state &s,void *object, const vector<double> &randoms){
			      double x=s.get_param(0);
			      double y=s.get_param(1);
			      double r=sqrt(x*x+y*y);
			      double arg=atan2(y,x),ss;
			      double result=1;
			      if(abs(arg)>=M_PI/2)ss=1/2.0;
			      else if(abs(arg)>=M_PI/4)ss=2.0;
			      else ss=1;
			      //ss+=1;//Adding this hack leads to measureable error in the proposal 
			      return ss;
			      });
  
  stateSpaceInvolution random_rotation(space,"randrot",1);// 1 means need 1 random number
  random_rotation.register_transformState
    (
     [](const state &s,void *object, const vector<double> &randoms){
       state result=s;
       double x=s.get_param(0);
       double y=s.get_param(1);
       double r=sqrt(x*x+y*y);
       double phi=atan2(y,x);
       double dphi=M_PI*(randoms[0]);
       //dphi=dphi/2+0.03;//Including this hack breaks symmetry enough to be detected in the proposal test
       //cout<<"rotating by "<<dphi<<endl;
       phi+=dphi;
       x=r*cos(phi);
       y=r*sin(phi);
       result.set_param(0,x);	      
       result.set_param(1,y);
       return result;
     });

  space.addSymmetry(scrunch);
  space.addSymmetry(random_rotation);

  //set target dist
  double seed=clock()/(double)CLOCKS_PER_SEC;
  cout<<"seed="<<seed<<endl;
  ProbabilityDist::setSeed(seed);
  vector<double> cents={1,0,0};
  vector<double> sigmas={1,2,3};
  gaussian_dist_product dist(&space,cents,sigmas);

  cout<<"Testing scrunch involution."<<endl;
  int ntest=20;
  double testsum=0;
  for(int i=0;i<ntest;i++){
    state s=dist.drawSample(*ProbabilityDist::getPRNG());
    testsum+=scrunch.test_involution(s,1000000);
  }
  cout<<"RMS test_involution: "<<sqrt(testsum/ntest)<<endl;

  cout<<"Testing randrot involution."<<endl;
  testsum=0;
  for(int i=0;i<ntest;i++){
    state s=dist.drawSample(*ProbabilityDist::getPRNG());
    random_rotation.set_random(*ProbabilityDist::getPRNG());
    testsum+=random_rotation.test_involution(s,1000000);
  }
  cout<<"RMS test_involution: "<<sqrt(testsum/ntest)<<endl;

  
  //set proposal
  vector<double> propsigmas={0.1,0.1,0.1};
  //gaussian_prop prop(propsigmas);
  //draw_from_dist prop(propsigmas);
  //involution_proposal prop(scrunch);
  //involution_proposal prop(random_rotation);
  proposal_distribution_set prop=involution_proposal_set(space);
  cout<<"Proposal is:\n"<<prop.show()<<endl;
  
  //Setup and perform test
  test_proposal testprop(prop,dist);
  //double test(int Nsamp=10000, double ncyc=5, int ntry=10,double hastings_err=0)
  double KLval=testprop.test(10000,500,10,0.01*0);
  cout<<"KLval="<<KLval<<endl;
  return 0;
}
