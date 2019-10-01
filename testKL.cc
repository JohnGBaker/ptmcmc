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
  
  //set target dist
  double seed=clock()/(double)CLOCKS_PER_SEC;
  cout<<"seed="<<seed<<endl;
  ProbabilityDist::setSeed(seed);
  vector<double> cents={0,0,0};
  vector<double> sigmas={1,2,3};
  gaussian_dist_product dist(&space,cents,sigmas);
  
  //set proposal
  vector<double> propsigmas={0.1,0.1,0.1};
  gaussian_prop prop(propsigmas);
  //draw_from_dist prop(propsigmas);
  
  //Setup and perform test
  test_proposal testprop(prop,dist);
  //double test(int Nsamp=10000, double ncyc=5, int ntry=10,double hastings_err=0)
  double KLval=testprop.test(10000,500,10,0.01);
  cout<<"KLval="<<KLval<<endl;
  return 0;
}
