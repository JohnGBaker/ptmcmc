#include "ProbabilityDist.h"
#include <sstream>
#include <fstream>
#include "newran.h"

using namespace std;

double fsig(const valarray<double> &vx){double x=vx[0];cout<<"fsig:x="<<x<<endl;return x*x;};
double fcent(const valarray<double> &vx){double x=vx[0];cout<<"fcent:x="<<x<<endl;return x*10;};

int main(int argc, char*argv[]){
  double seed;
  if(argc!=2||!(istringstream(argv[1])>>seed)){
    cout<<"argc="<<argc<<endl;
    if(argc>1)cout<<"argv[1]="<<argv[1]<<endl;
    cout<<"Usage testProbDist seed\n(seed) should bin in [0,1)."<<endl;
    return -1;
  }
  cout<<"seed="<<seed<<endl;
  ProbabilityDist::setSeed(seed);

  int Nbin=80;
  ProbabilityDist *dist;
  UniformIntervalDist uni(0.1,43.2);
  UniformLogDist lg(0.31,3.1e2);
  UniformPolarDist pol;
  UniformCoPolarDist copol;
  GaussianDist gauss(32.3,6.8);
  GaussianFunctionDist fgauss(&fcent,&fsig,1);


  int n,nn;
  dist=&uni;
  cout<<dist->show()<<endl;
  for(int i=0;i<5;i++){
    ostringstream ss;
    n=(2<<i)*Nbin;
    double err=testProbabilityDist(*dist,n,ss);		   
    cout<<n<<" bins : sigma  ="<<err<<endl;
    nn=(2<<i)*Nbin;
    err=testProbabilityDistIntegral(*dist,nn);		   
    cout<<nn<<" step : int err="<<err<<endl;
  }
  ofstream *file=new ofstream("uniform.dat");
  testProbabilityDist(*dist,n,*file);		   
  delete file;

  dist=&lg;
  cout<<dist->show()<<endl;
  for(int i=0;i<6;i++){
    ostringstream ss;
    n=(2<<i)*Nbin;
    double err=testProbabilityDist(*dist,n,ss);		   
    cout<<n<<" bins : sigma="<<err<<endl;
    nn=(2<<i)*Nbin;
    err=testProbabilityDistIntegral(*dist,nn);		   
    cout<<nn<<" step : int err="<<err<<endl;
  }
  file=new ofstream("log.dat");
  testProbabilityDist(*dist,n,*file);		   
  delete file;

  dist=&gauss;
  cout<<dist->show()<<endl;
  for(int i=0;i<5;i++){
    ostringstream ss;
    n=(2<<i)*Nbin;
    double err=testProbabilityDist(*dist,n,ss);		   
    cout<<n<<" bins : sigma="<<err<<endl;
    nn=(2<<i)*Nbin;
    err=testProbabilityDistIntegral(*dist,nn);		   
    cout<<nn<<" step : int err="<<err<<endl;
  }
  file=new ofstream("gauss.dat");
  testProbabilityDist(*dist,n,*file);		   
  delete file;

  dist=&pol;
  cout<<dist->show()<<endl;
  for(int i=0;i<7;i++){
    ostringstream ss;
    n=(2<<i)*Nbin;
    double err=testProbabilityDist(*dist,n,ss);		   
    cout<<n<<" bins : sigma="<<err<<endl;
    nn=(2<<i)*Nbin;
    err=testProbabilityDistIntegral(*dist,nn);		   
    cout<<nn<<" step : int err="<<err<<endl;
  }
  file=new ofstream("polar.dat");
  testProbabilityDist(*dist,n,*file);		   

  dist=&copol;
  cout<<dist->show()<<endl;
  for(int i=0;i<7;i++){
    ostringstream ss;
    n=(2<<i)*Nbin;
    double err=testProbabilityDist(*dist,n,ss);		   
    cout<<n<<" bins : sigma="<<err<<endl;
    nn=(2<<i)*Nbin;
    err=testProbabilityDistIntegral(*dist,nn);		   
    cout<<nn<<" step : int err="<<err<<endl;
  }
  file=new ofstream("copolar.dat");
  testProbabilityDist(*dist,n,*file);		   

  dist=&fgauss;
  valarray<double>given(0.75,1);
  //cout<<"given[0]="<<given[0]<<endl;
  dist->setGivens(given);
  cout<<dist->show()<<endl;
  for(int i=0;i<5;i++){
    ostringstream ss;
    n=(2<<i)*Nbin;
    double err=testProbabilityDist(*dist,n,ss);		   
    cout<<n<<" bins : sigma="<<err<<endl;
    nn=(2<<i)*Nbin;
    err=testProbabilityDistIntegral(*dist,nn);		   
    cout<<nn<<" step : int err="<<err<<endl;
  }
  file=new ofstream("function.dat");
  testProbabilityDist(*dist,n,*file);		   
}
