//Written by John G Baker NASA-GSFC (2017)
//This is to test the shaped gaussian proposal distribution.

#include "proposal_distribution.hh"
#include "chain.hh"
#include <Eigen/Eigen>

shared_ptr<Random> globalRNG;//used for some debugging... 

int main(int argc, char*argv[]){

  int ndim=5;
  int ndraws=2100000;
  double temp=10.;

  //Randomly construct a covariance matrix
  Eigen::MatrixXd cov(ndim,ndim);
  cov=Eigen::MatrixXd::Random(ndim,ndim);
  //cov.Random();        //draw a random matrix
  cov=cov.transpose()*cov; //and turn it into something symmetric and positive semidefinite.
  cov*=temp;
    
  //Next we define the proposal
  gaussian_prop p(cov);

  //We need a chain, to allow draws, just a dummy, but some setup is needed.
  ProbabilityDist::setSeed(.224);
  stateSpace ss(ndim);
  state s(&ss,vector<double>(ndim,0));
  gaussian_dist_product prior(&ss);
  MH_chain c(&prior,&prior);
  c.resetTemp(1/temp);

  //perform draws compute stats
  Eigen::MatrixXd covsum=Eigen::MatrixXd::Zero(ndim,ndim);
  Eigen::VectorXd sum=Eigen::VectorXd::Zero(ndim);
  Eigen::VectorXd min=Eigen::VectorXd::Constant(ndim,+1e100);
  Eigen::VectorXd max=Eigen::VectorXd::Constant(ndim,-1e100);
  valarray<double> statedata;
  valarray<double> statedata0;
  s.get_params_array(statedata0);
  Eigen::Map<Eigen::VectorXd> vec0(&statedata0[0],statedata0.size());
  state draw;
  
  for(int n=1;n<=ndraws;n++){
    draw=p.draw(s,&c);
    draw.get_params_array(statedata);
    Eigen::Map<Eigen::VectorXd> vec(&statedata[0],statedata.size());
    draw.get_params_array(statedata);
    //Eigen::Map<Eigen::VectorXd> vec0(&statedata0[0],statedata0.size());
    //cout<<"vec0:("<<vec0.rows()<<","<<vec0.cols()<<") vec:("<<vec.rows()<<","<<vec.cols()<<")"<<endl;
    Eigen::VectorXd dvec=vec-vec0;
    Eigen::MatrixXd covdelta(ndim,ndim);
    //cout<<"dvec=\n"<<dvec<<endl;
    sum+=dvec;
    for(int i=0;i<ndim;i++)for(int j=0;j<ndim;j++)covdelta(i,j)=dvec(i)*dvec(j);
    covsum+=covdelta;
    for(int i=0;i<ndim;i++){
      if(min(i)>vec(i))min(i)=vec(i);
      if(max(i)<vec(i))max(i)=vec(i);
    }
    if((n & (n - 1))==0){
      Eigen::MatrixXd coverr(ndim,ndim);
      coverr=covsum/n-cov;
      cout<<"cov=\n"<<coverr+cov<<endl;
      cout<<"coverr=\n"<<coverr<<endl;
      double coverrnorm=(coverr*coverr).trace();
      double coverrdiagnorm=0;
      for(int i=0;i<ndim;i++)coverrdiagnorm+=coverr(i,i)*coverr(i,i);
      cout<<"min:"<<min.transpose()<<endl;
      cout<<"max:"<<max.transpose()<<endl;
      cout<<n<<": errnorm="<<coverrnorm<<"\terrdiagnorm="<<coverrdiagnorm<<endl;
      
    }
  }
}

  
  
