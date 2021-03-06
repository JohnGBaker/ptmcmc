//Simplified likelihood for LISA example.  The simplified likelihood covers only
//extrinsic parameters based on  low-f limit, and short-duration observation
//as occurs for merger of ~1e6 Msun binaries. 
//Written by Sylvain Marsat AEI and John G Baker NASA-GSFC (2017)

//#include "mlfit.hh"
#include <valarray>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <complex>
#include "omp.h"
#include "options.hh"
//#include <mcheck.h>
#include "bayesian.hh"
#include "proposal_distribution.hh"
#include "ptmcmc.hh"

using namespace std;

const double MTSUN_SI=4.9254923218988636432342917247829673e-6;
const double PI=3.1415926535897932384626433832795029;
const complex<double> I(0.0, 1.0);
const bool narrowband=false;

//We implement 4 different example interfaces to bayes_likelihood:
// 1. no_class
// 2. !noclass&!use_inheritance_interface
// 3. !noclass&use_inheritance_interface&use_basic_setup
// 4. !noclass&use_inheritance_interface&!use_basic_setup
// They are progressively more formal and intimately connected to bayes_likelihood.
// Option 4. was the original before June 2019.

const bool no_class=false;
const bool use_inheritance_interface=false; //only relevant if no_class=false
const bool use_basic_setup=false; //only relevant with use_inheritance_interface=true

// Routines for simplified likelihood 22 mode, frozen LISA, lowf, fixed masses (near 1e6) and fixed t0
static double funcphiL(double m1, double m2, double tRef,double phiRef){
  double MfROMmax22 = 0.14;
  double fRef = MfROMmax22/((m1 + m2) * MTSUN_SI);
  return -phiRef + PI*tRef*fRef;
}
static double funclambdaL(double lambd, double beta) {
  return -atan2(cos(beta)*cos(lambd)*cos(PI/3) + sin(beta)*sin(PI/3), cos(beta)*sin(lambd));
}
static double funcbetaL(double lambd, double beta) {
  return -asin(cos(beta)*cos(lambd)*sin(PI/3) - sin(beta)*cos(PI/3));
}
static double funcpsiL(double lambd, double beta, double psi) {
  return atan2(cos(PI/3)*cos(beta)*sin(psi) - sin(PI/3)*(sin(lambd)*cos(psi) - cos(lambd)*sin(beta)*sin(psi)), cos(PI/3)*cos(beta)*cos(psi) + sin(PI/3)*(sin(lambd)*sin(psi) + cos(lambd)*sin(beta)*cos(psi)));
}
static complex<double> funcsa(double d, double phi, double inc, double lambd, double beta, double psi) {
  complex<double> Daplus = I * (double)( 3./4 * (3 - cos(2*beta)) * cos(2*lambd - PI/3) );
  complex<double> Dacross = I * (3.0*sin(beta) * sin(2*lambd - PI/3));
  complex<double> a22 = 0.5/d * sqrt(5/PI) * pow(cos(inc/2), 4) * exp(2.*I*(-phi-psi)) * 0.5*(Daplus + I*Dacross);
  complex<double> a2m2 = 0.5/d * sqrt(5/PI) * pow(sin(inc/2), 4) * exp(2.*I*(-phi+psi)) * 0.5*(Daplus - I*Dacross);
  return a22 + a2m2;
}
static complex<double> funcse(double d, double phi, double inc, double lambd, double beta, double psi) {
  complex<double> Deplus = -I*(3./4 * (3 - cos(2*beta)) * sin(2*lambd - PI/3));
  complex<double> Decross = I*(3*sin(beta) * cos(2*lambd - PI/3));
  complex<double> e22 = 0.5/d * sqrt(5/PI) * pow(cos(inc/2), 4) * exp(2.*I*(-phi-psi)) * 0.5*(Deplus + I*Decross);
  complex<double> e2m2 = 0.5/d * sqrt(5/PI) * pow(sin(inc/2), 4) * exp(2.*I*(-phi+psi)) * 0.5*(Deplus - I*Decross);
  return e22 + e2m2;
}

vector<double> dummy_Fisher_cov(){
  vector<double> cov={1,0,-0.25,3e-5,0,0.09};
  return cov;
}
  
vector<double> partial_simple_Fisher_cov(double d, double inc, double dscale,double phiscale,double incscale){
  //Here we compute the simple likelihood partial Fisher matrix for just the common cofactor of a/e parts/
  //This is surely not useful, and likely not even correct, but meant to allow an example implementation.
  //cofactor: f = 0.5/d * sqrt(5/PI) * pow(cos(inc/2), 4) * exp(2.*I*(-phi))
  //The raw fisher matrix then is:
  //                  [ a^2  ab   acX ] 
  // F_raw  =  rho^2  [ ab   b^2  bcX ]
  //                  [ acX  bcX  c^2 ]
  //where:
  // a = dlnf/dd, b = dlnf/dinc, c = dlnf/d(I*phi)
  // The X factor comes in where there is a real and imaginary derivatives
  //The raw F is signular, so we add to it scale matrix diag(1/La^2,1/Lb^2,1/Lc^2) to ensure reasonable finite eigenvalues.
  //For this small matrix we can analytically compute the covariance.  We write it in terms of
  //alphavec=[alpha,beta,gamma]=rho*[La*a, Lb*b, Lc*c]
  //Note hereafter we swap the order of the last two rows/columns
  double factor = 216147.866077;
  complex<double> sainj = 0.33687296665053773 + I*0.087978055005482114;
  complex<double> seinj = -0.12737105239204741 + I*0.21820079314765678;
  double rho2 = factor * (pow(abs(sainj), 2) + pow(abs(seinj), 2));
  double rho=sqrt(rho2);
  double Xcoeff = factor * imag(pow(sainj, 2) + pow(seinj, 2))/rho2;
  double X2=Xcoeff*Xcoeff;
  double dlogfac_d_R   = -1/d;
  double dlogfac_inc_R = -4*tan(inc/2);
  double dlogfac_phi_I = -2;
  double alpha=dlogfac_d_R*dscale*rho;
  double beta=dlogfac_inc_R*dscale*rho;
  double gamma=dlogfac_phi_I*dscale*rho;
  vector<double> alphavec={alpha,gamma,beta};
  vector<double> scales={dscale,phiscale,incscale};
  double ab2=alpha*alpha+beta*beta;
  double g2=gamma*gamma;
  double det=1+ab2+g2+ab2*g2*(1-X2);
  double RRfac=(1+g2*(1-X2))/det;
  double IIfac=(1+ab2*(1-X2))/det;
  double RIfac=Xcoeff/det;
  vector<double> cov(6);
  int icov=0;
  for(int i=0;i<3;i++){
    for(int j=i;j<3;j++){
      double coeff=-alphavec[i]*alphavec[j];
      int nI=(i==1)+(j==1);//This essentially counts the number of times one index is for phi, to work out which sector
      if(nI==0)coeff*=RRfac;
      else if(nI==1)coeff*=RIfac;
      else coeff*=IIfac;
      if(i==j)coeff+=1;
      coeff*=scales[i]*scales[j];
      cov[icov]=coeff;
      icov++;
    }
  }
  return cov;
};


double simpleCalculateLogLCAmpPhase(double d, double phiL, double inc, double lambdL, double betaL, double psiL) 
{
  //Simple likelihood for runcan 22 mode, frozen LISA, lowf, snr 200
  // normalization factor and injection values sainj, seinj hardcoded - read from Mathematica
  double factor = 216147.866077;
  complex<double> sainj = 0.33687296665053773 + I*0.087978055005482114;
  complex<double> seinj = -0.12737105239204741 + I*0.21820079314765678;
  //double d = params->distance / injectedparams->distance;
  complex<double> sa = funcsa(d, phiL, inc, lambdL, betaL, psiL);
  complex<double> se = funcse(d, phiL, inc, lambdL, betaL, psiL);
  double simplelogL = -1./2 * factor * (pow(abs(sa - sainj), 2) + pow(abs(se - seinj), 2));
  return simplelogL;
}



///Functions implementing potential parameter space symmetries
///This functions are of a standard form needed for specifying (potential)
///symmetries of the parameter state space, and can be exploited as
///specialized MCMC proposals.

//TDI A/E symmetric (in stationary/low-freq limit) half rotation of constellation or quarter rotation with polarization flip
//uses 1 random var
state LISA_quarter_rotation_symmetry(void *object, const state &s, const vector<double> &randoms){
  state result=s;
  int nrot=int(fabs(randoms[0])*2)+1;
  if(randoms[0]<0)nrot=-nrot;
  //cout<<"nrot="<<nrot<<endl;
  int ilambda=3,ipsi=5;
  double lamb=s.get_param(ilambda);
  double psi=s.get_param(ipsi);
  const double halfpi=M_PI/2;
  lamb+=nrot*halfpi;
  if(abs(nrot)%2==1)psi+=halfpi;
  result.set_param(ilambda,lamb);
  result.set_param(ipsi,psi);
  return result;
};

//Source (even l dominant) half rotation symmetry or quarter rotation with polarization flip
//uses 1 random var
state source_quarter_rotation_symmetry(void *object, const state &s, const vector<double> &randoms){
  state result=s;
  int nrot=int(fabs(randoms[0])*2)+1;
  if(randoms[0]<0)nrot=-nrot;
  int iphi=1,ipsi=5;
  double phi=s.get_param(iphi);
  double psi=s.get_param(ipsi);
  const double halfpi=M_PI/2;
  phi+=nrot*halfpi;
  if(abs(nrot)%2==1)psi+=halfpi;
  result.set_param(iphi,phi);
  result.set_param(ipsi,psi);
  return result;
};

void show_test(const state s){
  valarray<double>params=s.get_params();
  double d=params[0];
  double phi=params[1];
  double inc=params[2];
  double lambd=params[3];
  double beta=params[4];
  double psi=params[5];
  complex<double> sa= funcsa(d, phi, inc, lambd, beta, psi);
  complex<double> se= funcse(d, phi, inc, lambd, beta, psi);
  cout<<"state: "<<s.get_string()<<endl;
  cout<<"   sa: "<<sa<<" mod:"<<abs(sa)<<" arg:"<<arg(sa)<<endl;
  cout<<"   se: "<<se<<" mod:"<<abs(se)<<" arg:"<<arg(se)<<endl;
};


//TDI A/E symmetric (in stationary/low-freq limit) relection through constellation plane, simultaneous with source plane reflection and polarization flip
//uses 0 random vars
state LISA_plane_reflection_symmetry(void *object, const state &s, const vector<double> &randoms){
  state result=s;
  int nrot=1;if(randoms[0]<0)nrot=-1;
  int iinc=2,ibeta=4,ipsi=5;
  double beta=s.get_param(ibeta);
  double psi=s.get_param(ipsi);
  double inc=s.get_param(iinc);
  inc=M_PI-inc;
  beta=-beta;
  psi=M_PI-psi;
  result.set_param(iinc,inc);
  result.set_param(ibeta,beta);
  result.set_param(ipsi,psi);
  return result;
};


state transmit_receive_inc_swap_symmetry(void *object, const state &s, const vector<double> &randoms){
  const double halfpi=M_PI/2;
  state result=s;
  int iphi=1,iinc=2,ilamb=3,ibeta=4,ipsi=5;
  vector<double> params=s.get_params_vector();
  double phi=s.get_param(iphi);
  double inc=s.get_param(iinc);
  double lamb=s.get_param(ilamb);
  double theta=halfpi-s.get_param(ibeta);
  //double fourpsi=4*s.get_param(ipsi);
  double twopsi=2*s.get_param(ipsi);
  double ti4=pow(tan(inc/2),4);
  double tt4=pow(tan(theta/2),4);
  double Phi=atan2(sin(twopsi)*(ti4-tt4),cos(twopsi)*(ti4+tt4))/2;
  //cout<<"Phi="<<Phi<<" ti4="<<ti4<<" tt4="<<tt4<<endl;
  result.set_param(iinc,theta);
  result.set_param(ibeta,halfpi-inc);
  result.set_param(iphi,phi-Phi);//sign differs from that in notes
  result.set_param(ilamb,lamb-Phi);
  //cout<<"before:"<<endl;
  //show_test(s);
  //cout<<"after:"<<endl;
  //show_test(result);
  return result;
};

//Approximate distance inclination symmetry
//uses 2 random var
double dist_inc_jump_size=0.1;
double maxf2=100;
state dist_inc_scale_symmetry(void *object, const state &s, const vector<double> &randoms){
  state result=s;
  //We apply a symmetry to preserve d'*F(x')=d*F(x) where F(x)=1/cos(x)^2
  //Depending on the sign of the second random number, x is either the source inclination, or the
  //line-of-sight inclination relative to the LISA plane, theta=pi/2-beta;
  //To avoid issues at the edges we make sure that the transformation of the inclination
  //never crosses its limits.
  //Note that f:x->ln(pi/x-1) has range (inf,-inf) on domain (0,pi) with f(pi-x)=-f(x)
  //and inverse pi/(exp(f(x))+1)=x
  //We then step uniformly in f(x). So, for random number y,
  // x'=finv(f(x)+y)
  int idist=0,iinc=2;
  bool use_theta=false;
  if(abs(randoms[1]*2)<1){ //Half of the time we apply the transformation to theta (LISA includination) rather than source inclination
    use_theta=true;
    iinc=4;
  }
  double dist=s.get_param(idist);
  double df=randoms[0]*dist_inc_jump_size; //Uniformly transform reparameterized inc
  double oldinc=s.get_param(iinc);
  if(use_theta)oldinc=M_PI/2-oldinc;//convert beta to theta
  double oldf=log(M_PI/oldinc-1);
  double newf=oldf+df;
  //if(newf*newf>maxf2)newf=oldf-df;//Avoid drifting into numerically dubious region in reversible way
  double newinc=M_PI/(exp(newf)+1);
  double cosold=cos(oldinc),cosnew=cos(newinc);
  double fac=cosnew/cosold;
  //double fac=(cosnew*cosnew+1)/(cosold*cosold+1);
  dist=dist*fac;
  result.set_param(idist,dist);
  if(use_theta)newinc=M_PI/2-newinc;//convert back to beta if needed
  result.set_param(iinc,newinc);
  double t4=pow(tan(oldinc/2),4);
  if(false and (t4<0.02 or t4>50)){
    cout<<"small/large t4 --------"<<t4<<endl;
    cout<<"use_theta="<<use_theta<<endl;
    cout<<"fac = 1 + "<<fac-1<<" = "<<fac;
    cout<<"before:"<<endl;
    show_test(s);
    cout<<"after:"<<endl;
    show_test(result);
  }
  return result;
};
//Approximate distance inclination symmetry jacobian
//uses 1 random var
double dist_inc_scale_symmetry_jac(void *object, const state &s, const vector<double> &randoms){
  //The transformation has the form:
  //  d' = d F(x)/F'(x')
  //  x' = finv( f(x) + y )
  //  y' = -y
  //where x is the selected inclination variable and y is the random number.
  //The Jacobian is then -F(x)f'(x) / (F(x')f'(x'))
  //Because the random step is performed on the rescaled inclination f(x)=ln(pi/x-1)
  //we have 1/f'(x) = x(1-x/pi)
  state result=s;
  int iinc=2;
  bool use_theta=false;
  if(abs(randoms[1]*2)<1){ //Half of the time we apply the transformation to theta (LISA includination) rather than source inclination
    use_theta=true;
    iinc=4;
  }
  double df=randoms[0]*dist_inc_jump_size; //Uniformly transform reparameterized inc
  double oldinc=s.get_param(iinc);
  if(use_theta)oldinc=M_PI/2-oldinc;//convert beta to theta 
  double oldf=log(M_PI/oldinc-1);
  double newf=oldf+df;
  //if(newf*newf>maxf2)newf=oldf-df;//Avoid drifting into numerically dubious region in reversible way
  double newinc=M_PI/(exp(newf)+1);
  double cosold=cos(oldinc),cosnew=cos(newinc);
  double fac=cosnew/cosold;
  //double fac=(cosnew*cosnew+1)/(cosold*cosold+1);
  fac*=(M_PI-newinc)*newinc/(M_PI-oldinc)/oldinc;
  if(std::isnan(fac)){
    cout<<"fac is nan:"<<oldf<<"->"<<newf<<" "<<oldinc<<"->"<<newinc<<" --> fac="<<fac<<endl;
    cout<<"state: "<<s.get_string()<<endl;
  }
  return fabs(fac); 
};


///Likelihood function interface variants
///

// Least intimate interface for Simplified LISA likelihood not relying on any class
// Since are is no data, we don't even need a reference object. A struct or vector or
// other object could be set up if the likelihood needs some data.
// no_class=true option
double simple_likelihood_evaluate_log_nc(void *object, const state &s){
  valarray<double>params=s.get_params();
  double d=params[0];
  double phi=params[1];
  double inc=params[2];
  double lambd=params[3];
  double beta=params[4];
  double psi=params[5];
  double result=simpleCalculateLogLCAmpPhase(d, phi, inc, lambd, beta, psi);
  //static int count=0;
  //cout<<count<<endl;
  //count++;
  //cout<<"state: "<<s.get_string()<<endl;
  //cout<<"  logL="<<result<<endl;
  
  return result;
};

void simple_likelihood_setup_nc(bayes_likelihood *blike){
  //blike->register_reference_object(*some_object);  //Use this if likelihood needs some data
  blike->register_evaluate_log(simple_likelihood_evaluate_log_nc);
  
  //setup  stateSpace
  int npar=6;
  stateSpace space(npar);
  string names[]={"d","phi","inc","lambda","beta","psi"};
  space.set_names(names);
  space.set_bound(1,boundary(boundary::wrap,boundary::wrap,0,2*M_PI));//set 2-pi-wrapped space for phi.  Turn on if not narrow banding
  if(not narrowband){
    space.set_bound(0,boundary(boundary::limit,boundary::limit,0,30));//set 2-pi limited space for lambda.
    space.set_bound(2,boundary(boundary::limit,boundary::limit,0,M_PI));//set 2-pi limited space for lambda.
    space.set_bound(3,boundary(boundary::wrap,boundary::wrap,0,2*M_PI));//set 2-pi-wrapped space for lambda.
    space.set_bound(4,boundary(boundary::limit,boundary::limit,-M_PI/2,M_PI/2));//set limited space for beta.
  }
  //else space.set_bound(4,boundary(boundary::limit,boundary::limit,0.2,0.6));//set narrow limits for beta
  space.set_bound(5,boundary(boundary::wrap,boundary::wrap,0,M_PI));//set pi-wrapped space for pol.
  
  vector<double> centers((initializer_list<double>){  1.667,  PI, PI/2,  PI,    0, PI/2});
  vector<double>  scales((initializer_list<double>){  1.333,  PI, PI/2,  PI, PI/2, PI/2});
  vector<string>      types((initializer_list<string>){  "uni","uni", "pol", "uni", "cpol", "uni"});
  if(narrowband){
    centers[3] = 1.75*PI;scales[3] = PI/4.0;
    //centers[4] = 0.4    ;scales[4] = 0.2;
    centers[4] = PI/4    ;scales[4] = PI/4;
  }
  //cout<<"simple_likelihood::setup: space="<<space.show()<<endl;
  
  blike->basic_setup(&space, types, centers, scales);
};
  
///Alternative interface using the original class-inheritance based interface for defining
///the likelihood.  There are two variants allowing bayes_likelihood to do differing degrees of the
///work in defining the prior particularly.
class simple_likelihood : public bayes_likelihood {
  int idx_phi,idx_d,idx_inc,idx_lambda,idx_beta,idx_psi;
public:
  //simple_likelihood():bayes_likelihood(nullptr,nullptr,nullptr){};
  simple_likelihood():bayes_likelihood(){};
  virtual void setup(){    
    ///Set up the output stateSpace for this object

    //setup  stateSpace
    int npar=6;
    stateSpace space(npar);
    string names[]={"d","phi","inc","lambda","beta","psi"};//double phiRef, double d, double inc, double lambd, double beta, double psi) 
    space.set_names(names);
    space.set_bound(1,boundary(boundary::wrap,boundary::wrap,0,2*M_PI));//set 2-pi-wrapped space for phi.  Turn on if not narrow banding
    if(not narrowband){
      space.set_bound(0,boundary(boundary::limit,boundary::limit,0,30));//set 2-pi limited space for lambda.
      space.set_bound(2,boundary(boundary::limit,boundary::limit,0,M_PI));//set 2-pi limited space for lambda.
      space.set_bound(3,boundary(boundary::wrap,boundary::wrap,0,2*M_PI));//set 2-pi-wrapped space for lambda.
      space.set_bound(4,boundary(boundary::limit,boundary::limit,-M_PI/2,M_PI/2));//set limited space for beta.
    }
    //else space.set_bound(4,boundary(boundary::limit,boundary::limit,0.2,0.6));//set narrow limits for beta
    space.set_bound(5,boundary(boundary::wrap,boundary::wrap,0,M_PI));//set pi-wrapped space for pol.
    if(use_basic_setup){

      vector<double> centers((initializer_list<double>){  1.667,  PI, PI/2,  PI,    0, PI/2});
      vector<double>  scales((initializer_list<double>){  1.333,  PI, PI/2,  PI, PI/2, PI/2});
      vector<string>      types((initializer_list<string>){  "uni","uni", "pol", "uni", "cpol", "uni"});
      if(narrowband){
	centers[3] = 1.75*PI;scales[3] = PI/4.0;
	//centers[4] = 0.4    ;scales[4] = 0.2;
	centers[4] = PI/4    ;scales[4] = PI/4;
      }
     //cout<<"simple_likelihood::setup: space="<<space.show()<<endl;

      basic_setup(&space, types, centers, scales);
    } else {
      haveSetup();
      //cout<<"Parameter space:\n"<<space.show()<<endl;
      
      nativeSpace=space;
      defWorkingStateSpace(nativeSpace);
      //cout<<"simple_likelihood: Setting best stateSpace="<<&nativeSpace<<endl;
      best=state(&nativeSpace,space.size());
      //Set the prior...
      
      const int uni=mixed_dist_product::uniform, gauss=mixed_dist_product::gaussian, pol=mixed_dist_product::polar, cpol=mixed_dist_product::copolar, log=mixed_dist_product::log; 
      valarray<double> centers((initializer_list<double>){  1.667,  PI, PI/2,  PI,    0, PI/2});
      valarray<double>  scales((initializer_list<double>){  1.333,  PI, PI/2,  PI, PI/2, PI/2});
      valarray<int>      types((initializer_list<int>)   {    uni, uni, pol, uni, cpol,  uni});
      if(narrowband){
	centers[3] = 1.75*PI;scales[3] = PI/4.0;
	//centers[4] = 0.4    ;scales[4] = 0.2;
	centers[4] = PI/4    ;scales[4] = PI/4;
      }
      setPrior(new mixed_dist_product(&nativeSpace,types,centers,scales));
    }
    //cout<<"simple_likelihood::setup:this="<<this<<endl;
    //cout<<"simple_likelihood::setup:&space="<<getObjectStateSpace()<<endl;
    //cout<<"simple_likelihood::setup:space="<<getObjectStateSpace()->show()<<endl;
    //cout<<"best stateSpace is:"<<best.getSpace()<<endl;
    //cout<<"simple_likelihood::setup:best="<<best.show()<<endl;

  };
  void defWorkingStateSpace(const stateSpace &sp){
    checkSetup();//Call this assert whenever we need options to have been processed.
    idx_d=sp.requireIndex("d");
    idx_phi=sp.requireIndex("phi");
    idx_inc=sp.requireIndex("inc");
    idx_lambda=sp.requireIndex("lambda");
    idx_beta=sp.requireIndex("beta");
    idx_psi=sp.requireIndex("psi");
    haveWorkingStateSpace();
  };
  //int size()const{return 0;};
  double evaluate_log(state &s){
    valarray<double>params=s.get_params();
    double d=params[idx_d];
    double phi=params[idx_phi];
    double inc=params[idx_inc];
    double lambd=params[idx_lambda];
    double beta=params[idx_beta];
    double psi=params[idx_psi];
    double result=simpleCalculateLogLCAmpPhase(d, phi, inc, lambd, beta, psi);
    double post=result;
    post+=nativePrior->evaluate_log(s);//May need a mechanism to check that prior is set
    #pragma omp critical
    {     
      if(post>best_post){
        best_post=post;
        best=state(s);
      }
      if(!isfinite(post)){
        cout<<"Whoa dude, logpost is NAN! What's up with that?"<<endl;
        cout<<"params="<<s.get_string()<<endl;
	cout<<"like="<<result<<"  post="<<post<<endl; 
	result=-INFINITY;
      }
    }
    return result;
  };
};

timing_data qrperf;

//holder for instance specific information for fisher proposal
typedef struct {
  int id;
  int counter;
  vector< vector<double> > covars;
} fisher_instance;

// Next alternative interface for Simplified LISA likelihood not relying on class inheritance
// In this case we still implement as a class for maximum similarity to the original inheritance
// interface, but no class is needed, as demonstrated with the no_class _nc option
// This form of interface can be applied in cython for interfacing with a likelihood written
// in python.
class simple_likelihood_ni {
  int idx_phi,idx_d,idx_inc,idx_lambda,idx_beta,idx_psi;
  bayes_likelihood *blike;
  Options *opt;
  vector<stateSpaceInvolution> symmetries;
  //Stuff for Fisher
  int fisher_update_len,fisher_nmax,fisher_counter;
  vector< fisher_instance* > fisher_instances;
  
public:
  //simple_likelihood():bayes_likelihood(nullptr,nullptr,nullptr){};
  simple_likelihood_ni(bayes_likelihood *blike,Options &opt):blike(blike),opt(&opt){
    blike->register_reference_object(this);
    blike->register_evaluate_log(simple_likelihood_ni::evaluate_log);
    blike->register_defWorkingStateSpace(simple_likelihood_ni::defWorkingStateSpace);
    //Add options for fisher
    this->opt->add(Option("fisher_update_len","Mean number of steps before drawing an update of the Fisher-matrix based proposal. Default 0 (Never update)","0"));
    this->opt->add(Option("fisher_nmax","Max number of Fisher covariance options to hold for proposal draw. Default 0 (No Fisher Proposal)","0"));
  };
  ~simple_likelihood_ni(){for(auto instance : fisher_instances)delete instance;};
  virtual void setup(){    
    ///Set up the output stateSpace for this object

    //setup  stateSpace
    int npar=6;
    stateSpace space(npar);
    string names[]={"d","phi","inc","lambda","beta","psi"};//double phiRef, double d, double inc, double lambd, double beta, double psi) 
    space.set_names(names);
    space.set_bound(1,boundary(boundary::wrap,boundary::wrap,0,2*M_PI));//set 2-pi-wrapped space for phi.  Turn on if not narrow banding
    //maybe this breaks testsuite?
    if(not narrowband){
      space.set_bound(0,boundary(boundary::limit,boundary::limit,0,30));//set 2-pi limited space for lambda.
      space.set_bound(2,boundary(boundary::limit,boundary::limit,0,M_PI));//set 2-pi limited space for lambda.
      space.set_bound(3,boundary(boundary::wrap,boundary::wrap,0,2*M_PI));//set 2-pi-wrapped space for lambda.
      space.set_bound(4,boundary(boundary::limit,boundary::limit,-M_PI/2,M_PI/2));//set limited space for beta.
    }
    else if (true) {
      space.set_bound(0,boundary(boundary::limit,boundary::limit,0,30));//set 2-pi limited space for lambda.

      space.set_bound(2,boundary(boundary::limit,boundary::limit,0,M_PI));//set 2-pi limited space for lambda.
      space.set_bound(3,boundary(boundary::wrap,boundary::wrap,0,2*M_PI));//set 2-pi-wrapped space for lambda.
      space.set_bound(4,boundary(boundary::limit,boundary::limit,-M_PI/2,M_PI/2));//set limited space for beta.
    }
    space.set_bound(5,boundary(boundary::wrap,boundary::wrap,0,M_PI));//set pi-wrapped space for pol.

    //Set up potential state-space symmetries
    qrperf.every=1000000;
    stateSpaceInvolution LISA_quarter_rotation(space,"LISA_quarter_rotation",1);// 1 means need 1 random number
    LISA_quarter_rotation.register_transformState(LISA_quarter_rotation_symmetry);
    stateSpaceInvolution source_quarter_rotation(space,"source_quarter_rotation",1);// 1 means need 1 random number
    source_quarter_rotation.register_transformState(source_quarter_rotation_symmetry);
    stateSpaceInvolution LISA_plane_reflection(space,"LISA_plane_reflection",0);
    LISA_plane_reflection.register_transformState(LISA_plane_reflection_symmetry);
    stateSpaceInvolution transmit_receive_inc_swap(space,"transmit_receive_inc_swap",0);
    transmit_receive_inc_swap.register_transformState(transmit_receive_inc_swap_symmetry);
    stateSpaceInvolution dist_inc_scale(space,"dist_inc_scale",2);
    //stateSpaceInvolution dist_inc_scale(space,"dist_inc_scale",2,&qrperf); //shows timing data
    dist_inc_scale.register_transformState(dist_inc_scale_symmetry);
    dist_inc_scale.register_jacobian(dist_inc_scale_symmetry_jac);

    symmetries.push_back(LISA_quarter_rotation);
    symmetries.push_back(source_quarter_rotation);
    symmetries.push_back(LISA_plane_reflection);
    symmetries.push_back(transmit_receive_inc_swap);
    symmetries.push_back(dist_inc_scale);
    //symmetries.push_back(dist_inc_scale);//Even more of the same
    //symmetries.push_back(dist_inc_scale);//Even more of the same
    //symmetries.push_back(dist_inc_scale);//Even more of the same
    for(auto & symmetry: symmetries)space.addSymmetry(symmetry);
 

    //Set up prior
    vector<double> centers((initializer_list<double>){  1.667,  PI, PI/2,  PI,    0, PI/2});
    vector<double>  scales((initializer_list<double>){  1.333,  PI, PI/2,  PI, PI/2, PI/2});
    vector<string>      types((initializer_list<string>){  "uni","uni", "pol", "uni", "cpol", "uni"});
    if(narrowband){
      centers[3] = 1.75*PI;scales[3] = PI/4.0;
      //centers[4] = 0.4    ;scales[4] = 0.2;
      centers[4] = PI/4    ;scales[4] = PI/4;
    }
    //cout<<"simple_likelihood::setup: space="<<space.show()<<endl;
    
    //blike->basic_setup(&space, types, centers, scales);
    vector<double>likelyscales=scales;
    for( auto & scale:likelyscales)scale=1;
    blike->basic_setup(&space, types, centers, scales, likelyscales);

    //If needed, construct Fisher-based user_gaussian_prop
    istringstream(opt->value("fisher_update_len"))>>fisher_update_len;
    istringstream(opt->value("fisher_nmax"))>>fisher_nmax;
    if(fisher_nmax>0){
      vector<string>fisher_names={"d","phi","inc"};
      //ostringstream ss;ss<<"this="<<this<<" is about to create new user_gaussian_prop. new_fisher_instance="<<new_fisher_instance;cout<<ss.str()<<endl;			 
      user_gaussian_prop prop(this,fisher_check_update,space.subspace_by_name(fisher_names),vector<double>(), 2, "simple_fisher",new_fisher_instance);
      blike->addProposal(&prop);
    }
  };
  
  
  static void defWorkingStateSpace(void *object, const stateSpace &sp){
    //In this case we implement *object as a class object very similar to the inherited version
    //but this is not essential *object could be any user object
    //(or none, if no external data need be referenced).
    simple_likelihood_ni *mythis = static_cast<simple_likelihood_ni*>(object);
    mythis->idx_d=sp.requireIndex("d");
    mythis->idx_phi=sp.requireIndex("phi");
    mythis->idx_inc=sp.requireIndex("inc");
    mythis->idx_lambda=sp.requireIndex("lambda");
    mythis->idx_beta=sp.requireIndex("beta");
    mythis->idx_psi=sp.requireIndex("psi");
  };

  static double evaluate_log(void *object, const state &s){
    simple_likelihood_ni *mythis = static_cast<simple_likelihood_ni*>(object);
    valarray<double>params=s.get_params();
    double d=params[mythis->idx_d];
    double phi=params[mythis->idx_phi];
    double inc=params[mythis->idx_inc];
    double lambd=params[mythis->idx_lambda];
    double beta=params[mythis->idx_beta];
    double psi=params[mythis->idx_psi];
    double result=simpleCalculateLogLCAmpPhase(d, phi, inc, lambd, beta, psi);
    //static int count=0;
    //cout<<count<<endl;
    //count++;
    //cout<<"state: "<<s.get_string()<<endl;
    //cout<<"  logL="<<result<<endl;
    return result;
  };
  
  static void * new_fisher_instance(void *object,int id){
    simple_likelihood_ni *mythis = static_cast<simple_likelihood_ni*>(object);
    fisher_instance * instance= new fisher_instance;
    mythis->fisher_instances.push_back(instance);
    instance->id=id;
    instance->counter=0;
    //ostringstream ss;ss<<"this="<<mythis<<" new fisher instance="<<instance<<" id="<<id<<". Now there are "<< mythis->fisher_instances.size();cout<<ss.str()<<endl;    
    return instance;
  };
    
  static bool fisher_check_update(const void *object, void *instance_object, const state &s, double invtemp, const vector<double> &rands, vector<double> &covarvec){
    //Note: requires nrand==2 for evolving fisher
    const simple_likelihood_ni *mythis = static_cast<const simple_likelihood_ni*>(object);
    fisher_instance *instance = static_cast<fisher_instance*>(instance_object);
    int nfish=instance->covars.size();
    vector<double>randoms=rands;
    if(nfish>0 and (randoms.size()>0 and randoms.back()*mythis->fisher_update_len<1))return false;
    if(randoms.size()>0)randoms.pop_back();
    int add_every=nfish*2;//how long to go before adding a new Fisher covariance to the stack
    //cout<<"check_update: nfish,count: "<<nfish<<" ,"<<instance->counter<<"/"<<add_every<<endl;
    if(nfish==0 or instance->counter>add_every){
      //Here we construct a new fisher matrix and add it to the stack
      instance->counter=0;
      valarray<double>params=s.get_params();
      double d=params[mythis->idx_d];
      double inc=params[mythis->idx_inc];
      valarray<double> scales;mythis->blike->getObjectPrior()->getScales(scales);
      double dscale=scales[mythis->idx_d];
      double incscale=scales[mythis->idx_inc];
      double phiscale=scales[mythis->idx_phi];
      if(nfish>=mythis->fisher_nmax)instance->covars.erase(instance->covars.begin());
      //instance->covars.push_back(partial_simple_cov(d, inc, dscale, phiscale, incscale));
      instance->covars.push_back(dummy_Fisher_cov());
      ostringstream ss("");
      if(nfish==0){
	ss<<"Adding Fisher Covariance ["<<instance->covars.size()<<"] ="<<endl;
	//ss<<"mythis="<<mythis<<" instance="<<instance<<endl;
	vector<double>cov=instance->covars.back();
	ss<<"  "<<setw(20)<<cov[0]<<" "<<setw(20)<<cov[1]<<" "<<setw(20)<<cov[2]<<endl;
	ss<<"  "<<setw(20)<<cov[1]<<" "<<setw(20)<<cov[3]<<" "<<setw(20)<<cov[4]<<endl;
	ss<<"  "<<setw(20)<<cov[2]<<" "<<setw(20)<<cov[4]<<" "<<setw(20)<<cov[5]<<endl;
	cout<<ss.str()<<endl;
      }
	

    }
    //Draw one of the fisher covariances from the stack.  Could make this likelihood weighted, etc...
    int ifish=randoms.back()*instance->covars.size();
    //cout<<"ifish,size:"<<ifish<<","<<instance->covars.size()<<endl;
    randoms.pop_back();
    covarvec=instance->covars[ifish];
    instance->counter+=1;
    return true;
  };
};

shared_ptr<Random> globalRNG;//used for some debugging... 

//***************************************************************************************8
//main test program
int main(int argc, char*argv[]){
  ptmcmc_sampler::Init(argc,argv);
  Options opt(true);
  //Create the sampler
  ptmcmc_sampler mcmc;
  bayes_sampler *s0=&mcmc;
  //Create the model components and likelihood;
  //bayes_data *data=new GRBpop_z_only_data();
  //bayes_signal *signal=new GRBpop_one_break_z_signal();
  
  bayes_likelihood *like;
  simple_likelihood_ni *slike;
  if(no_class){
    like=new bayes_likelihood();
    simple_likelihood_setup_nc(like);
  } else {
    if(use_inheritance_interface)like=new simple_likelihood();
    else{
      like=new bayes_likelihood();
      slike=new simple_likelihood_ni(like,opt);
    }
  }
  
  //prep command-line options
  s0->addOptions(opt);
  //data->addOptions(opt);
  //signal->addOptions(opt);
  like->addOptions(opt);


  //Add some command more line options
  opt.add(Option("nchains","Number of consequtive chain runs. Default 1","1"));
  opt.add(Option("seed","Pseudo random number grenerator seed in [0,1). (Default=-1, use clock to seed.)","-1"));
  opt.add(Option("precision","Set output precision digits. (Default 13).","13"));
  opt.add(Option("outname","Base name for output files (Default 'mcmc_output').","mcmc_output"));
  
  int Nlead_args=1;

  bool parseBAD=opt.parse(argc,argv);
  if(parseBAD) {
    cout << "Usage:\n mcmc [-options=vals] " << endl;
    cout <<opt.print_usage()<<endl;
    return 1;
  }
    
  cout<<"flags=\n"<<opt.report()<<endl;

  //Setup likelihood
  //data->setup();  
  //signal->setup();
  //Note: in this case the simple likelihood has no options and setup could be called at definition
  //If options are passed in this way then setup must be called after the options are processed.
  if(not no_class){
    if(use_inheritance_interface)like->setup();
    else slike->setup();
  }
  
  double seed;
  int Nchain,output_precision;
  int Nsigma=1;
  int Nbest=10;
  string outname;
  ostringstream ss("");
  istringstream(opt.value("nchains"))>>Nchain;
  istringstream(opt.value("seed"))>>seed;
  //if seed<0 set seed from clock
  if(seed<0)seed=fmod(time(NULL)/3.0e7,1);
  istringstream(opt.value("precision"))>>output_precision;
  istringstream(opt.value("outname"))>>outname;

  //report
  cout.precision(output_precision);
  cout<<"\noutname = '"<<outname<<"'"<<endl;
  cout<<"seed="<<seed<<endl; 
  cout<<"Running on "<<omp_get_num_threads()<<" thread"<<(omp_get_num_threads()>1?"s":"")<<"."<<endl;

  //Should probably move this to ptmcmc/bayesian
  ProbabilityDist::setSeed(seed);
  globalRNG.reset(ProbabilityDist::getPRNG());//just for safety to keep us from deleting main RNG in debugging.

  //Get the space/prior for use here
  stateSpace space;
  shared_ptr<const sampleable_probability_function> prior;  
  space=*like->getObjectStateSpace();
  cout<<"like.nativeSpace=\n"<<space.show()<<endl;
  prior=like->getObjectPrior();
  cout<<"Prior is:\n"<<prior->show()<<endl;
  valarray<double> scales;prior->getScales(scales);

  //Read Params
  int Npar=space.size();
  cout<<"Npar="<<Npar<<endl;
  
  //Bayesian sampling [assuming mcmc]:
  //Set the proposal distribution 
  int Ninit;
  //proposal_distribution *prop=ptmcmc_sampler::new_proposal_distribution(Npar,Ninit,opt,prior.get(),&scales);
  //cout<<"Proposal distribution is:\n"<<prop->show()<<endl;
  //set up the mcmc sampler (assuming mcmc)
  //mcmc.setup(Ninit,*like,*prior,*prop,output_precision);
  //mcmc.setup(*like,*prior,output_precision);
  mcmc.setup(*like,output_precision);
  mcmc.select_proposal();

  //Testing (will break testsuite)
  //state  s=like->draw_from_prior();
  
  //Prepare for chain output
  ss<<outname;
  string base=ss.str();

  //cout<<"exampleLISA:like is:"<<like<<endl;
  //cout<<"exampleLISA:&space="<<like->getObjectStateSpace()<<endl;
  //cout<<"exampleLISA:space is:"<<like->getObjectStateSpace()->show();
  //cout<<"best stateSpace is:"<<like->bestState().getSpace()<<endl;
  //cout<<"exampleLISA:best is:"<<like->bestState().show()<<endl;

  //Loop over Nchains
  for(int ic=0;ic<Nchain;ic++){
    bayes_sampler *s=s0->clone();
    s->initialize();
    s->run(base,ic);
    //s->analyze(base,ic,Nsigma,Nbest,*like);
    delete s;
  }
  
  //Dump summary info
  cout<<"best_post "<<like->bestPost()<<", state="<<like->bestState().get_string()<<endl;
  //delete data;
  //delete signal;
  delete like;
}

