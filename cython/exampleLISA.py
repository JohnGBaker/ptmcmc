#Simplified likelihood for LISA example based on python interface.  The simplified likelihood covers only
#extrinsic parameters based on  low-f limit, and short-duration observation
#as occurs for merger of ~1e6 Msun binaries. 


#include <valarray>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <complex>
#include "omp.h"
#include "options.hh"
#include "bayesian.hh"
#include "proposal_distribution.hh"
#include "ptmcmc.hh"

#using namespace std;

import numpy as np
import ptmcmc
import random

MTSUN_SI=4.9254923218988636432342917247829673e-6
PI=3.1415926535897932384626433832795029
I = complex(0.0, 1.0)
narrowband=True

# Routines for simplified likelihood 22 mode, frozen LISA, lowf, fixed masses (near 1e6) and fixed t0
def funcphiL(m1, m2, tRef, phiRef):
  MfROMmax22 = 0.14
  fRef = MfROMmax22/((m1 + m2) * MTSUN_SI)
  return -phiRef + PI*tRef*fRef

def funclambdaL(lambd, beta):
  return -np.atan2(np.cos(beta)*np.cos(lambd)*np.cos(PI/3) + np.sin(beta)*np.sin(PI/3), np.cos(beta)*np.sin(lambd))

def funcbetaL(lambd, beta):
  return -np.asin(np.cos(beta)*np.cos(lambd)*np.sin(PI/3) - np.sin(beta)*np.cos(PI/3))

def funcpsiL(lambd, beta, psi):
  return np.atan2(np.cos(PI/3)*np.cos(beta)*np.sin(psi) - np.sin(PI/3)*(np.sin(lambd)*np.cos(psi) - np.cos(lambd)*np.sin(beta)*np.sin(psi)), np.cos(PI/3)*np.cos(beta)*np.cos(psi) + np.sin(PI/3)*(np.sin(lambd)*np.sin(psi) + np.cos(lambd)*np.sin(beta)*np.cos(psi)))

def funcsa(d, phi, inc, lambd, beta, psi):
  Daplus = I * ( 3./4 * (3 - np.cos(2*beta)) * np.cos(2*lambd - PI/3) )
  Dacross = I * (3.0*np.sin(beta) * np.sin(2*lambd - PI/3))
  a22 = 0.5/d * np.sqrt(5/PI) * pow(np.cos(inc/2), 4) * np.exp(2.*I*(-phi-psi)) * 0.5*(Daplus + I*Dacross)
  a2m2 = 0.5/d * np.sqrt(5/PI) * pow(np.sin(inc/2), 4) * np.exp(2.*I*(-phi+psi)) * 0.5*(Daplus - I*Dacross)
  return a22 + a2m2

def funcse(d, phi, inc, lambd, beta, psi):
  Deplus = -I*(3./4 * (3 - np.cos(2*beta)) * np.sin(2*lambd - PI/3))
  Decross = I*(3*np.sin(beta) * np.cos(2*lambd - PI/3))
  e22 = 0.5/d * np.sqrt(5/PI) * pow(np.cos(inc/2), 4) * np.exp(2.*I*(-phi-psi)) * 0.5*(Deplus + I*Decross)
  e2m2 = 0.5/d * np.sqrt(5/PI) * pow(np.sin(inc/2), 4) * np.exp(2.*I*(-phi+psi)) * 0.5*(Deplus - I*Decross)
  return e22 + e2m2


def simpleCalculateLogLCAmpPhase(d, phiL, inc, lambdL, betaL, psiL):
  #Simple likelihood for runcan 22 mode, frozen LISA, lowf, snr 200
  #normalization factor and injection values sainj, seinj hardcoded - read from Mathematica
  factor = 216147.866077
  sainj = 0.33687296665053773 + I*0.087978055005482114
  seinj = -0.12737105239204741 + I*0.21820079314765678
  sa = funcsa(d, phiL, inc, lambdL, betaL, psiL)
  se = funcse(d, phiL, inc, lambdL, betaL, psiL)
  simplelogL = -1./2 * factor * (pow(abs(sa - sainj), 2) + pow(abs(se - seinj), 2))
  return simplelogL

class simple_likelihood(ptmcmc.likelihood):
    def __init__(self):

        #Set up  stateSpace
        npar=6;
        space=ptmcmc.stateSpace(dim=npar);
        names=["d","phi","inc","lambda","beta","psi"];
        space.set_names(names);
        space.set_bound('phi',ptmcmc.boundary('wrap','wrap',0,2*PI));#set 2-pi-wrapped space for phi. 
        #if(not narrowband):space.set_bound('lambda',ptmcmc.boundary('wrap','wrap',0,2*PI))
        #else: space.set_bound('beta',ptmcmc.boundary('limit','limit',0.2,0.6))
        space.set_bound('psi',ptmcmc.boundary('wrap','wrap',0,PI))

        #Set up prior
        centers= [ 1.667,    PI,    PI/2,    PI,       0,    PI/2]
        scales=  [ 1.333,    PI,    PI/2,    PI,    PI/2,    PI/2]
        types=   [ "uni", "uni",   "pol", "uni",  "cpol",   "uni"]
        if(narrowband):
            centers[3] = 1.75*PI;scales[3] = PI/4.0;
            centers[4] = PI/4   ;scales[4] = PI/4;
            
        #print("simple_likelihood::setup: space="+space.show())
        self.basic_setup(space, types, centers, scales);

    def evaluate_log(self,s):
        params=s.get_params()
        d=params[0]
        phi=params[1]
        inc=params[2]
        lambd=params[3]
        beta=params[4]
        psi=params[5]
        result=simpleCalculateLogLCAmpPhase(d, phi, inc, lambd, beta, psi);
        #global count
        #print(count)
        #count+=1
        #print("state:",s.get_string())
        #print("  logL={0:.13g}".format(result))
        return result

count=0



#//***************************************************************************************8
#//main test program
def main(argv):


    #//prep command-line options
    #Options opt(true);
    opt=ptmcmc.Options()
    #s0->addOptions(opt);
    #//data->addOptions(opt);
    #//signal->addOptions(opt);
    #like->addOptions(opt); #There are no options for this or it might be  more complicated
    
    #//Add some command more line options
    ##opt.add("nchains","Number of consequtive chain runs. Default 1","1")
    opt.add("seed","Pseudo random number grenerator seed in [0,1). (Default=-1, use clock to seed.)","-1")
    ##opt.add("precision","Set output precision digits. (Default 13).","13")
    opt.add("outname","Base name for output files (Default 'mcmc_output').","mcmc_output")
    #int Nlead_args=1;


    #//Create the sampler
    #ptmcmc_sampler mcmc;
    s0=ptmcmc.sampler(opt)
    #//Create the likelihood
    like=simple_likelihood()  


    print('calling opt.parse')
    opt.parse(argv)
    #bool parseBAD=opt.parse(argc,argv);
    #if(parseBAD) {
    #  cout << "Usage:\n mcmc [-options=vals] " << endl;
    #  cout <<opt.print_usage()<<endl;
    #  return 1;
    #}
    #cout<<"flags=\n"<<opt.report()<<endl;
    
    #//Setup likelihood
    #//data->setup();  
    #//signal->setup();  
    #like->setup();
    
    #double seed;
    #int Nchain,output_precision;
    #int Nsigma=1;
    #int Nbest=10;
    #string outname;
    #ostringstream ss("");
    #istringstream(opt.value("nchains"))>>Nchain;
    #istringstream(opt.value("seed"))>>seed;
    #//if seed<0 set seed from clock
    #if(seed<0)seed=fmod(time(NULL)/3.0e7,1);
    seed=float(opt.value('seed'))
    if seed<0:seed=random.random();
    #istringstream(opt.value("precision"))>>output_precision;
    #istringstream(opt.value("outname"))>>outname;
    outname=opt.value('outname')
    
    #//report
    #cout.precision(output_precision);
    print("\noutname = '"+outname+"'")
    #cout<<"seed="<<seed<<endl; 
    #cout<<"Running on "<<omp_get_max_threads()<<" thread"<<(omp_get_max_threads()>1?"s":"")<<"."<<endl;
    
    #//Should probably move this to ptmcmc/bayesian
    ptmcmc.resetRNGseed(seed);
    
    #globalRNG.reset(ProbabilityDist::getPRNG());//just for safety to keep us from deleting main RNG in debugging.
          
    #//Get the space/prior for use here
    #stateSpace space;
    #shared_ptr<const sampleable_probability_function> prior;  
    space=like.getObjectStateSpace();
    print("like.nativeSpace=\n"+space.show())
    #prior=like->getObjectPrior();
    #cout<<"Prior is:\n"<<prior->show()<<endl;
    #valarray<double> scales;prior->getScales(scales);
    
    #//Read Params
    Npar=space.size();
    print("Npar=",Npar)
    
    #//Bayesian sampling [assuming mcmc]:
    #//Set the proposal distribution 
    #int Ninit;
    #proposal_distribution *prop=ptmcmc_sampler::new_proposal_distribution(Npar,Ninit,opt,prior.get(),&scales);
    #cout<<"Proposal distribution is:\n"<<prop->show()<<endl;
    #//set up the mcmc sampler (assuming mcmc)
    #//mcmc.setup(Ninit,*like,*prior,*prop,output_precision);
    #mcmc.setup(*like,*prior,output_precision);
    #mcmc.select_proposal();
    s0.setup(like)

    #//Testing (will break testsuite)
    #s=like.draw_from_prior();
    #print("test state:",s.get_string())
    #print("logL=",like.evaluate_log(s))
  
    
    #//Prepare for chain output
    #ss<<outname;
    #string base=ss.str();
    
    #//Loop over Nchains
    #for(int ic=0;ic<Nchain;ic++){
    s=s0.clone();
    s.initialize();
    print('initialization done')
    s.run(outname,0);
    #  //s->analyze(base,ic,Nsigma,Nbest,*like);
    #del s;
    #}
    
    #//Dump summary info
    #cout<<"best_post "<<like->bestPost()<<", state="<<like->bestState().get_string()<<endl;
    #//delete data;
    #//delete signal;
    #delete like;
    #}

if __name__ == "__main__":
    import sys
    argv=sys.argv[:]
    del argv[0]
    main(argv)
