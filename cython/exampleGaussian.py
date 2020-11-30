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
import pyximport; pyximport.install()
import math
from scipy.stats import wishart
import sys

MTSUN_SI=4.9254923218988636432342917247829673e-6
PI=3.1415926535897932384626433832795029
I = complex(0.0, 1.0)



class gaussian_likelihood(ptmcmc.likelihood):
  def __init__(self,opt):
    #Note: We define a multivariate Gaussian likelihood with the covariance provided
    #The prior is uniform over a hyper-rectangular larger than the Gaussian core by priorscale*sigma
    opt.add("priorscale","Factor by which the prior is larger than the Gaussian 1-sigma scale. (Default=100)","100")
    opt.add("fisher_cov_rescale","Factor by which 'fisher' proposal is rescaled from nominal value (Default=1,theoretical optimum for Gaussian target dist.)","1")
    opt.add("fisher_basescale_fac","Factor by prior widths are rescaled for addition to fisher_proposal_precision matrix. (Default=0, nothing added)","0")
    
    self.opt=opt
      
  def setup(self,cov,reporting=True):
    
    cov=np.array(cov)
    self.cov=cov
    npar=cov.shape[0]
    self.npar=npar
    
    lndetcov=np.linalg.slogdet(self.cov)[1]
    self.like0=-0.5*(self.npar*np.log(2*np.pi)+lndetcov)
    if reporting:
      print("Seting up likelihood with ln(max)=",self.like0)
      sig=np.sqrt(np.diag(self.cov))
      print("Sigmas:",sig)
      print("Corr:\n"+"\n".join( ('{:6.2f}'*npar).format(*[self.cov[i,j]/sig[i]/sig[j] for j in range(npar)]) for i in range(npar)),'\n') 

    self.invcov=np.linalg.inv(self.cov)
    self.reporting=reporting

    #Set up  stateSpace with trival boundaries
    space=ptmcmc.stateSpace(dim=npar);
    names=["x"+str(i) for i in range(npar)]
    self.names=names
    space.set_names(names);
        
    #Set up prior
    priorscale=100
    centers= [0]*npar
    scales=  [np.sqrt(self.cov[i,i])*priorscale for i in range(npar)]
    types=   [ "uni" ]*npar

    self.basic_setup(space, types, centers, scales);

    #Set up "Fisher" proposal stuff
    propspace=ptmcmc.stateSpace(dim=npar)
    propspace.set_names(self.names)

    #See Optimal Proposal Distributions and Adaptive MCMC,Jeffrey S. Rosenthal* [Chapter for MCMC Handbook]
    # ... based on Roberts, G. O., et al, "WEAK CONVERGENCE AND OPTIMAL SCALING OF RANDOM WALK METROPOLIS ALGORITHMS" Ann Appl Prob,Vol. 7, No. 1, 110-120 (1997)
    #Expect optimal convergence for gaussian with large ndim with fisher_cov_rescale=1.
    self.fisher_cov_rescale=float(self.opt.value("fisher_cov_rescale"))
    self.fisher_basescale_factor=float(self.opt.value("fisher_basescale_fac"))
    fish_cov_fac=2.38**2/npar*self.fisher_cov_rescale 
    if self.fisher_basescale_factor>0: #We simulate the effect of the prior, pretending it is Gaussian.
      basescales=self.fisher_basescale_factor*np.array(scales)
      basescale_invcov=np.diag(basescales**-2)
      fish_cov=np.linalg.inv(self.invcov+basescale_invcov)*fish_cov_fac
    else:
      fish_cov=self.cov*fish_cov_fac          
    proposal=ptmcmc.gaussian_prop(self,frozen_fisher_check_update,propspace,fish_cov, 0, "Fisher-like")
    self.addProposal(proposal)

  def evaluate_log(self,s):
    params=s.get_params()
    params=np.array(params)
    
    llike=self.like0-0.5*np.dot(params,np.dot(self.invcov,params))
    
    return llike

  def writeCovar(self,filename,pars=None):
    cov=self.cov
    names=self.names
    n=cov.shape[0]
    with open(filename,'w') as f:
      if names is not None:
        f.write('#')
        for name in names[:n]: f.write(name+' ')
        f.write('\n')
      if pars is not None:
        f.write('#State ')
        for par in pars[:n]: f.write(str(par)+' ')
        f.write('\n')
      f.write("#Covariance\n")
      for i in range(n):
        for j in range(n):
          f.write(str(cov[i,j])+" ")
        f.write('\n')
      f.write("#Sigmas\n")
      sigmas=[np.sqrt(cov[i,i]) for i in range(n)]
      for i in range(n):
        f.write(str(sigmas[i])+" ")
      f.write('\n')
      f.write("#Correlation\n")
      for i in range(n):
        for j in range(n):
          f.write(str(cov[i,j]/sigmas[i]/sigmas[j])+" ")
        f.write('\n')


#This will be the callback for a gaussian_prop, so it must be declared outside the class
def frozen_fisher_check_update(likelihood, s, invtemp, randoms, covarvec):
  #As it is, we never update. Could do other things for experiments
  return False

#//***************************************************************************************8
#//main test program
def main(argv):

    ptmcmc.Init() 
    #//prep command-line options
    #Options opt(true);
    opt=ptmcmc.Options()
    #//Add some command more line options
    opt.add("seed","Pseudo random number grenerator seed in [0,1). (Default=-1, use clock to seed.)","-1")
    opt.add("outname","Base name for output files (Default 'mcmc_output').","mcmc_output")
    opt.add("p","Parameter dimension for the test.(Default 3)","3")

    #//Create the sampler and likelihood
    s0=ptmcmc.sampler(opt)
    like=gaussian_likelihood(opt)  

    print('calling opt.parse')
    opt.parse(argv)
    print("flags=\n"+opt.report())

    #setup
    p=int(opt.value("p"))
    nu=5+p
    cov=wishart.rvs(nu,np.diag([1]*p))
    like.setup(cov,s0.reporting());
       
    seed=float(opt.value('seed'))
    if seed<0:seed=random.random();
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
    like.writeCovar(outname+"_covar.dat")
    
    #//Read Params
    Npar=space.size();
    print("Npar=",Npar)
    
    #//Bayesian sampling 
    s0.setup(like)

    s=s0.clone();
    s.initialize();
    print('initialization done')
    s.run(outname,0);

if __name__ == "__main__":
    import sys
    argv=sys.argv[:]
    del argv[0]
    main(argv)
