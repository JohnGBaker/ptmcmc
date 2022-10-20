#Simplified likelihood for LISA example based on python interface.  The simplified likelihood covers only extrinsic parameters based on  low-f limit, and short-duration observation as occurs for merger of ~1e6 Msun binaries. 

import numpy as np
import ptmcmc
import random
import pyximport; pyximport.install()
import math

#import simple_likelihood_funcs
import sys

MTSUN_SI=4.9254923218988636432342917247829673e-6
PI=3.1415926535897932384626433832795029
I = complex(0.0, 1.0)
narrowband=False

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

def dummy_Fisher_cov():
  cov=np.array([1,0,-0.25,3e-5,0,0.09]);
  return cov;
  

def simpleCalculateLogLCAmpPhase(d, phiL, inc, lambdL, betaL, psiL):
  #Simple likelihood for runcan 22 mode, frozen LISA, lowf, snr 200
  #normalization factor and injection values sainj, seinj hardcoded - read from Mathematica
  factor = 216147.866077
  sainj = 0.33687296665053773 + I*0.087978055005482114
  seinj = -0.12737105239204741 + I*0.21820079314765678
  #sa2 = simple_likelihood_funcs.funcsa(d, phiL, inc, lambdL, betaL, psiL)
  sa = funcsa(d, phiL, inc, lambdL, betaL, psiL)
  #print('sa compare:',sa,sa2)
  #sys.stdout.flush()
  se = funcse(d, phiL, inc, lambdL, betaL, psiL)
  simplelogL = -1./2 * factor * (pow(abs(sa - sainj), 2) + pow(abs(se - seinj), 2))
  #simplelogL = -1./2 * factor * ( (sa - sainj).real**2+(sa-sainj).imag**2 + (se - seinj).real**2+ (se-seinj).imag**2)
  return simplelogL

#Implementing potential parameter space symmetries
#These class definitions are of a standard form needed for specifying (potential)
#symmetries of the parameter state space, and can be exploited as
#specialized MCMC proposals.

#TDI A/E symmetric (in stationary/low-freq limit) half rotation of constellation or quarter rotation with polarization flip
#uses 1 random var

from LISAsymmetries import LISA_quarter_rotation_symmetry_transf,source_quarter_rotation_symmetry_transf,LISA_plane_reflection_symmetry_transf,transmit_receive_inc_swap_symmetry_transf,dist_inc_scale_symmetry_transf,dist_inc_scale_symmetry_jac,dist_alt_pol_symmetry_transf,dist_alt_pol_symmetry_jac
halfpi=PI/2;
idist=0
iphi=1
iinc=2
ilamb=3
ibeta=4
ipsi=5;
import LISAsymmetries
LISAsymmetries.idist=idist
LISAsymmetries.iphi=iphi
LISAsymmetries.iinc=iinc
LISAsymmetries.ilamb=ilamb
LISAsymmetries.ibeta=ibeta
LISAsymmetries.ipsi=ipsi
LISAsymmetries.reverse_phi_sign=True

    
def trivial_transf(s, randoms): 
    return s

class simple_likelihood(ptmcmc.likelihood):
    def __init__(self,opt):

        #Set up  stateSpace
        npar=6;
        self.opt=opt
        space=ptmcmc.stateSpace(dim=npar);
        names=["d","phi","inc","lambda","beta","psi"];
        space.set_names(names);
        space.set_bound('phi',ptmcmc.boundary('wrap','wrap',0,2*PI));#set 2-pi-wrapped space for phi. 
        space.set_bound('psi',ptmcmc.boundary('wrap','wrap',0,PI))
        if(not narrowband):
            space.set_bound("d",ptmcmc.boundary("limit","limit",0,30)) #set 2-pi limited space for lambda.
            space.set_bound("inc",ptmcmc.boundary("limit","limit",0,PI)) #set 2-pi limited space for lambda.
            space.set_bound("lambda",ptmcmc.boundary("wrap","wrap",0,2*PI)) #set 2-pi-wrapped space for lambda.
            space.set_bound("beta",ptmcmc.boundary("limit","limit",-PI/2,PI/2)) #set limited space for beta.

        #Add potential symmetries
        tevery=1000000; #1000000 for timing test
        space.addSymmetry(ptmcmc.involution(space,"LISA_quarter_rotation",1,LISA_quarter_rotation_symmetry_transf,timing_every=tevery))
        space.addSymmetry(ptmcmc.involution(space,"source_quarter_rotation",1,source_quarter_rotation_symmetry_transf,timing_every=tevery))
        space.addSymmetry(ptmcmc.involution(space,"LISA_plane_reflection",0,LISA_plane_reflection_symmetry_transf,timing_every=tevery))
        space.addSymmetry(ptmcmc.involution(space,"transmit_receive_inc_swap",0,transmit_receive_inc_swap_symmetry_transf,timing_every=tevery))
        space.addSymmetry(ptmcmc.involution(space,"dist_inc_scale",2,dist_inc_scale_symmetry_transf,dist_inc_scale_symmetry_jac,timing_every=tevery))
        space.addSymmetry(ptmcmc.involution(space,"dist_alt_pol",2,dist_alt_pol_symmetry_transf,dist_alt_pol_symmetry_jac,timing_every=tevery))
        #space.addSymmetry(ptmcmc.involution(space,"trivial",1,trivial_transf))
        
        #Set up prior
        centers= [ 1.667,    PI,    PI/2,    PI,       0,    PI/2]
        scales=  [ 1.333,    PI,    PI/2,    PI,    PI/2,    PI/2]
        types=   [ "uni", "uni",   "pol", "uni",  "cpol",   "uni"]
        if(narrowband):
            centers[3] = 1.75*PI;scales[3] = PI/4.0;
            centers[4] = PI/4   ;scales[4] = PI/4;
            
        #print("simple_likelihood::setup: space="+space.show())
        #lscales=[0.1 for x in scales]
        #self.basic_setup(space, types, centers, scales, lscales);
        self.basic_setup(space, types, centers, scales, scales, check_posterior=False);

        #Fisher options
        self.opt.add("fisher_update_len","Mean number of steps before drawing an update of the Fisher-matrix based proposal. Default 0 (Never update)","0");
        self.opt.add("fisher_nmax","Max number of Fisher covariance options to hold for proposal draw. Default 0 (No Fisher Proposal)","0");

      
        
    def setup(self):
      #Evertyhing but Fisher options has been done in constructor
      #istringstream(opt->value("fisher_update_len"))>>fisher_update_len;
      #istringstream(opt->value("fisher_nmax"))>>fisher_nmax;
      self.fisher_update_len=int(self.opt.value("fisher_update_len"))
      self.fisher_nmax=int(self.opt.value("fisher_nmax"))
      if self.fisher_nmax>0:
        self.fisher_counter=0;
        self.fisher_names=["d","phi","inc"];
        propspace=ptmcmc.stateSpace(dim=3)
        propspace.set_names(["d","phi","inc"])
        print("creating dummy_fisher proposal self=",self)
        if False:
            default_data={}
            default_data['covars']=[]
            default_data['counter']=0
            proposal=ptmcmc.gaussian_prop(self,fisher_check_update,propspace,np.zeros(0), 2, "dummy_fisher",default_instance_data=default_data)
        else:
            proposal=ptmcmc.gaussian_prop(self,frozen_fisher_check_update,propspace,np.zeros(0), 1, "frozen_dummy_fisher")
        self.addProposal(proposal)


    def evaluate_log(self,s):
        params=s.get_params()
        d=params[0]
        phi=params[1]
        inc=params[2]
        lambd=params[3]
        beta=params[4]
        psi=params[5]
        result=simpleCalculateLogLCAmpPhase(d, phi, inc, lambd, beta, psi);
        #result=simple_likelihood_funcs.simpleCalculateLogLCAmpPhase(d, phi, inc, lambd, beta, psi);
        #if False:
        #  global count
        #  print(count)
        #  count+=1
        #  print("state:",s.get_string())
        #  print("  logL={0:.13g}".format(result))
        return result


#This will be the callback for a gaussian_prop, so it must be declared outside the class
def frozen_fisher_check_update(likelihood, s, randoms, covarvec):
        #Note: need nrand>=1
        if(randoms[-1]*likelihood.fisher_update_len<1):return False
        if(len(randoms)>0):randoms=randoms[:-1]
        covarvec=dummy_Fisher_cov();
        return True

#This will be the callback for a gaussian_prop, so it must be declared outside the class
def fisher_check_update(likelihood, instance, s, randoms, covarray):
    #Note: requires nrand==2 for evolving fisher
    fisher_covars=instance['covars']
    nfish=len(fisher_covars);
    if(nfish>0 and (len(randoms)>0 and randoms[-1]*likelihood.fisher_update_len>1)):return False
    if(len(randoms)>0):randoms=randoms[:-1]
    everyfac=1
    add_every=nfish*everyfac;#how long to go before adding a new Fisher covariance to the stack
    #print("check_update: nfish,count: ",nfish," ,",counter,"/",add_every)
    if(nfish==0 or instance['counter']>add_every):
      if nfish==0 and likelihood.reporting:
        print("Fisher replenishment scale is ",likelihood.fisher_nmax**2*everyfac*likelihood.fisher_update_len,"draws. Consider increasing nmax or update_len if this is not greater than autocorrelation length.")
            
      #Here we construct a new fisher matrix and add it to the stack
      instance['counter']=0;
      params=s.get_params()
      if(nfish>=likelihood.fisher_nmax):fisher_covars=fisher_covars[1:]
      cov=dummy_Fisher_cov();
      #cov=cov/likelihood.fisher_reduce_fac
      fisher_covars.append(cov)
      verbose=((randoms[0]<1/likelihood.fisher_nmax) or (nfish<likelihood.fisher_nmax) ) and likelihood.reporting
      if verbose:
        print("evalFisherCov time:",end-start)#, "\ns=",s.get_params())
        print("Fisher Covariance [",len(fisher_covars),"] ")
        #print(cov)
        sigs=np.sqrt(np.diag(cov))
        print("New Fisher, sigmas:",sigs)
        n=len(sigs)
        print("Corr:\n"+"\n".join( ('{:6.2f}'*n).format(*[cov[i,j]/sigs[i]/sigs[j] for j in range(n)]) for i in range(n)),'\n') 
    #Draw one of the fisher covariances from the stack.  Could make this likelihood weighted, etc...
    ifish=int(randoms[-1]*len(fisher_covars));
    #cout<<"ifish,size:"<<ifish<<","<<fisher_covars.size()<<endl;
    randoms=randoms[:-1]
    np.copyto(covarray,fisher_covars[ifish])
    instance['counter']+=1
    return True

count=0



#//***************************************************************************************8
#//main test program
def main(argv):

    ptmcmc.Init() 
    opt=ptmcmc.Options()
    opt.add("seed","Pseudo random number grenerator seed in [0,1). (Default=-1, use clock to seed.)","-1")
    opt.add("outname","Base name for output files (Default 'mcmc_output_t0').","mcmc_output_t0")

    #Create the sampler
    s0=ptmcmc.sampler(opt)
    #Create the likelihood
    like=simple_likelihood(opt)  


    print('calling opt.parse')
    opt.parse(argv)
    print("flags=\n"+opt.report())

    #Setup likelihood
    like.setup();
    space=like.getObjectStateSpace();
       
    seed=float(opt.value('seed'))
    if seed<0:seed=random.random();
    outname=opt.value('outname')
    
    #Should probably move this to ptmcmc/bayesian
    ptmcmc.resetRNGseed(seed);
    
    #report
    print("\noutname = '"+outname+"'")
    print("like.nativeSpace=\n"+space.show())
    Npar=space.size();
    print("Npar=",Npar)
    
    #Bayesian sampler setup
    s0.setup(like)

    #Do Sampling
    s=s0.clone();
    s.initialize();
    print('initialization done')
    s.run(outname,0);
    
if __name__ == "__main__":
    import sys
    argv=sys.argv[:]
    del argv[0]
    main(argv)
