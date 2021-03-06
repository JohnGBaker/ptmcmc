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

"""
def LISA_quarter_rotation_symmetry_transf(s, randoms): 
    #print("applying quarter rotation")
    #sp=s.getSpace()
    #ilamb=sp.requireIndex("lambda")
    #ipsi=sp.requireIndex("psi")   #Takes an extra us to do this step, slows testing
    parvals=s.get_params()
    nrot=randoms[0]
    nrot=int(abs(randoms[0])*2)+1
    if(randoms[0]<0):nrot=-nrot;
    lamb=parvals[ilamb]
    psi=parvals[ipsi]
    lamb+=nrot*halfpi;
    if(abs(nrot)%2==1):psi+=halfpi;
    parvals[ilamb]=lamb
    parvals[ipsi]=psi
    #print("applied quarter rotation")
    return ptmcmc.state(s,parvals);

def source_quarter_rotation_symmetry_transf(s, randoms): 
    #sp=s.getSpace()
    #ilamb=sp.requireIndex("lambda")
    #ipsi=sp.requireIndex("psi")   #Takes an extra us to do this step, slows testing
    iphi=1
    ipsi=5
    param=s.get_params()
    nrot=randoms[0]
    nrot=int(abs(randoms[0])*2)+1
    if(randoms[0]<0):nrot=-nrot;
    phi=param[iphi]
    psi=param[ipsi]
    phi+=nrot*halfpi;
    if(abs(nrot)%2==1):psi+=halfpi;
    param[iphi]=phi
    param[ipsi]=psi
    return ptmcmc.state(s,param);

#TDI A/E symmetric (in stationary/low-freq limit) relection through constellation plane, simultaneous with source plane reflection and polarization flip
#Uses 0 random vars
def LISA_plane_reflection_symmetry_transf(s, randoms): 
    param=s.get_params()
    beta=param[ibeta]
    psi=param[ipsi]
    inc=param[iinc]
    inc=PI-inc;
    beta=-beta;
    psi=PI-psi;
    param[iinc]=inc
    param[ibeta]=beta
    param[ipsi]=psi
    return ptmcmc.state(s,param);

def transmit_receive_inc_swap_symmetry_transf(s, randoms): 
    param=s.get_params()
    phi=param[iphi]
    inc=param[iinc]
    lamb=param[ilamb]
    beta=param[ibeta]
    psi=param[ipsi]
    theta=halfpi-beta
    twopsi=2*psi
    ti4=math.tan(inc/2)**4;
    tt4=math.tan(theta/2)**4;
    Phi=math.atan2(math.sin(twopsi)*(ti4-tt4),math.cos(twopsi)*(ti4+tt4))/2;
    param[iinc]=theta
    param[ibeta]=halfpi-inc
    param[iphi]=phi-Phi#sign differs from that in notes
    param[ilamb]=lamb-Phi
    return ptmcmc.state(s,param);

#Approximate distance inclination symmetry
#uses 2 random var
dist_inc_jump_size=0.1;
def dist_inc_scale_symmetry_transf(s, randoms): 
    #We apply a symmetry to preserve d'*F(x')=d*F(x) where F(x)=1/cos(x)^2
    #Depending on the sign of the second random number, x is either the source inclination, or the
    #line-of-sight inclination relative to the LISA plane, theta=pi/2-beta;
    #To avoid issues at the edges we make sure that the transformation of the inclination
    #never crosses its limits.
    #Note that f:x->ln(pi/x-1) has range (inf,-inf) on domain (0,pi) with f(pi-x)=-f(x)
    #and inverse pi/(exp(f(x))+1)=x
    #We then step uniformly in f(x). So, for random number y,
    # x'=finv(f(x)+y)
    param=s.get_params()
    use_theta=False;
    if(abs(randoms[1]*2)<1): #Half of the time we apply the transformation to theta (LISA includination) rather than source inclination
      use_theta=True
      oldalt=halfpi-param[ibeta]
    else:
      oldalt=param[iinc]
    dist=param[idist]
    df=randoms[0]*dist_inc_jump_size #Uniformly transform reparameterized inc
    oldf=math.log(PI/oldalt-1);
    newf=oldf+df;
    newalt=PI/(math.exp(newf)+1);
    cosold=math.cos(oldalt)
    cosnew=math.cos(newalt)
    fac=cosnew/cosold;
    #double fac=(cosnew*cosnew+1)/(cosold*cosold+1);
    dist=dist*fac;
    param[idist]=dist
    if(use_theta):
      param[ibeta]=halfpi-newalt #convert back to beta
    else:
      param[iinc]=newalt
    return ptmcmc.state(s,param);

#Approximate distance inclination symmetry jacobian
#uses 1 random var
def dist_inc_scale_symmetry_jac(s, randoms): 
    #The transformation has the form:
    #  d' = d F(x)/F'(x')
    #  x' = finv( f(x) + y )
    #  y' = -y
    #where x is the selected inclination variable and y is the random number.
    #The Jacobian is then -F(x)f'(x) / (F(x')f'(x'))
    #Because the random step is performed on the rescaled inclination f(x)=ln(pi/x-1)
    #we have 1/f'(x) = x(1-x/pi)
    param=s.get_params()
    use_theta=False
    if(abs(randoms[1]*2)<1): #Half of the time we apply the transformation to theta (LISA includination) rather than source inclination
      use_theta=True
      oldalt=halfpi-param[ibeta]
    else:
      oldalt=param[iinc]
    dist=param[idist]
    df=randoms[0]*dist_inc_jump_size #Uniformly transform reparameterized inc
    oldf=math.log(PI/oldalt-1);
    newf=oldf+df;
    newalt=PI/(math.exp(newf)+1);
    cosold=math.cos(oldalt)
    cosnew=math.cos(newalt)
    fac=cosnew/cosold;
    fac*=(PI-newalt)*newalt/(PI-oldalt)/oldalt;
    return fac; 

#Exact-in-limit distance-altitude-polarization 2-D symmetry
dist_alt_pol_size=0.1;
Tlimit=1e15
def dist_alt_pol_symmetry_jac(s, randoms): 
    '''
    This transform exercizes the an exact version of the distance-altitude-polarization symmetry
    Which exists exactly for signals from a quadrupolar rotatoring source detected by a full
    polarization non-accelerating detector that is small compared to the wavelength. It uses 
    2 random vars. 

    The symmetry is realized by a random step in two variables in a transformed coordinate system.
    '''
    
    param=s.get_params()

    #params
    iota=param[iinc]
    theta=halfpi-param[ibeta]
    psi=param[ipsi]
    dist=param[idist]
    
    #intermediates
    x=math.tan(theta/2)**4
    y=math.tan(iota/2)**4
    x2=x*x
    y2=y*y
    twoc4psi=2*math.cos(4*psi)
    ztwoc4psi=x*y*twoc4psi
    R=(x2+y2+ztwoc4psi)/(1+x2*y2+ztwoc4psi)
    Delta=(1-x2)*(1-y2)/(dist*(1+math.sqrt(x))*(1+math.sqrt(y)))**2

    #The notes (currently) apply when R<=1, we extend to R>1 noting that
    #R->1/R when z<->w:
    #Thus in the case R>1 we need to swap the definitions of z and w, meaning wee change y->1/y in these exprs
    ypow=1
    if R>1: ypow=-1

    z=x*y**ypow
    #z2=z*z
    if z<1:sgn=-1
    else: sgn=1
    s2psi=math.sin(2*psi)
    c2psi=math.cos(2*psi)
    
    #forward reparameterization
    w=x/y**ypow
    
    #transform
    psit=psi+dist_alt_pol_size*randoms[0]
    c4psit=math.cos(4*psit)
    wt=sgn*w+dist_alt_pol_size*randoms[0]
    if wt>0:sgn=1
    else:sgn=-1
    wt=abs(wt)

    #reverse reparameterization and intermediates    
    T  = ( (wt+1/wt)/2 + c4psit*(1-R**ypow) )/R**ypow
    if True and  T>Tlimit: #limiting case near poles
        rootr=math.sqrt(R)
        if sgn>0:
            zt=1/(2*T)
            if wt>1:
                xt=1/rootr
                yt=2*T*rootr
            else:
                xt=2*T*rootr
                yt=1/rootr
        else:
            zt=2*T
            if wt>1:
                xt=rootr/2/T
                yt=1/rootr
            else:
                xt=1/rootr
                yt=rootr/2/T
        zt=xt*yt**sgn
        
    else:
        zt = T+sgn*math.sqrt(T*T-1)
        xt = math.sqrt(zt*wt)
        if xt<wt*1e-60: yt=R  #handle pathological case
        else: yt = (xt/wt)**ypow
    distt = math.sqrt((1-xt*xt)*(1-yt*yt)/Delta) / ((1+math.sqrt(xt))*(1+math.sqrt(yt)))

    #test
    #Rt=(xt**2+yt**2+xt*yt*2*c4psit)/(1+xt**2*yt**2+xt*yt*2*c4psit)
    #Deltat=(1-xt**2)*(1-yt**2)/(distt*(1+math.sqrt(xt))*(1+math.sqrt(yt)))**2
    #print("check: ",x,y,z,w,dist,T,R,Delta)
    #print("checkt:",xt,yt,xt*yt,wt,distt,T)

    #Jacobian
    #Ratio of the forward transformation to the directly transformed coords at old point over new point
    #theta->x: dx/dtheta ~ x*(x^1/4+x^-1/4)
    #iota->y:  dy/diota  ~ y*(y^1/4+y^-1/4)
    #(x,y,d)->(w,R,Delta): ~ w*(1-z2)/(z*d*(1+z2+2zcos4psi))
    #combined: w/d*(x^1/4+x^-1/4)*(y^1/4+y^-1/4)*(1-z2)/(1+z2+2zcos4psi)
    eps=1e-60
    x4th=x**0.25+eps
    y4th=y**0.25+eps
    xt4th=xt**0.25+eps
    yt4th=yt**0.25+eps
    
    jac=w/dist*(x4th+1/x4th)*(y4th+1/y4th)*(z-1/z)/(z+1/z+twoc4psi)
    jact=wt/distt*(xt4th+1/xt4th)*(yt4th+1/yt4th)*(zt-1/zt)/(zt+1/zt+2*c4psit)

    return abs(jac/jact)
"""
    
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
        #if(not narrowband):space.set_bound('lambda',ptmcmc.boundary('wrap','wrap',0,2*PI))
        #else: space.set_bound('beta',ptmcmc.boundary('limit','limit',0.2,0.6))
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
    opt.add("outname","Base name for output files (Default 'mcmc_output_t0').","mcmc_output_t0")
    #int Nlead_args=1;


    #//Create the sampler
    #ptmcmc_sampler mcmc;
    s0=ptmcmc.sampler(opt)
    #//Create the likelihood
    like=simple_likelihood(opt)  


    print('calling opt.parse')
    opt.parse(argv)
    print("flags=\n"+opt.report())

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
    like.setup();
       
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
