#Functions implementing potential parameter space symmetries
#John Baker (2019) NASA-GSFC
#
#These functions are of a standard form needed for specifying (potential)
#symmetries of the parameter state space, and can be exploited as
#specialized MCMC proposals.

#Implementing potential parameter space symmetries
#These class definitions are of a standard form needed for specifying (potential)
#symmetries of the parameter state space, and can be exploited as
#specialized MCMC proposals.

#TDI A/E symmetric (in stationary/low-freq limit) half rotation of constellation or quarter rotation with polarization flip
#uses 1 random var

import math
import ptmcmc

PI=3.1415926535897932384626433832795029
halfpi=PI/2;
idist=0
iphi=1
iinc=2
ilamb=3
ibeta=4
ipsi=5;
reverse_phi_sign=False


####just for debugging:
I = complex(0.0, 1.0)
import numpy as np
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
  #sa2 = simple_likelihood_funcs.funcsa(d, phiL, inc, lambdL, betaL, psiL)
  sa = funcsa(d, phiL, inc, lambdL, betaL, psiL)
  #print('sa compare:',sa,sa2)
  #sys.stdout.flush()
  se = funcse(d, phiL, inc, lambdL, betaL, psiL)
  simplelogL = -1./2 * factor * (pow(abs(sa - sainj), 2) + pow(abs(se - seinj), 2))
  #simplelogL = -1./2 * factor * ( (sa - sainj).real**2+(sa-sainj).imag**2 + (se - seinj).real**2+ (se-seinj).imag**2)
  return simplelogL
######


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
    if reverse_phi_sign: 
        param[iphi]=phi-Phi#sign differs from that in notes
    else:
        param[iphi]=phi+Phi#sign differs from that in simplelikelihood
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
    #print(s.show())
    param=s.get_params()
    use_theta=False;
    if(abs(randoms[1]*2)<1): #Half of the time we apply the transformation to theta (LISA includination) rather than source inclination
      use_theta=True
      oldalt=halfpi-param[ibeta]
    else:
      oldalt=param[iinc]
    dist=param[idist]
    df=randoms[0]*dist_inc_jump_size #Uniformly transform reparameterized inc
    #print("oldalt=",oldalt)
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
    #  d' = d F(x)/F(x')
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
    #if fac<=0:print("fac=",fac)
    return abs(fac); 



#Exact-in-limit distance-altitude-polarization 2-D symmetry
dist_alt_pol_psi_size=0.5;
dist_alt_pol_w_size=0.5;
lncut=math.log(100)*0
Tlimit=1e8
xxcount=0
xxcut=0
def dist_alt_pol_symmetry_transf(s, randoms): 
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
    lamb=param[ilamb]
    phi=param[iphi]
    
    #intermediates
    x=math.tan(theta/2)**4
    y=math.tan(iota/2)**4
    x2=x*x
    y2=y*y
    twoc4psi=2*math.cos(4*psi)
    ztwoc4psi=x*y*twoc4psi
    R=(x2+y2+ztwoc4psi)/(1+x2*y2+ztwoc4psi)
    Delta=(1-x2)*(1-y2)/(dist**2*((1+math.sqrt(x))*(1+math.sqrt(y)))**4)

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
    Psip=math.atan2((y-x)*s2psi,(x+y)*c2psi)
    Psim=math.atan2((x*y-1)*s2psi,(x*y+1)*c2psi)    
    
    #forward reparameterization
    w=x/y**ypow
    
    #transform
    psit=psi+dist_alt_pol_psi_size*randoms[0]
    c4psit=math.cos(4*psit)
    wt=sgn*w+dist_alt_pol_w_size*randoms[1]
    if wt>0:sgn=1
    else:sgn=-1
    wt=abs(wt)

    #reverse reparameterization and intermediates    
    T  = ( (wt+1/wt)/2 + c4psit*(1-R**ypow) )/R**ypow
    if True and  T>Tlimit: #limiting case near poles
        rootr=math.sqrt(R)
        if ypow>0:
            if wt>0:
                xt=1/rootr
                yt=2*T*rootr
            else:
                xt=2*T*rootr
                yt=1/rootr
        else:
            if wt<0:
                xt=rootr/2/T
                yt=rootr
            else:
                xt=rootr
                yt=rootr/2/T
        zt=xt*yt**ypow
    else:
        zt = T+sgn*math.sqrt(T*T-1)
        xt = math.sqrt(zt*wt)
        if xt<wt*1e-60: yt=R  #handle pathological case
        else: yt = (xt/wt)**ypow
    if (1-xt*xt)*(1-yt*yt)/Delta<0:
        print('dist',T,math.sqrt(R),(1-xt*xt),(1-yt*yt),Delta,(1-xt*xt)*(1-yt*yt)/Delta,sgn,ypow)
        print('old',w,z,x,y)
        print('new',wt,zt,xt,yt)
    distt = math.sqrt((1-xt*xt)*(1-yt*yt)/Delta) / ((1+math.sqrt(xt))*(1+math.sqrt(yt)))**2

    s2psi=math.sin(2*psit)
    c2psi=math.cos(2*psit)
    Psipt=math.atan2((yt-xt)*s2psi,(xt+yt)*c2psi)
    Psimt=math.atan2((xt*yt-1)*s2psi,(xt*yt+1)*c2psi)

    #test
    #Rt=(xt**2+yt**2+xt*yt*2*c4psit)/(1+xt**2*yt**2+xt*yt*2*c4psit)
    #Deltat=(1-xt**2)*(1-yt**2)/(distt**2*((1+math.sqrt(xt))*(1+math.sqrt(yt)))**4)
    #print("check: ",x,y,z,w,dist,T,R,Delta)
    #print("checkt:",xt,yt,zt,wt,distt,T,Rt,Deltat)

    #complete angle params
    thetat=math.atan(xt**0.25)*2
    iotat=math.atan(yt**0.25)*2
    lambt=lamb+0.25*(Psim-Psimt-Psip+Psipt)
    global reverse_phi_sign
    if reverse_phi_sign:
        phit = phi-0.25*(Psim-Psimt+Psip-Psipt)
    else:
        phit = phi+0.25*(Psim-Psimt+Psip-Psipt)

    #hacks
    eps=1e-60
    #Next line restricts to just cases where xy<<1 or xy>>1
    global xxcount,xxcut
    xxcount+=1
    if abs(math.log(x*y+eps))<lncut or abs(math.log(xt*yt+eps))<lncut:
        xxcut+=1
        return ptmcmc.state(s,param);
    else:
        """
        sa=funcsa(dist, phi, iota, lamb, halfpi-theta, psi)
        se=funcse(dist, phi, iota, lamb, halfpi-theta, psi)
        sat=funcsa(distt, phit, iotat, lambt, halfpi-thetat, psit)
        set=funcse(distt, phit, iotat, lambt, halfpi-thetat, psit)
        sigp=sa+I*se
        sigm=sa-I*se
        sigpt=sat+I*set
        sigmt=sat-I*set
        Rt=(xt**2+yt**2+xt*yt*2*c4psit)/(1+xt**2*yt**2+xt*yt*2*c4psit)
        Deltat=(1-xt**2)*(1-yt**2)/(distt**2*((1+math.sqrt(xt))*(1+math.sqrt(yt)))**4)
        L= simpleCalculateLogLCAmpPhase(dist, phi, iota, lamb, halfpi-theta, psi);
        Lt=simpleCalculateLogLCAmpPhase(distt, phit, iotat, lambt, halfpi-thetat, psit);
        if abs(1-Lt/L)>1e-2:
            print("check: ",x,y,z,w,dist,Psim,Psip,T,math.sqrt(R),Delta,L)
            print("checkt:",xt,yt,zt,wt,distt,Psimt,Psipt,T,math.sqrt(Rt),Deltat,Lt)
            print(sa,se,sigp,sigm,abs(sa),abs(se),abs(sigp),abs(sigm),abs(sigm/sigp)**2,16*(abs(sigp)**2-abs(sigm)**2))
            print(sat,set,sigpt,sigmt,abs(sat),abs(set),abs(sigpt),abs(sigmt),abs(sigmt/sigpt)**2,16*(abs(sigpt)**2-abs(sigmt)**2))
        """
        pass
    #if xxcount%2000==0:print("cutfrac=",xxcut/xxcount)

        
    #store new params
    param[ipsi]=psit
    param[idist]=distt
    param[ibeta]=halfpi-thetat
    param[iinc]=iotat
    param[ilamb]=lambt
    param[iphi]=phit

    return ptmcmc.state(s,param);
    

#Exact-in-limit distance-altitude-polarization 2-D symmetry
dist_alt_pol_size=0.1;
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
    Delta=(1-x2)*(1-y2)/(dist**2*((1+math.sqrt(x))*(1+math.sqrt(y)))**4)

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
    psit=psi+dist_alt_pol_psi_size*randoms[0]*0
    c4psit=math.cos(4*psit)
    wt=sgn*w+dist_alt_pol_w_size*randoms[1]
    if wt>0:sgn=1
    else:sgn=-1
    wt=abs(wt)

    #reverse reparameterization and intermediates    
    T  = ( (wt+1/wt)/2 + c4psit*(1-R**ypow) )/R**ypow
    if True and  T>Tlimit: #limiting case near poles
        rootr=math.sqrt(R)
        if ypow>0:
            zt=1/(2*T)
            if wt>1:
                xt=1/rootr
                yt=2*T*rootr
            else:
                xt=2*T*rootr
                yt=1/rootr
        else:
            zt=2*T
            if wt<0:
                xt=rootr/2/T
                yt=rootr
            else:
                xt=rootr
                yt=rootr/2/T
        zt=xt*yt**ypow
        
    else:
        zt = T+sgn*math.sqrt(T*T-1)
        xt = math.sqrt(zt*wt)
        if xt<wt*1e-60: yt=R  #handle pathological case
        else: yt = (xt/wt)**ypow
    distt = math.sqrt((1-xt*xt)*(1-yt*yt)/Delta) / ((1+math.sqrt(xt))*(1+math.sqrt(yt)))**2

    #test
    #Rt=(xt**2+yt**2+xt*yt*2*c4psit)/(1+xt**2*yt**2+xt*yt*2*c4psit)
    #Deltat=(1-xt**2)*(1-yt**2)/(distt**2*((1+math.sqrt(xt))*(1+math.sqrt(yt)))**4)
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
    zt=zt+eps
    
    jac=w/(dist+eps)*(x4th+1/x4th)*(y4th+1/y4th)*(z-1/z)/(z+1/z+twoc4psi)
    jact=wt/(distt+eps)*(xt4th+1/xt4th)*(yt4th+1/yt4th)*(zt-1/zt)/(zt+1/zt+2*c4psit)

    #hacks
    eps=1e-60
    #Next line restricts to just cases where xy<<1 or xy>>1
    if abs(math.log(x*y+eps))<lncut or abs(math.log(xt*yt+eps))<lncut:
        return 1

    return abs(jac/jact)

    




