import ptmcmc
import test_proposal
import numpy as np
import matplotlib.pyplot as plt
import sys
import time


test=test_proposal.test()
npar=3;
space=ptmcmc.stateSpace(dim=npar);
names=["m1","m2","a1","a2","tc","dist","inc","phi","lamb","beta","psi"];
#space.set_names(names);
#PI=np.pi
#space.set_bound('phi',ptmcmc.boundary('wrap','wrap',0,2*PI));#set 2-pi-wrapped space for phi. 
#space.set_bound('lamb',ptmcmc.boundary('wrap','wrap',0,2*PI));#set 2-pi-wrapped space for lambda. 
#space.set_bound('psi',ptmcmc.boundary('wrap','wrap',0,PI))

dmu=0.5
dsigmasq=2.0

mu1=np.zeros(npar)
mu2=np.zeros(npar)
mu2[0]+=dmu

sigs1=np.array([1+x for x in range(npar)])
sigs2=np.copy(sigs1)
sigs1[0]+=dsigmasq
print('mu,sigma P:',mu1,np.sqrt(sigs1[0]))
print('mu,sigma Q:',mu2,np.sqrt(sigs2[0]))

sig2AoverB=(1.0*sigs1[0]/sigs2[0])
expectedKL=0.5*(sig2AoverB-1-np.log(sig2AoverB)+dmu*dmu/sigs2[0]) 
print("Theoretical result:",expectedKL)

nall=int(sys.argv[1])
if(len(sys.argv)>2):n1overn2=float(sys.argv[2])
else:n1overn2=1
n1=int(np.sqrt(nall*n1overn2))
n2=int(np.sqrt(nall/n1overn2))
if(len(sys.argv)>3):approxNN=(sys.argv[3]=='1' or sys.argv[3]=='True' or sys.argv[3]=='true')
else: approxNN=False

ntry=12000
kls=[]
fkls=[]
start=time.time()
kltime=0
fkltime=0
fake_only=True
for i in range(ntry): 
    samps1=np.random.multivariate_normal(mu1,np.diag(sigs1),n1)
    samps2=np.random.multivariate_normal(mu2,np.diag(sigs2),n2)
    sampstates1=[ptmcmc.state(space,samps1[i,:].tolist()) for i in range(n1)]
    sampstates2=[ptmcmc.state(space,samps2[i,:].tolist()) for i in range(n2)]
    start=time.time()
    if(not fake_only):kls.append(test.KL_divergence(sampstates1,sampstates2,approxNN))
    kltime-=start;start=time.time();kltime+=start
    fkls.append(test.fake_KL_divergence(sampstates1,sampstates2))
    fkltime-=start;start=time.time();fkltime+=start

#plt.scatter(samps1[:,0],samps1[:,1])
#plt.scatter(samps2[:,0],samps2[:,1])
#plt.show()
if(not fake_only):
    std=np.std(kls)
    mean=np.mean(kls)
    print('n=('+str(n1)+','+str(n2)+') approx='+str(approxNN) )
    print('KL test ('+str(ntry)+' trials) :',mean,'+/-',std/np.sqrt(ntry*1.0),"err=",mean-expectedKL,"stderr=",std) 
    print('time=',kltime)

std=np.std(fkls)
mean=np.mean(fkls)
print('n=('+str(n1)+','+str(n2)+') approx='+str(approxNN) )
print('KL test ('+str(ntry)+' trials) :',mean,'+/-',std/np.sqrt(ntry*1.0),"err=",mean-expectedKL,"stderr=",std) 
print('time=',fkltime)
