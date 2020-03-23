#ptmcmc analysis support
#John Baker (2019)
#Find at github.com/johngbaker/ptmcmc
#
#Provides class for holding chain data

import sys
import os
import numpy as np
import math
import subprocess
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import ess as esspy
#import matplotlib
#matplotlib.use('TkAgg')

#from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.widgets import Slider, Button, RadioButtons
filestyle=0

useLikeDefault=False
noPostDefault=False

class chainData:
    def __init__(self,fname,noPost=None,useLike=None):
        if noPost is not None:
            self.noPost=noPost #set true if data does not include Post and Like in first two cols
        else:self.noPost=noPostDefault
        if useLike is not None:
            self.useLike=useLike #set true to retain like if present
        else:self.useLike=useLikeDefault
        self.have_ess=False
        print('noPost=',self.noPost)
        self.basename,self.chainfilepath,self.filestyle=self.get_basename(fname)
        self.read_chain(self.chainfilepath)
        print("Chain:", self.N,"steps, every",self.dSdN)
        
    def get_basename(self,fname):
        filestyle=0 #ptmcmc style
        if(fname.endswith(".out")):
            #we assume ".out does not otherwise appear in the name... that
            #could check... if(fname.count(".out")>1):...
            basename=fname.replace(".out","")
            fname=fname.replace(".out","_t0.dat")
        elif(fname.endswith("_t0.dat")):
            basename=fname.replace("_t0.dat","")
        elif(fname.endswith(".dat")):
            basename=fname.replace(".dat","")
            filestyle=1 #general style (post,like, pars,...) by default
            print('general style detected')
        print ("basename="+basename)
        return basename,fname,filestyle

    def read_chain(self, chainfilepath):
        print("Reading data from :",chainfilepath)
        i0=0
        if self.filestyle==0: #ptmcmc style
            data=np.loadtxt(chainfilepath,converters={4: lambda s:-1})
            if self.noPost: print("Warning noPost incompatible with ptmcmc chainstyle!")
            while(data[i0,0]<0):i0+=1
            data=data[i0:]
            self.dSdN=data[4,0]-data[3,0]
            parnames=["post",]+self.get_par_names(chainfilepath)
            data=np.delete(data,[3,4,len(data[0])-1],1)
            parnames=["samp","post",]
            if self.useLike:
                parnames+=['like']
                self.ipar0=3
            else:
                self.ipar0=2
                data=np.delete(data,[2],1)
            parnames+=self.get_par_names(chainfilepath)
        elif self.filestyle==1:
            data=np.loadtxt(chainfilepath)
            self.dSdN=1
            N=len(data)
            data=np.hstack((np.reshape(range(N),(N,1)),data))
            if self.noPost: #pars only
                parnames=["samp"]+self.get_par_names(chainfilepath,startcol=0)
                self.ipar0=1
            else: #post, like, pars
                havelike=False
                varparnames=self.get_par_names(chainfilepath,startcol=0)
                if 'like' in varparnames: 
                    varparnames=varparnames[varparnames.index('like')+1:]
                    havelike=True
                if 'post' in varparnames: 
                    varparnames=varparnames[varparnames.index('post')+1:]
                parnames=["samp","post",]
                if havelike:
                    if self.useLike:
                        parnames+=['like']
                        self.ipar0=3
                    else:
                        data=np.delete(data,[1],1)
                        self.ipar0=2
                else: 
                    self.ipar0=2
                parnames+=varparnames
        self.npar=len(parnames)-self.ipar0
        self.names=parnames
        self.N=len(data)
        print ("Data have ",self.N," rows representing ",self.N*self.dSdN," steps.")
        #if "post" in parnames:
        #    self.maxPost=max(data[:,parnames.index("post")])
        self.data=data
        print ("data[1]=",data[1])
        print(self.npar,"params:",parnames)

    def get_par_names(self,fname,startcol=5):
        with open(fname) as f:
            names = read_par_names(f,startcol)
        if names is None:
            names=['p'+str(i) for i in range(len(line1.split()))]
        return names

    def getSteps(self):
        return self.N*self.dSdN

    def getState(self,idx): #fixme for non-filetype==0
        i=int(idx/self.dSdN)
        return self.data[i,self.ipar0:]

    def get_samples(self,nsamp,good_length):
        ngood=int(good_length/self.dSdN)
        if(nsamp>ngood):nsamp=ngood
        n0=int(len(self.data)-ngood)
        rows=n0+np.random.choice(int(ngood),nsamp)
        i0=self.names.index('post')+1
        if self.useLike: i0+=1
        return self.data[rows,i0:]

    def readCovar(self,filename=None):
        if filename is None:
            filename=self.basename+"_covar.dat"
        pars=[]
        done=False
        with open(filename,'r') as f:
            names=read_par_names(f)
            idxs=[]
            try:
                for name in names:
                    tryname=name
                    idxs.append(self.parnames.index(name))
            except:
                if names[0]!='p0': print("Failed to find '"+tryname+"' among chain's parnames.")
                print("Assuming params align with chain params")
                idxs=range(len(names))
                idxs=[self.ipar0+idx for idx in idxs]
            line=f.readline()
            while(not "#Covariance" in line): line=f.readline() #Skip until the good stuff
            covar=np.zeros((self.npar,self.npar))
            for iidx in idxs:
                line=f.readline()
                print(i,":",line)
                elems=np.array(line.split())
                for j,val in elems:
                    covar[iidx,idxs[j]]=val
        return covar

    def estimate_ess(self,esslimit=10000):
        if not self.have_ess: 
            ess,length=esspy.report_param_effective_samples(self,esslimit=esslimit)
            self.ess=ess
            self.acl=(1.0*length)/ess
            print("ess,acl:",self.ess,self.acl)
            self.have_ess=True
        return self.ess,self.acl
    
    def KLdivergence(self,other,upsample=1,esslimit=10000):
        ess,acl=self.estimate_ess(esslimit=esslimit)
        nP=int(ess*upsample)
        #print("nP:",ess*upsample,nP,ess,acl)
        length=int(ess*acl)
        samplesP=self.get_samples(nP,length)
        ess,acl=other.estimate_ess(esslimit=esslimit)
        nQ=int(ess*upsample)
        length=int(ess*acl)
        samplesQ=other.get_samples(nQ,length)
        #print("nP,nQ:",nP,nQ)
        return KL_divergence(samplesP,samplesQ)

    def fakeKLdivergence(self,other,upsample=1,esslimit=10000):
        ess,acl=self.estimate_ess(esslimit=esslimit)
        nP=int(ess*upsample)
        #print("nP:",ess*upsample,nP,ess,acl)
        length=int(ess*acl)
        samplesP=self.get_samples(nP,length)
        ess,acl=other.estimate_ess(esslimit=esslimit)
        nQ=int(ess*upsample)
        length=int(ess*acl)
        samplesQ=other.get_samples(nQ,length)
        #print("nP,nQ:",nP,nQ)
        i0=self.names.index('post')+1
        #print('pars:',self.names[i0:])
        return fake_KL_divergence(samplesP,samplesQ)
        
        
#####################################
#general functions
                                       
def read_par_names(f,startcol=0):
    pos=f.tell()
    line1=f.readline()
    if(line1[0]=='#'):
        pos=f.tell()
        line2=f.readline()
        if(line2[0]=='#'):
            names=line2.split()
            names=names[startcol:]
            return names
        else:
            f.seek(pos)
            names=line1.split()
            names=names[startcol:]
            return names
    else:
        f.seek(pos)
        names=['p'+str(i) for i in range(len(line1.split()))]
    return names


def KL_divergence(samplesP, samplesQ):
    '''
    ///Interesting to apply to both target samples and transformed samples
    /// Implements eqn 2 of Perez-Cruz 2008 "Kullback-Leibler Divergence Estimation of Continuous Distributions"
    /// also https://www.princeton.edu/~kulkarni/Papers/Journals/j068_2009_WangKulVer_TransIT.pdf
    /// with k=1. We infer that the sign is wrong on the sum term there.
    '''
    k=2;
    result=0;
    N=len(samplesP);
    r1sqs=all_nnd2(samplesP);
    #print('r1sqs',r1sqs)
    s1sqs=np.zeros(N);
    for i in range(N):s1sqs[i]=one_nnd2(samplesP[i],samplesQ);
    #print('s1sqs',s1sqs)

    #Here we put a floor on the smallest value of all NN distances
    #based on the kfloorth smallest distance within the P set
    kfloor=5;
    if(kfloor>0):
      dists=r1sqs;
      sdists=np.sort(dists)
      floor=dists[kfloor+1]
      r1sqs[r1sqs<floor]=floor
      s1sqs[s1sqs<floor]=floor
    
    result+=-sum(np.log(r1sqs/s1sqs))
    dim=len(samplesP[0])
    M=len(samplesQ);
    result *= (0.5*dim)/N;#//factor of 1/2 because we use nearest neighbor dist^2
    result += np.log(M/(N-1.0));
    return result;
  
def all_nnd2(samples):
    #Computation of all nearest neighbor distances, brute force.
    N=len(samples)
    nni=[-1]*N
    nnd2=np.zeros(N)-1
    for i in range(N):
        for j in range(i+1,N):
            #print("i,j=",i,j)
            diff=samples[i]-samples[j]
            dist2=np.dot(diff.T,diff)
        if (nnd2[i]<0 or nnd2[i]>dist2):
            nni[i]=j
            nnd2[i]=dist2;
        if(nnd2[i]<0 or nnd2[j]>dist2):
            nni[j]=i
            nnd2[j]=dist2;
    return nnd2

def one_nnd2(x,samples):
    #Computation of nearest neighbor distance, brute force.
    N=len(samples)
    i0=-1
    for i in range(N):
        diff=samples[i]-x
        dist2=np.dot(diff.T,diff)
        if(i0<0 or nnd2>dist2):
            i0=i
            nnd2=dist2;
    return nnd2

def KLdivergence_Wang09(samplesP, samplesQ, k=1):
    """ 
    KL-Divergence estimator based on Wang09:
    https://www.princeton.edu/~kulkarni/Papers/Journals/j068_2009_WangKulVer_TransIT.pdf
    Using brute-force kNN
    k: Number of neighbours considered (default 1)
    """

    n, m = len(samplesP), len(samplesQ)
    KLdiv = np.log(m / (n - 1.0))
    d = float(s1.shape[1])

    for p1 in s1:
        nu = np.sort(np.linalg.norm(s2-p1, axis=1))[k-1]
        rho = np.linalg.norm(s1-p1, axis=1)[k]
        D += (d/n)*np.log(nu/rho)
    return D

def knn_distance(point, sample, k):
    """ Euclidean distance from `point` to it's `k`-Nearest
    Neighbour in `sample` """
    norms = np.linalg.norm(sample-point, axis=1)
    return np.sort(norms)[k]
                                                                       
def get_sample_cov(samples):
    N=len(samples)
    dim=len(samples[0])
    cov=np.zeros((dim,dim));
    mean=samples.mean(axis=0)
    out_mean=np.copy(mean)
    ssum=np.zeros(dim)
    for  s in samples:
        for j in range(dim):
            jdiff=s[j]-mean[j]
            cov[j,j]+=jdiff*jdiff
            for i in range(j+1,dim):
                idiff=s[i]-mean[i]
                val=idiff*jdiff
                cov[i,j]+=val
                cov[j,i]+=val
    cov/=(N-1.0)
    return cov,out_mean

def fake_KL_divergence(samplesP, samplesQ, verbose=False):
    #This applies a simplified alternative to the KL divergence (which is difficult to compute accurately from samples).
    #The calculation is based on the means and variances of the two samples and would agree with the KL diverences
    #for large samples of Gaussian distributions.
    #The KL-divergence between two Gaussians is
    # 2 KLdiv(P,Q) = Tr [ cov(P) cov(Q)^-1 ]- dim - log | cov(P) cov(Q)^-1 | - (mu(P)-mu(Q))^t cov(Q)^-1 (mu(P)-mu(Q))^t
    covP,meanP=get_sample_cov(samplesP)
    covQ,meanQ=get_sample_cov(samplesQ)
    dim=len(covP)
    if verbose:
        print("meanP:",[x for x in meanP])
        print("meanQ:",[x for x in meanQ])
        print("sigmaP:",[covP[i,i] for i in range(dim)])
        print("sigmaQ:",[covQ[i,i] for i in range(dim)])

    nQ=len(samplesQ)
    nP=len(samplesP)
    unbiasing_factor=(nQ-dim-2.0)/(nQ-1.0) #The final factor is to make unbiased for finite nQ, assuming nQ=nP
    #unbiasing_factor=1
    invCovQ=np.linalg.pinv(covQ)*unbiasing_factor
    covPinvCovQ=np.matmul(covP,invCovQ)
    dmu=meanP-meanQ
    result=0
    result += -dim + covPinvCovQ.trace();
    s,val = np.linalg.slogdet(covPinvCovQ/unbiasing_factor)
    result+=-val
    result += np.dot(np.dot(dmu,invCovQ),dmu)
    result +=- (dim + covPinvCovQ.trace())/nP;
    #result += (0.5*dim*(dim+1)+1)*(1.0/nP-1.0/nQ) - covPinvCovQ.trace()/nP;
    return 0.5*result;
  
#Read in a set of chain files
def read_all_chains(names):
    global allparnames,Nmax,Smax
    datanames=[]
    Nmax=0
    Smax=0
    allparnames=[]
    allchains=[]
    
    for fname in names:
        chain=chainData(fname)
        #print("chain:",chain)
        allchains.append(chain)
        #print("allchains-->",allchains)
        if(chain.N>Nmax):Nmax=chain.N
        S=chain.getSteps()
        if(S>Smax):Smax=S
        #maxPost=max(data[:,1])
        print (chain.names)
        for name in chain.names[1:]:
            if(not name in allparnames):
                allparnames.append(name)
    return allchains

#make sample names from the most relevant part of the filenames
def make_short_labels(files):
    longnames=[os.path.basename(filename) for filename in files]
    if(len(set(longnames))<len(longnames)):
        longnames=[os.path.basename(os.path.dirname(os.path.abspath(filename))) for filename in files]
    si=0;ei=1
    sgood=True;egood=True
    if(len(longnames)>1):
        for c in longnames[0]:
            #print("sitest",[name[si] for name in longnames])
            #print("s set:",set([name[si] for name in longnames]))
            if(sgood and len(set([name[si] for name in longnames]))==1):
                si+=1
            else:sgood=False
            #print("eitest",[name[-ei] for name in longnames])
            #print("e set:",set([name[-ei] for name in longnames]))
            if(egood and len(set([name[-ei] for name in longnames]))==1):
                ei+=1
            else:egood=False
            #print("si,ei=",si,ei)
            if(not (sgood or egood)):break
    #print([si+ei-len(name) for name in longnames])
    if(np.min([si+ei-len(name) for name in longnames])<0):
        si=0
        ei=1
    if(ei<=1):
        sample_labels=[name[si:] for name in longnames]
    else:
        sample_labels=[name[si:-(ei-1)] for name in longnames]
    #print("si/ei:",si,ei)
    print(sample_labels)
    return sample_labels

#get an appropriate set of data for a plot
def get_xydata(data,i,j,dens,samps):
    d=data[data[:,0]>samps]
    #d=d[d[:,6]>4]
    Nd=len(d)
    #print("Reduced data len =",Nd)
    every=max([1,int(Nd/dens)])
    #print(i,j,every)
    #print(Nd,dens,every)
    x=d[::every,i]
    y=d[::every,j]
    #print (len(x))
    return x,y
    
##################
#Widget functions
##################
import matplotlib.patches as mpatches

class viewer:
    def __init__(self,fnames,selectX=False):
        self.fnames=fnames
        self.labels=make_short_labels(fnames)
        self.allchains=read_all_chains(self.fnames)
        #print('allchains:',self.allchains)
        
        self.fig, self.ax = plt.subplots()
        if selectX:
            leftlim=0.35
            bottomlim=0.30
        else:
            leftlim=0.25
            bottomlim=0.25
        plt.subplots_adjust(left=leftlim, bottom=bottomlim)

        cx0 = 1
        cy0 = 0
        s0 = 3
        d0 = 2
        
        x,y=get_xydata(self.allchains[0].data,cx0,cy0,10**d0,10**s0)
        try:
            cmapname='tab10'
            self.cmap = matplotlib.cm.get_cmap(cmapname)
        except ValueError:
            cmapname='Vega10'
            self.cmap = matplotlib.cm.get_cmap(cmapname)
        self.cmap_norm=10
        self.scat = plt.scatter(x, y, s=1, c=x, cmap=cmapname,norm=colors.Normalize(0,self.cmap_norm))
        #scat = plt.scatter([], [], s=1,cmap="tab10")

        axcolor = 'lightgoldenrodyellow'
        axstart = plt.axes([leftlim, 0.1, 0.9-leftlim, 0.03])
        axdens = plt.axes([leftlim, 0.15, 0.9-leftlim, 0.03])
        height=(len(allparnames)-1)*0.05
        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        if not selectX:
            #rax = plt.axes([0.025, 0.5-height/2, 0.1, height], facecolor=axcolor)
            rYax = plt.axes([0.025, 0.5-height/2, 0.1, height])
        else:
            rXax = plt.axes([0.025, 0.5-height/2, 0.1, height])
            rXax.text(0.9, 0.95, "X", transform=rXax.transAxes, fontsize=11,
                      verticalalignment='top',horizontalalignment='right')
            rYax = plt.axes([0.15, 0.5-height/2, 0.1, height])
            rYax.text(0.9, 0.95, "Y", transform=rYax.transAxes, fontsize=11,
                      verticalalignment='top',horizontalalignment='right')

        ilogS=(math.log10(Smax))
        ilogSamps=int(math.log10(Nmax))
        print("ilogSmax=",ilogS)
        
        #Start slider
        print('axstart',axstart)
        self.sstart = Slider(axstart, 'log-Start', 2, ilogS, valinit=s0)
        self.sstart.on_changed(self.update)

        #Density slider
        self.sdens = Slider(axdens, 'log-Density', 1, ilogSamps, valinit=d0)
        self.sdens.on_changed(self.update)

        #X/y-axis radio buttons
        if selectX:
            self.radioX = RadioButtons(rXax, allparnames, active=cx0)
            self.radioY = RadioButtons(rYax, allparnames, active=cy0)
            parnameswidth=max([len(x) for x in allparnames])
            fontsize=self.radioX.labels[0].get_fontsize()/max([1,parnameswidth/5.])
            #print("fontsize=",fontsize)
            for label in self.radioX.labels:
                label.set_fontsize(fontsize)
            for label in self.radioY.labels:
                label.set_fontsize(fontsize)
            for circle in self.radioX.circles: # adjust radius here. The default is 0.05
                circle.set_radius(0.03)
            self.radioX.on_clicked(self.update)
            self.haveX=True
            #print('set radio')
        else:
            self.radioY = RadioButtons(rYax, allparnames, active=0)
            parnameswidth=max([len(x) for x in allparnames])
            fontsize=self.radioY.labels[0].get_fontsize()/max([1,parnameswidth/5.])
            #print("fontsize=",fontsize)
            for label in self.radioY.labels:
                label.set_fontsize(fontsize)
            self.haveX=False
        for circle in self.radioY.circles: # adjust radius here. The default is 0.05
            circle.set_radius(0.03)
        self.radioY.on_clicked(self.update)

        #Reset button
        self.button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
        self.button.on_clicked(self.reset)

        #print('calling update')
        self.update()
        #print('save fig')
        #plt.savefig('junk.png')
        #print('calling show')
        plt.show()
        #print('finished init')
    
    
    def update(self,val=0): #argument is not used
        #c0=1+allparnames[1:].index(radio.value_selected)
        #print("index->",c0)
        start = int(10**self.sstart.val)
        samps = int(10**self.sdens.val)
        xmin=1e100;xmax=-1e100
        ymin=1e100;ymax=-1e100
        xy=np.array([])
        cc=np.array([])
        ic0=0.5;
        ind=0
        plotlabels=[]
        for chain in self.allchains:
            includeChain=True
            if self.haveX:
                if(self.radioX.value_selected in chain.names):
                    cx=chain.names.index(self.radioX.value_selected)
                else: includeChain=False
            else: cx=0
            if(self.radioY.value_selected in chain.names):
                cy=chain.names.index(self.radioY.value_selected)
            else: includeChain=False
            if includeChain:
                x,y=get_xydata(chain.data,cx,cy,samps,start)
                n=len(x)
                #xy=np.array([xyi for xyi in xy if np.all(np.isfinite(xyi))])
                colorval=ic0+ind
                if(cc.size>0):
                    cc = np.concatenate((cc,[colorval]*n))
                    xy = np.vstack((xy,np.vstack((x, y)).T))
                else:
                    cc=np.array([colorval]*n)
                    xy = np.vstack((x, y)).T
                if(n==0):
                    ind+=1
                    continue
                lim=x.min()
                if(xmin>lim):xmin=lim
                lim=x.max()
                if(xmax<lim):xmax=lim
                lim=y.min()
                if(ymin>lim):ymin=lim
                lim=y.max()
                if(ymax<lim):ymax=lim
                plotlabels.append(mpatches.Patch(color=self.cmap(colorval/self.cmap_norm), label=self.labels[ind],hatch='.'))
                ind=ind+1            

        self.scat.set_offsets(xy)
        self.scat.set_array(cc)
        self.ax.set_xlim(xmin-0.1*(xmax-xmin),xmax+0.1*(xmax-xmin))
        self.ax.set_ylim(ymin-0.1*(ymax-ymin),ymax+0.1*(ymax-ymin))
        self.ax.legend(handles=plotlabels,fontsize=8)

        for tick in self.ax.get_xticklabels():
            tick.set_rotation(20)

        self.fig.canvas.draw_idle()
   
    def reset(self,event): #fixme
        self.allchains=read_all_chains(self.fnames)
        self.sstart.reset()
        self.sdens.reset()

###################
#main code
###################


if __name__ == "__main__":
    ##########
    # Arguments and argument injest
    # determine basename from first argument.

    parser = argparse.ArgumentParser(description='Provide snapshot of chain state.')
    parser.add_argument('fname', metavar='chain_file', nargs='+', type=str, 
                        help='chain file path')
    parser.add_argument('-uselike',action='store_true',help='Include the likelihood')
    parser.add_argument('-noPost',action='store_true',help='Data has no Posterior or Likelihood in first columns ')
    parser.add_argument('-selectX',action='store_true',help='View two param projections')
    args = parser.parse_args()
    print(args)


    useLikeDefault=args.uselike
    noPostDefault=args.noPost
    viewer(args.fname,selectX=args.selectX)

