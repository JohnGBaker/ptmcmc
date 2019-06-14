#Code for estimating effective sample size of chain data
import numpy as np
import argparse
import sys

class chainData:
    def __init__(self,fname):
        self.basename,self.chainfilepath=self.get_basename(fname)
        self.read_chain(self.chainfilepath)
        print("Chain:", self.N,"steps, every",self.dSdN)
        
    def get_basename(self,fname):
        if(fname.endswith(".out")):
            #we assume ".out does not otherwise appear in the name... that
            #could check... if(fname.count(".out")>1):...
            basename=fname.replace(".out","")
            fname=fname.replace(".out","_t0.dat")
        elif(fname.endswith("_t0.dat")):
            basename=fname.replace("_t0.dat","")
        print ("basename="+basename)
        return basename,fname

    def read_chain(self, chainfilepath):
        print("reading data from :",chainfilepath)
        data=np.loadtxt(chainfilepath,converters={4: lambda s:-1})
        i0=0
        while(data[i0,0]<0):i0+=1
        data=data[i0:]
        self.dSdN=dSdN=data[4,0]-data[3,0]
        data=np.delete(data,[0,3,4,len(data[0])-1],1)
        self.N=N=len(data)
        print ("Data have ",N," rows representing ",N*dSdN," steps.")
        maxPost=max(data[:,1])
        self.data=data
        print ("data[1]=",data[1])
        parnames=self.get_par_names(self.chainfilepath)
        self.npar=len(parnames)
        print(self.npar,"params:",parnames)
        self.names=["post","like",]+parnames

    def get_par_names(self, fname):
        with open(fname) as f:
            line=f.readline()
            line=f.readline()
            names=line.split()
            names=names[5:]
        return names

    def getStep(self):
        return self.N*self.dSdN

    def getState(self,idx):
        i=int(idx/self.dSdN)
        return self.data[i,2:]

#Most code here has be directly adapted from autocorr code in chain.cc

#/A routine for processing chain history to estimate autocorrelation of
#/windowed chain segments
#/
#/For some feature function \f$f({\bf x})\f$ over the state space, we will
#/compute the correlation length \f$\rho\f$ over various segments 
#/\f$\mathcal{S}\f$of the chain history data. Given some \f$f\f$, 
#/the correltation length is defined as:
#/\f[
#/ \rho =  1 + 2*\sum_k^{n_{\mathrm{lag}}} \rho(\tau_i)
#/\f]
#/where \f$ \rho(\tau_k) \f$ is the autocorrelation with lag \f$\tau_k\f$
#/computed by
#/\f[
#    \rho(\tau) = \frac{\sum_{i\in\mathcal{S}}( f(x_i)-\bar f_{\mathcal{S},\tau})( f(x_{i-\tau})-\bar f_{\mathcal{S},\tau})}{\sum_{i\in\mathcal{S}}( f(x_i)-\bar f_{\mathcal{S},\tau})^2}.
#/\f]
#/We define the average symmetrically over the nominal data segment \f$\mathcal{S}\f$ and the lagged segment, specifically
#/\f[
  #   \bar f_{\mathcal{S},\tau}=\sum_{i\in\mathcal{S}}(f(x_i)+f(x_(i-\tau__)/2/N_{\mathcal{S}}
#/\f]
#                    Sum[ ( f(i)-avg[iwin,lag] )*( f(i-lag)-avg[iwin,lag] ) ]
  # covar[iwin,lag] = ---------------------------------------------------------
  #                                           count
#                Sum[covar[iwin,lag] + (avg-avg[iwin,lag])^2,{iwin in set}]
#    \rho(\tau) \= ----------------------------------------------------------
#                  Sum[covar[iwin,0] + (avg-avg[iwin,0])^2,{iwin in set}]
#/
#/is basednecessary to perform autocorrelation analysis are computed block-wise over a set of window subdomains \f$\mathcal{W}_i\f$ in the step history.  
def compute_autocovar_windows(chain, feature, width, nevery, burn_windows=3, loglag=True, max_lag=0, dlag=2.0**0.5):
  #inputs:
  #  chain           structured chain object
  #  feature         arbitrary feature functions. Assume state vars now
  #  width           int (>1)width in steps of each window
  #  nevery          int (>=1) sampling rate in steps for the correlation analysis
  #  burn_windows    int (>=1) number of initial windows to skip
  #  loglag          bool flag to use logarithmic lag spacing
  #  dlag            float (>=1.01) factor for logarithmic spacing of lag windows
  #  max_lag         int (<=burn_windows) maxiumum lag, in number of window widths
  #outputs:
  #  covar           [Nwin][Nlag] lag covariance covar[iwin][0] is variance
  #  means           [Nwin][Nlag] Array of per-window feature means. 
  #  counts          [Nwin][Nlag] Number of samples included.
  #  outwindows      [Nwin+1] window start index labels for the output rows
  #  outlags         [Nlag] lag-size labels for the output columns 
  #
  # Then to compute rho(lag) for each window:
  #
  #                Sum[covar[iwin,lag] + (avg-avg[iwin,lag])^2,{iwin in set}]
  #    rho(lag) = ----------------------------------------------------------
  #                  Sum[covar[iwin,0] + (avg-avg[iwin,0])^2,{iwin in set}]
  #
  #                    Sum[ ( f(i)-avg[iwin,lag] )*( f(i-lag)-avg[iwin,lag] ) ]
  # covar[iwin,lag] = ---------------------------------------------------------
  #                                           count
  #
  #   avg[iwin,lag] = Sum[f(i)+f(i-lag)]/2/count
  #
  # where in the last lines, the sum and avg is over all the points in iwin,
  # and, for simplicity, we have left out weighting by count[iwin,lag] in the
  # expression for rho.  It is not obvious, but happens to be true, that the
  # particular form of mean given in the last line allows the covariances to
  # be simply combined as in the first line.
  #
  # Notes:
  #   -length of averages will be Nwin+max_lag, with first "window" segment
  #    mean at averages[max_lag], and the earlier ones corresponding to the
  #    buffer windows
  #   -Each window variance is w.r.t its avg in averages[max_lag+iwin] 
  #   -if (loglag==true) lags are approximately logarithmically spaced 
  #   -for meaningful results need max_lag>=burn_windows
  #   -last entry for outwindows be next value, for use in sums
  #   -windows are aligned to end at latest step
  #   -everything will be done in units of nevery
  #   -(*feature) should generally return true; if false then the feature is
  #    understood to apply only to a subdomain of the state space and such
  #    points are not included in in nums or denoms
  #   -this is why denoms may depend on lag
  #   -it will probably be necessary to replace these "feature" functions with
  #    some kind of class objects, probably to be defined in states.hh
    
    ifeat=feature([i for i in range(chain.npar)])

    if(width<=1):width=2;
    if(nevery<1):nevery=1;
    swidth=int(width/nevery);
    width=swidth*nevery;
    if(burn_windows<1):burn_windows=1;
    if(max_lag==0 or max_lag>burn_windows):max_lag=burn_windows;
    if(dlag<1.01):dlag=1.01;

    #determine output structure
    ncount=chain.getStep();
    #print("ncount=",ncount)
    Nwin=int(ncount/width) - burn_windows;
    if(Nwin<0):Nwin=0;
  
    #set up output grid
    istart=ncount-Nwin*width;
    outwindows=[istart+i*width for i in range(Nwin+1)]
    if(loglag): #logarithmic
        outlags=[]
        outlags.append(0);
        fac=1.0;
        idx=1;
        while(idx<max_lag*swidth):
            outlags.append(nevery*idx);
            lastidx=idx;
            while(lastidx==idx):
                fac*=dlag;
                idx=int(fac);
        Nlag=len(outlags)
    else: #linear
        Nlag=int(swidth*burn_windows+1)
        outlags=[nevery*i for i in range(Nlag)]
  
    #print("Nwin,Nlag:",Nwin,Nlag)
    covar=np.zeros((Nwin,Nlag))
    means=np.zeros((Nwin,Nlag))
    counts=np.zeros((Nwin,Nlag),dtype=np.int)

    fdata=chain.data[:,2+ifeat]
    #populate output vectors
    print(Nwin," windows:",end='')
    for k in range(Nwin):
        print(k," ",end='')
        sys.stdout.flush()
        #We hold on to the zero-lag results
        f=np.zeros(swidth)
        idxmap=np.zeros(swidth,dtype=np.int)
        for i in range(swidth):
            idx=outwindows[k]+i*nevery;
            idxmap[i]=int(idx/chain.dSdN)
        f=fdata[idxmap]
        counts[k,0]=len(f);
        #print("f:",f)
        fsum=np.sum(f)
        means[k,0]=fsum/counts[k,0]
        covar[k,0]=np.mean(f**2)-means[k][0]**2
        for j in range(1,Nlag):
            ilag=outlags[j];
            #print("ilag=",ilag)
            idxmap=np.zeros(swidth,dtype=np.int)
            for i in range(swidth):
                idx=outwindows[k]+i*nevery-ilag;
                idxmap[i]=int(idx/chain.dSdN)
            lagf=fdata[idxmap]
            counts[k,j]=len(lagf)
            means[k,j]=(fsum+np.sum(lagf))/2./counts[k,j]
            xxsum=np.sum(lagf*f);
            #means[k][j]=xsum/counts[k][j]/2; #this is the avg of mean and lagged-mean
            covar[k][j]=xxsum/counts[k][j]-means[k][j]*means[k][j];#this equals covar wrt above mean
            #print("k,j,mean,covar:",k,j,means[k][j],covar[k][j])
    print()
    #print("means,counts:",means,counts)
    return covar,means,counts,outwindows,outlags

#Estimate effective number of samples for some feature of the state samples
def compute_effective_samples(chain, features, width, nevery, burn_windows=3, loglag=True, max_lag=0, dlag=2.0**0.5):  
     #inputs:
     #  features        functions which return feature values from a state.
     #  chain           chain object
     #  width           int (>1)width in steps of each window
     #  nevery          int (>=1) sampling rate in steps for the correlation analysis
     #  burn_windows    int (>=1) number of initial windows to skip
     #  dlag            float (>=1.1) factor for logarithmic spacing of lag windows
     #  max_lag         int (<=burn_windows) maxiumum lag, in number of window widths
     #outputs:
     #  effSampSize     the estimate for effective sample size     
     #  best_nwin       the number of windows which optimized effSampSize
     #
     #  This routine uses compute_autocovar_windows to compute autocorrelation
     #  lenghts for the state feature functions provided.  Then, downsampling
     #  by this length, computes the effective number of samples for each
     #  feature, taking the minimum of the set.  The routine does this repeatedly,
     #  considering a range of possible sample sizes, beginning at the end of the
     #  chain and working backward.  Generally, one expects to find an optimal
     #  effective length which maximizes the effective sample size as correlation
     #  lengths should be longer when earlier, more immature parts of the chain
     #  are included in the calculation.
     #
     # Notes:
     #   -the autocorrlength over best_nwin is ac_len=width*best_nwin/effSampSize
     #   -See also notes for compute_autocovar_windows
    
  
    #controls
    oversmall_aclen_fac=3.0;

    nf=len(features)

    covars=[]
    means=[]
    counts=[]
    esses=np.zeros(nf)
    acls=np.zeros(nf)
    best_esses=np.zeros(nf)
    best_acls=np.zeros(nf)
    best_means=np.zeros(nf)
    for i in range(nf):
        print("Computing window autocovariances for feature ",i)
        covar,mean,count,windows,lags = compute_autocovar_windows(chain,features[i],width,nevery,burn_windows, loglag, max_lag, dlag);
        covars.append(covar)
        means.append(mean)
        counts.append(count)
    Nwin=len(windows)-1;
    Nlag=len(lags);
    #print("ESS: Nwin=",Nwin," Nlag=",Nlag)

    lmin=0
    ess_max=0;
    nwin_max=0;
    #Loop over options for how much of the chain is used for the computation
    for nwin in range(1,Nwin+1):
        #Compute autocorr length
        #We compute an naive autocorrelation length
        # ac_len =  1 + 2*sum( corr[i] )
        ess=1e100;
        feature_means=np.zeros(nf)
        for ifeat in range(nf):
            #Compute sample mean.
            #print("shape:",means[ifeat].shape)
            #print("shape':",means[ifeat][-nwin:,0].shape)
            fmean=sum(means[ifeat][-nwin:,0])/nwin
            #print("len,fmean=",len(means[ifeat][-nwin:,0]),fmean)
            last_lag=0;
            ac_len=1.0;
            lastcorr=1;
            dacl=0;
            for ilag in range(1,Nlag):
                lag=lags[ilag];
                #compute the correlation for each lag
                num=0;
                denom=0;
                for iwin in range(Nwin-nwin,Nwin):
                    dmean=fmean-means[ifeat][iwin][ilag];
                    dmean0=fmean-means[ifeat][iwin][0];
                    cov=covars[ifeat][iwin][ilag]+dmean*dmean;
                    var=covars[ifeat][iwin][0]+dmean0*dmean0;
                    c=counts[ifeat][iwin][ilag];
                    num   += cov*c;
                    denom += var*c;
                    #print("cov,var:",cov,var)
                    #print(nwin," ",ifeat," ",ilag," ",iwin,": count,fmean,dmean,num,den=",c,", ",fmean,", ",dmean,", ",num,", ",denom)
                #print("num,denom:",num," ",denom)
                corr=num/denom;
                if(lastcorr<0 and corr<0):
                    #keep only "initally positive sequence" (IPS)
                    ac_len-=dacl;
                    break
                lastcorr=corr;
                dacl=2.0*(lag-last_lag)*corr;
                ac_len+=dacl;
                #print("nwin,lag,corr,acsum:",nwin," ",lag," ",corr," ",ac_len)
                last_lag=lag;

            #print("baselen=",nwin*width,"  aclen=",ac_len)
            #compute effective sample size estimate
            #esses[ifeat]=nwin*width/ac_len;
            essi=nwin*width/ac_len;
            #print("nwin,lag,aclen,effss:",nwin," ",last_lag," ",ac_len," ",essi)
            if(ac_len<nevery):
                #Ignore as spurious any
                print("aclen<nevery!: ",ac_len," < ",nevery)
                #esses[ifeat]=0;
                #We don't really trust this result, and so instead set ess supposing aclen=nevery*oversmall_aclen_fac
                essi=nwin*width/oversmall_aclen_fac/nevery;      
            # if(esses[ifeat]<ess){
            #ess=esses[ifeat];
            if(essi<ess):ess=essi
            esses[ifeat]=essi;
            acls[ifeat]=ac_len;
            feature_means[ifeat]=fmean;
            #cout,"nwin,ifeat=",nwin,",",ifeat,":  -> ",essi,endl;
        if(ess>ess_max):
            ess_max=ess;
            nwin_max=nwin;
            best_esses=esses;
            best_acls=acls;
            best_means=feature_means;
        print("len=",nwin,"*",width,": ",end='')
        for  ess in esses:print(" ",ess,end='') 
        print("\n              : ",end='')
        for  x in acls:print(" ",x,end='') 
        print()
    best_nwin=nwin_max;
    effSampSize=ess_max;

    #dump info
    print("best:")
    print("len=",nwin_max,"*",width,": ",end='')
    for  ess in best_esses:print(" ",ess,end='')
    print("\n              : ",end='')
    for  x in best_acls:print(" ",x,end='') 
    print("\n     means = ",end='')
    for val in best_means:print(" ",val,end='')
    print()

    #For testing, we dump everything and quit
    #static int icount=0;
    #icount++;
    #cout,"COUNT=",icount,endl;
    #if(false and not loglag):
    #ofstream os("ess_test.dat");
    #for(int iwin=Nwin-best_nwin-burn_windows;iwin<Nwin;iwin++){
    # for(int i=0;i<int(width/nevery);i++){
    #  int idx=windows[iwin]+i*nevery;
    #  double fi;
    #  if((*features[0])(getState(idx),fi)){
    #  os<<idx<<" "<<fi<<endl;
    # }
    #}
    #cout<<"aclen="<<best_nwin*width/ess_max<<endl;
    #cout<<"range="<<best_nwin*width<<endl;
    #cout<<"ess="<<ess_max<<endl;
    #cout<<"burnfrac="<<burn_windows/(1.0*best_nwin+burn_windows)<<endl;
    #cout<<"Quitting for test!"<<endl;
    #exit(0);
    return effSampSize,best_nwin



#Report effective samples
#Compute effective samples for the vector of features provided.
#Report the best case effective sample size for the minimum over all features
#allowing the length of the late part of the chain under consideration to vary
#to optimize the ESS.
#Returns (ess,length)
def report_effective_samples(chain, features,width,nevery):
  burn=2;
  i=1;
  ess,nwin=compute_effective_samples(chain, features, width, nevery, burn, True,0,1.1);
  print("Over ",len(features)," pars: ess=",ess,"  useful chain length is: ",width*nwin," autocorrlen=",width*nwin/ess)
  return ess,width*nwin


#Report effective samples
#This is simplified interface producing effective sample estimates for each parameter
def report_param_effective_samples(chain, imax=None,width=None, every=None):
    if(every is None):every=chain.dSdN
    if(width is None):width=every*1000
    while(width<chain.getStep()*0.05):width*=2;
    if(imax is None or imax<0):imax=chain.npar;
    if(imax>chain.npar):imax=chain.npar;
    def getparam(i):return lambda x:x[i]
    features=[getparam(i) for i in range(imax)]
    print("data[0] features:",[features[i](chain.getState(0)) for i in range(imax)])
    return report_effective_samples(chain,features,width,every)

#Resample Posterior
#This routine draws samples from the part of the data found relevant, and reports data with nup times the ess from the relevant portion of the data
def resample_posterior(chain, ess, length, upsample_fac):
    fname=chain.basename+"_resampled.dat"
    print("Writing resampled data to '"+fname+"'")
    nsamp=int(ess*upsample_fac)
    ngood=int(length/chain.dSdN)
    if(nsamp>ngood):nsamp=ngood
    n0=int(len(chain.data)-ngood)
    rows=n0+np.random.choice(int(ngood),nsamp)
    with open(fname,"w") as f:
        f.write("#ess="+str(ess)+" nsamp="+str(nsamp)+"\n")
        f.write("#")
        for name in chain.names:f.write(name+" ")
        f.write("\n")
        for i in rows:
            rowvals=chain.data[i]
            for val in rowvals:f.write(str(val)+" ")
            f.write("\n")

################# MAIN ################

parser = argparse.ArgumentParser(description='Compute best effective samples and autocorrelation from chain file.')
parser.add_argument('fname', metavar='chain_file', type=str, 
                    help='chain file path')
args = parser.parse_args()

chain=chainData(args.fname)
ess,length=report_param_effective_samples(chain)
upsample_fac=5
resample_posterior(chain,ess,length,upsample_fac)

