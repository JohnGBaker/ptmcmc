#Usage:
#python comparechains.py chainfile_1 chainfile_2 ... chainfile_N fisherfile_1 fisherfile_2 ... fisherfile_N

import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import corner_with_covar as corner
import matplotlib.pyplot as plt
#from matplotlib.ticker import MultipleLocator
#from scipy.interpolate import interp1d
#from scipy.ndimage import gaussian_filter1d
import sys
import os.path
import re
import astropy.units as units
import math
import random
import argparse
import ptmcmc_analysis

def readCovar(filename=None):
    with open(filename,'r') as f:
        line=f.readline()
        while(not "State" in line):
            print('skipped line',line)
            line=f.readline() #Skip until the good stuff
        
        pars=[float(x) for x in line.split()[1:]]
        print('State pars:',pars)
        while(not "#Covariance" in line): line=f.readline() #Skip until the good stuff
        line=f.readline()
        npar=len(line.split())
        covar=np.zeros((npar,npar))
        for iidx in range(npar):
            if(iidx>0):line=f.readline()
            print(iidx,":",line)
            elems=np.array(line.split())
            print('elems',elems)
            for j,val in enumerate(elems):
                covar[iidx,j]=val
    return pars,covar

def get_par_names(fname,par0col=None):
    if par0col is None:
        par0col=5
        if("resampled.dat" in fname):par0col=2
        
    with open(fname) as f:
        line=f.readline()
        line=f.readline()
        names=line.split()
        names=names[par0col:]
    return names


def cornerFisher(Npar,parnames,parvals,samples,in_cov,cred_levs,selectedPars=None,sample_labels=None):
    if selectedPars is None: selectedPars=parnames
    print("selecting pars:",selectedPars,'from',parnames)
    selection=[parnames.index(par) for par in selectedPars]
    print('selection=',selection)
    data=[]
    for datum in samples:
        data.append(datum[:,selection])
    cov=[]
    if in_cov is not None:
        if not isinstance(in_cov,list):in_cov=[in_cov] 
        print('in_cov:',in_cov)
        for acov in in_cov:
            cov+=[acov[selection][:,selection]]
    print('cov:',cov)
    if parvals is not None:
        pars=np.array(parvals)[selection]
    else: pars=None
    names=[name for name in parnames if name in selectedPars]
    fontscale=0.1+Npar/9.0
    #print "cov=",cov
    # Plot it.
    #levels = 1.0 - np.exp(-0.5 * np.linspace(1.0, 3.0, num=3) ** 2)
    levels = cred_levs
    print ("levels=",levels)
    #set plot range limits
    if(False):
        rangelimits=[corner_range_cut for x in range(Npar)] #precentile cuts
    else:
        rangelimits=[[0,0]]*Npar #explicit cuts.
        q = [0.5 - 0.5*corner_range_cut, 0.5 + 0.5*corner_range_cut]
        datum=data[0]
        for i in range(Npar):
            #print('datum shape',datum.shape)
            rangelimits[i] = list(corner.quantile(datum[:,i], q))
        for datum in data[1:]:
            for i in range(Npar):
                #print("i=",i," range=",rangelimits[i])
                xmin, xmax =  corner.quantile(datum[:,i], q)
                if( xmin < rangelimits[i][0] ):rangelimits[i][0]=xmin
                if( xmax > rangelimits[i][1] ):rangelimits[i][1]=xmax
                #print("    range-->",rangelimits[i])
        #Now enlarge just a bit
        rangelimits=[[xmin-0.1*(xmax-xmin),xmax+0.1*(xmax-xmin)] for xmin,xmax in rangelimits ]
    print("rangelimits=",rangelimits)

    figure = corner.corner(data[0], bins=50,labels=names,levels=levels,cov=None,
                             truths=pars,quantiles=[0.159, 0.5, 0.841],show_titles=True,range=rangelimits,use_math_text=True,
                             #title_args={"fontsize": 35},title_fmt='.2e',smooth1d=None,smooth=1,label_kwargs={"fontsize":30},hist_kwargs={"normed":True})
                             title_args={"fontsize": 35},title_fmt='.2e',smooth1d=None,smooth=1,label_kwargs={"fontsize":30},hist_kwargs={"density":True})
    count=0
    print ("nsamp[0] = ",len(data[0]))
    colors=["k","r","b","g","m","c","y"]*2
    #Plot samples from additional files
    for datum in data[1:]:
        count+=1
        print ("nsamp[i] ",len(datum))
        figure = corner.corner(datum, bins=50,labels=names,fig=figure,color=colors[count],levels=levels,cov=None, plot_density=False,
                               truths=pars,quantiles=[0.159, 0.5, 0.841],show_titles=True,range=rangelimits,use_math_text=True,
                               #title_args={"fontsize": 35},title_fmt='.2e',smooth1d=None,smooth=1,label_kwargs={"fontsize":30},hist_kwargs={"normed":True})
                               title_args={"fontsize": 35},title_fmt='.2e',smooth1d=None,smooth=1,label_kwargs={"fontsize":30},hist_kwargs={"density":True})
    #Plot covariance ellipses
    for acov in cov:
        count+=1
        figure = corner.corner(None, bins=50,labels=names,fig=figure,color=colors[count],levels=levels,cov=acov, plot_density=False,
                               truths=pars,quantiles=[0.159, 0.5, 0.841],show_titles=True,range=rangelimits,use_math_text=True,
                               #title_args={"fontsize": 35},title_fmt='.2e',smooth1d=None,smooth=1,label_kwargs={"fontsize":30},hist_kwargs={"normed":True})
                               title_args={"fontsize": 35},title_fmt='.2e',smooth1d=None,smooth=1,label_kwargs={"fontsize":30},hist_kwargs={"density":True})

    if( sample_labels is None or len(sample_labels)<count+1 ):
        sample_labels=[ "Set "+str(i) for i in range(count+1)]
    for i in range(count+1):
        print("i=",i)
        figure.gca().annotate("----", color=colors[i], 
                          xy=(0.5, 1.0), xycoords="figure fraction",
                          xytext=(0, -35*fontscale*(i+0.5)), textcoords="offset points",
                          ha="left", va="top",fontsize=30*fontscale)
        figure.gca().annotate(sample_labels[i],
                          xy=(0.5, 1.0), xycoords="figure fraction",
                          xytext=(60*fontscale,  -35*fontscale*(i+0.5)), textcoords="offset points",
                          ha="left", va="top",fontsize=30*fontscale)
    #figure.gca().annotate(run+"  SNR="+str(snr)+modes+res+mmodal, xy=(0.5, 1.0), xycoords="figure fraction",
    #                      xytext=(0, -5), textcoords="offset points",
    #                      ha="center", va="top",fontsize=30*fontscale)
    if(pars is not None):
        for i in range(Npar):
            #figure.gca().annotate(names[i]+"= %.3e"%+pars[i], xy=(0.75, 0.9-i/20.0), xycoords="figure fraction",fontsize=30*fontscale)
            True

    figure.savefig(outpath)
    return


parser = argparse.ArgumentParser(description="Run standard analysis for runs started in Aug 2017");
parser.add_argument('chainfile',help="The name of the chain output file", nargs="*")
parser.add_argument('--fishfile',help="The name of the Fisher output file")
parser.add_argument('-l',help="Set the length of the effective portion of the chain at end default=-1,use last 1/4).",type=int,default=-1)
parser.add_argument('--sigma',help="Which of 1,2,3 sigma cuts to show.",default="1 2 3")
parser.add_argument('--lens',help="Set individual lengths of the effective portion of ends of the chains.",default="")
parser.add_argument('--acls',help="Individual autocorrelation lengths. May be used in defining plots. ).",default="")
parser.add_argument('--pars',help="Names of specific parameters to use.",default="")
parser.add_argument('--tag',help="Set a tag for the output.",default="")
parser.add_argument('--allfish',help="Use all Fisher runs together.",action="store_true")

    
args=parser.parse_args()

ncases=1

chainfiles=args.chainfile
fishfile=None
fishfiles=None
if  args.fishfile is not None: fishfile=args.fishfile


cred_levs=[0.68268949213709,#1-sigma
           0.95449973610364,#2-sigma
           0.99730020393674,#3-sigma
           ]
cred_levs=[cred_levs[int(i)-1] for i in args.sigma.split()]
print( "contours at ",cred_levs)

corner_range_cut=0.997
print ("chainfiles:",chainfiles)
print ("fishfile:",fishfile)

if(len(args.lens)>0):
    lens=[int(a) for a in args.lens.split()]
    print("lens=",lens)

if(len(args.acls)>0):
    acls=[float(a) for a in args.acls.split()]
    print("acls=",acls)

testing=False

pars=None


if(True):
    chainfile=chainfiles[0]
    print ("Processing posterior in ",chainfile)
    print (" with Fisher results in ",fishfile)
    parnames=get_par_names(chainfile)
    selectedParNames=args.pars.split()
    if(len(selectedParNames)==0):selectedParNames=parnames
    selectedPars=[parnames.index(name) for name in selectedParNames]
    print("sp:",selectedPars)
    parnames=[parnames[i] for i in selectedPars]
    Npar=len(parnames)
    print("parnames:",parnames)
    #run=re.search("Run\d*",chainfile).group(0)
    run=os.path.basename(chainfile)
    #run=os.path.basename(fishfile)
    if(None==re.search("lm22",chainfile)):modes=""
    else: modes=" (2-2 only)"
    if(None==re.search("nl8k",chainfile)):res=""
    else: res=" (higher-res sampling)"
    if(None==re.search("nmm",chainfile)):mmodal=""
    else: mmodal=" (no modal decomp sampling)"
    print ("run=",run)
    dirname=os.path.dirname(os.path.abspath(chainfile))
    basename=os.path.basename(dirname)
    outbasename=basename[:]
    if(len(args.tag)>0):
        ibase=basename.find("bambi")
        if(ibase<0):ibase=basename.find("ptmcmc")
        if(ibase<0):ibase=len(basename)
        outbasename=basename[:ibase]+args.tag
    if(testing):
        outpath=dirname+"/"+basename+"_TEST_corner.png"
        credfile=dirname+"/"+basename+"_TEST_credibility_levels.txt"
    else:
        outpath=dirname+"/"+outbasename+"_corner.png"
        outsamples=dirname+"/"+outbasename+"_samples.dat"
        credfile=dirname+"/"+basename+"_credibility_levels.txt"
    print ("reading posterior samples from file:",chainfile)
    print ("corner output to ",outpath)
    if(args.allfish):
        cov=[]
        for tag in ['a','b','c','d','k','l']:
            newfish=fishfile.replace("_c_fishcov","_"+tag+"_fishcov")
            if(os.path.isfile(newfish)):
                print ("reading covariance from ",newfish)
                cov.append(readCovar(newfish))
    elif(fishfile is not None):
        fishfiles=None
        import glob
        fishfiles=glob.glob(fishfile)
        fishfiles.sort()
        allpars=[]
        cov=[]
        for fishfile in fishfiles:
            print ("reading covariance from ",fishfile)
            pars,acov=readCovar(fishfile)
            pars=pars
            allpars+=[pars]
            cov+=[acov]
    else:
        allcov=None
        cov=None
    samples=[]
    i=0

    #make sample names from the most relevant part of the filenames
    longnames=[os.path.basename(chainfile) for chainfile in chainfiles]
    if fishfiles is not None: longnames +=[os.path.basename(path) for path in fishfiles]
    if(len(set(longnames))<len(longnames)):
        longnames=[os.path.basename(os.path.dirname(os.path.abspath(chainfile))) for chainfile in chainfiles]
    si=0;ei=1
    sgood=True;egood=True
    if(len(longnames)>1):
        for c in longnames[0]:
            #print(i)
            #print("sitest",[name[si] for name in longnames])
            #print("set:",set([name[si] for name in longnames]))
            if(sgood and len(set([name[si] for name in longnames]))==1):
                si+=1
            else:sgood=False
            #print("eitest",[name[-ei] for name in longnames])
            #print("set:",set([name[-ei] for name in longnames]))
            if(egood and len(set([name[-ei] for name in longnames]))==1):
                ei+=1
            else:egood=False
            if(not (sgood or egood)):break
    if(np.min([si+ei-len(name) for name in longnames])<=0):
        si=0
        ei=1
    if(ei<=1):
        sample_labels=[name[si:] for name in longnames]
    else:
        sample_labels=[name[si:-(ei-1)] for name in longnames]
    print("si/ei:",si,ei)
    print(sample_labels)
        
    #loop over chains for processing samples
    for chainfile in chainfiles:
        do_ess=False
        if("post_equal_weights.dat" in chainfile):
            code="bambi"
            burnfrac=0
            iskip=1
        elif("_resampled" in chainfile): #resampled ptmcmc data
            code="ptmcmc"
            burnfrac=0
            iskip=1
        elif("_t0.dat" in chainfile):
            code="ptmcmc"
            do_ess=True
        else:
            code="unk"
        lkeep=args.l
        if(len(args.lens)>0):lkeep=lens[i]
        nextch=ptmcmc_analysis.chainData(chainfile)
        if do_ess:
            upsample_fac=15
            ess,aclength=nextch.estimate_ess(3000)
            nsamp=int(ess*upsample_fac)
            length=ess*aclength
            print("nsamp,length:",nsamp,length)
            nextsamp=nextch.get_samples(nsamp,length)
        else:
            nextsamp=nextch.data
            print('samp shape=',nextsamp.shape, 'i0=', nextch.ipar0)
            nextsamp=nextsamp[:,nextch.ipar0:]
        print("nextsamp shape",nextsamp.shape)
        samples.append(nextsamp)
        i+=1
        print("calling cornerFisher: Npar=",Npar," pars=",pars," len(samples)",len(samples),samples[0].shape)
        
        cornerFisher(Npar,nextch.names[nextch.ipar0:],pars,samples,cov,cred_levs,selectedParNames,sample_labels=sample_labels)

    rs=[ math.sqrt(s[1]**2+s[2]**2) for s in samples[0]]
    rs.sort()
    nn=len(rs)
    rcuts=[rs[int(x*nn)] for x in cred_levs]
    print (dict(zip(cred_levs,rcuts)))
    
    print("Results in ",outpath)
