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

def get_par_names(fname):
    with open(fname) as f:
        line=f.readline()
        line=f.readline()
        names=line.split()
        names=names[5:]
    return names

def read_samples(Npar,sample_file,code="bambi",burnfrac=0,keeplen=-1,iskip=1,maxsamps=30000):
    if(code=="bambi"):
        print ("Reading BAMBI samples")
        data=np.loadtxt(sample_file,usecols=range(Npar))
    else:
        print ("Reading PTMCMC samples")
        #data=np.loadtxt(chainfile,usecols=range(5,5+Npar))
        #data=np.loadtxt(sample_file,usecols=[0]+list(range(5,5+Npar)))
        data=np.loadtxt(sample_file,usecols=[0]+[5+x for x in selectedPars])
        print("data[-1]=",data[-1],"data[-2]=",data[-2][0])
        every=data[-1][0]-data[-2][0]
        data=data[:,1:]
        
        #hack for sign of q
        if "logq" in parnames:
            iq=parnames.index("logq")
            for d in data:
                if(d[iq]<0):# if logq<0 flip sign and add pi to phi0
                    d[iq]*=-1
                    if "phi0" in parnames:
                        ip=parnames.index("phi0")
                        d[ip]+=math.pi
                        if(d[ip]>2*math.pi):d[ip]-=2*math.pi
                        
        if(keeplen/iskip>maxsamps):iskip=int(keeplen/maxsamps)
        iev=int(iskip/every)
        if(iev>1):
            data=data[::iev]
            every*=iev
        print ("every=",every," iev=",iev," iskip=",iskip)            
        #print "shape=",data.shape
        keeplen=int(keeplen/every)

    print ("code=",code,"  keeplen=",keeplen,"  burnfrac=",burnfrac,"  len=",len(data) )

    if(keeplen>0):
        data=data[len(data)-keeplen:]
    else:
        data=data[int(len(data)*burnfrac):]

    if(code!="bambi"):
        outfile=re.sub("_t0.dat","_post_samples.dat",sample_file)
        print("Writing PTMCMC samples to '",outfile,"'")
        np.savetxt(outfile,data)
        
    return data

def cornerFisher(Npar,pars,samples,in_cov,cred_levs,iparmin=0,iparend=9,sample_labels=None):
    data=[]
    for datum in samples:
    #Sylvain:Here crop down to a limited parameter set for a smaller plot
    #the overlaid text is added with the "annotate" commands below
        #print ("data shape=",datum.shape)
        istart=iparmin;
        iend=iparend
        data.append(datum[:,istart:iend])
    if(pars is not None):
        pars=pars[istart:iend]
    names=parnames[istart:iend]
    if(isinstance(in_cov,list)):
        cov=[]
        for ic in in_cov:
            cov.append(ic[istart:iend,istart:iend])
    elif in_cov is not None:
        cov=in_cov[istart:iend,istart:iend]
    else:
        cov=None
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

    figure = corner.corner(data[0], bins=50,labels=names,levels=levels,cov=cov,
                             truths=pars,quantiles=[0.159, 0.5, 0.841],show_titles=True,range=rangelimits,use_math_text=True,
                             title_args={"fontsize": 35},title_fmt='.2e',smooth1d=None,smooth=1,label_kwargs={"fontsize":30},hist_kwargs={"normed":True})
    count=0
    print ("nsamp[0] = ",len(data[0]))
    colors=["k","r","b","g","m","c","y"]
    for datum in data[1:]:
        count+=1
        print ("nsamp[i] ",len(datum))
        figure = corner.corner(datum, bins=50,labels=names,fig=figure,color=colors[count],levels=levels,cov=cov, plot_density=False,
                               truths=pars,quantiles=[0.159, 0.5, 0.841],show_titles=True,range=rangelimits,use_math_text=True,
                               title_args={"fontsize": 35},title_fmt='.2e',smooth1d=None,smooth=1,label_kwargs={"fontsize":30},hist_kwargs={"normed":True})
    
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
if args.fishfile is not None: fishfile=args.fishfile


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
        print ("reading covariance from ",fishfile)
        cov=readCovar(fishfile)
    else:
        cov=None
    samples=[]
    i=0

    #make sample names from the most relevant part of the filenames
    longnames=[os.path.basename(chainfile) for chainfile in chainfiles]
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
        if("post_equal_weights.dat" in chainfile):
            code="bambi"
            burnfrac=0
            iskip=1
        elif("_t0.dat" in chainfile):
            code="ptmcmc"
            burnfrac=0.75
            iskip=50
            print("ACLS=",args.acls)
            if(len(args.acls)>0):iskip=int(50+acls[i]/100.0)
        elif("fishcov.dat" in fishfile):
            code="ptmcmc"
            burnfrac=0
            iskip
        lkeep=args.l
        if(len(args.lens)>0):lkeep=lens[i]
        samples.append(read_samples(Npar,chainfile,code=code,burnfrac=burnfrac,keeplen=lkeep,iskip=iskip))
        i+=1
        cornerFisher(Npar,pars,samples,cov,cred_levs,sample_labels=sample_labels)

    rs=[ math.sqrt(s[1]**2+s[2]**2) for s in samples[0]]
    rs.sort()
    nn=len(rs)
    rcuts=[rs[int(x*nn)] for x in cred_levs]
    print (dict(zip(cred_levs,rcuts)))
    
    print("Results in ",outpath)
