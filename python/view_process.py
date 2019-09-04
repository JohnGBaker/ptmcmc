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
#from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.widgets import Slider, Button, RadioButtons
nparmax=12

##########
# Arguments and argument injest
# determine basename from first argument.

parser = argparse.ArgumentParser(description='Provide snapshot of chain state.')
parser.add_argument('fname', metavar='chain_file', nargs='+', type=str, 
                    help='chain file path')
parser.add_argument('-uselike',action='store_true',help='include the likelihood')
args = parser.parse_args()
print(args)


#####################################
#general functions

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

#####################################
#ptmcmc-specific functions:

#process given filename to get basename and chainfile name
def get_basename(fname):
    if(fname.endswith(".out")):
        #we assume ".out does not otherwise appear in the name... that
        #could check... if(fname.count(".out")>1):...
        basename=fname.replace(".out","")
        fname=fname.replace(".out","_t0.dat")
    elif(fname.endswith("_t0.dat")):
        basename=fname.replace("_t0.dat","")
    else: print("File name '"+fname+"' not recognized.")
    print ("basename="+basename)
    return basename,fname

def get_par_names(fname):
    with open(fname) as f:
        line=f.readline()
        line=f.readline()
        names=line.split()
        names=names[5:]
    return names

def read_data(names):
    global allparnames,Nmax,Smax
    datanames=[]
    Nmax=0
    Smax=0
    allparnames=[]
    for fname in names:
        basename,chainname=get_basename(fname)
        #read in data
        data=np.loadtxt(chainname,converters={4: lambda s:-1})
        N=len(data)
        dSdN=data[4,0]-data[3,0]
        S=N*dSdN
        if(N>Nmax):Nmax=N
        if(S>Smax):Smax=S
        #print ("Data have ",N," rows representing ",N*dSdN," steps.")
        #print ("data[1]=",data[1])
        maxPost=max(data[:,1])
        parnames=["samp","post",]
        if args.uselike:
            data=np.delete(data,[3,4,len(data[0])-1],1)
            parnames+=['like']
        else:
            data=np.delete(data,[2,3,4,len(data[0])-1],1)
        parnames+=get_par_names(chainname)
        print (parnames)
        for name in parnames:
            if(not name in allparnames):
                allparnames.append(name)
        datanames.append([data,parnames])
    return datanames

def get_xydata(data,i,j,dens,samps):
    d=data[data[:,0]>samps]
    #d=d[d[:,6]>4]
    Nd=len(d)
    #print("Reduced data len =",Nd)
    every=int(1+Nd/dens)
    #print(Nd,dens,every)
    x=d[::every,i]
    y=d[::every,j]
    #print (len(x))
    return x,y
    
##################
#Widget functions
##################
import matplotlib.patches as mpatches

def update(val):
    #c0=1+allparnames[1:].index(radio.value_selected)
    #print("index->",c0)
    start = int(10**sstart.val)
    samps = int(10**sdens.val)
    xmin=1e100;xmax=-1e100
    ymin=1e100;ymax=-1e100
    xy=np.array([])
    cc=np.array([])
    ic0=0.5;
    ind=0
    plotlabels=[]
    for data,parnames in alldata:
        if(radio.value_selected in parnames):
            c0=1+parnames[1:].index(radio.value_selected)
            x,y=get_xydata(data,0,c0,samps,start)
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
            plotlabels.append(mpatches.Patch(color=cmap(colorval/cmap_norm), label=labels[ind],hatch='.'))
        ind=ind+1            
    
    scat.set_offsets(xy)
    scat.set_array(cc)
    ax.set_xlim(xmin-0.1*(xmax-xmin),xmax+0.1*(xmax-xmin))
    ax.set_ylim(ymin-0.1*(ymax-ymin),ymax+0.1*(ymax-ymin))
    print("plot labels=",plotlabels)
    ax.legend(handles=plotlabels)
    fig.canvas.draw_idle()
    
def reset(event):
    alldata=read_data(args.fname)
    sstart.reset()
    sdens.reset()

###################
#main code
###################

alldata = read_data(args.fname)
[data,parnames]=alldata[0]
labels=make_short_labels(args.fname)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
c0 = 1
s0 = 3
d0 = 2

x,y=get_xydata(data,0,c0,10**d0,10**s0)
cmap = matplotlib.cm.get_cmap('tab10')
cmap_norm=10
scat = plt.scatter(x, y, s=1, c=x, cmap="tab10",norm=colors.Normalize(0,cmap_norm))
#scat = plt.scatter([], [], s=1,cmap="tab10")

axcolor = 'lightgoldenrodyellow'
#axstart = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
#axdens = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
axstart = plt.axes([0.25, 0.1, 0.65, 0.03])
axdens = plt.axes([0.25, 0.15, 0.65, 0.03])
height=(len(allparnames)-1)*0.05
#rax = plt.axes([0.025, 0.5-height/2, 0.1, height], facecolor=axcolor)
rax = plt.axes([0.025, 0.5-height/2, 0.1, height])
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])

ilogS=(math.log10(Smax))
ilogSamps=int(math.log10(Nmax))
print("ilogS=",ilogS)
sstart = Slider(axstart, 'log-Start', 2, ilogS, valinit=s0)
sdens = Slider(axdens, 'log-Density', 1, ilogSamps, valinit=d0)
radio = RadioButtons(rax, allparnames[1:], active=0)
for circle in radio.circles: # adjust radius here. The default is 0.05
    circle.set_radius(0.03)
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

sstart.on_changed(update)
sdens.on_changed(update)
radio.on_clicked(update)
button.on_clicked(reset)

update(0)

plt.show()
