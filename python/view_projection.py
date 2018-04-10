import sys
import os
import numpy as np
import math
import subprocess
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
#from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.widgets import Slider, Button, RadioButtons
nparmax=12

##########
# Arguments and argument injest
# determine basename from first argument.

parser = argparse.ArgumentParser(description='Provide snapshot of chain state.')
parser.add_argument('fname', metavar='chain_file', type=str, 
                    help='chain file path')

args = parser.parse_args()
print(args)


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
    print ("basename="+basename)
    return basename,fname

def get_par_names(fname):
    with open(fname) as f:
        line=f.readline()
        line=f.readline()
        names=line.split()
        names=names[5:]
    return names

def get_xydata(data,i,j,dens,samps):
    d=data[data[:,0]>samps]
    Nd=len(d)
    #print("Reduced data len =",Nd)
    every=int(Nd/dens)
    print(Nd,dens,every)
    x=d[::every,i]
    y=d[::every,j]
    print (len(x))
    return x,y
    
##################
#Widget functions
##################

def update(val):
    cx=2+parnames.index(radioX.value_selected)
    cy=2+parnames.index(radioY.value_selected)
    print("index->",cx,cy)
    start = int(10**sstart.val)
    samps = int(10**sdens.val)
    x,y=get_xydata(data,cx,cy,samps,start)
    xx = np.vstack ((x, y))
    scat.set_offsets (xx.T)
    xmin=x.min(); xmax=x.max()
    ymin=y.min(); ymax=y.max()
    ax.set_xlim(xmin-0.1*(xmax-xmin),xmax+0.1*(xmax-xmin))
    ax.set_ylim(ymin-0.1*(ymax-ymin),ymax+0.1*(ymax-ymin))
    fig.canvas.draw_idle()

def reset(event):
    sstart.reset()
    sdens.reset()

###################
#main code
###################

basename,chainname=get_basename(args.fname)

#read in data
data=np.loadtxt(chainname,converters={4: lambda s:-1})
N=len(data)
dSdN=data[4,0]-data[3,0]
print ("Data have ",N," rows representing ",N*dSdN," steps.")
print ("data[1]=",data[1])
maxPost=max(data[:,1])
data=np.delete(data,[2,3,4,len(data[0])-1],1)
print ("data[1]=",data[1])
parnames=get_par_names(chainname)
print (parnames)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.35, bottom=0.25)
cx0= 0
cy0= 1
s0 = 3
d0 = 2

x,y=get_xydata(data,2+cx0,2+cy0,10**d0,10**s0)
scat = plt.scatter(x, y, s=1)

axcolor = 'lightgoldenrodyellow'
axstart = plt.axes([0.35, 0.1, 0.55, 0.03], facecolor=axcolor)
axdens = plt.axes([0.35, 0.15, 0.55, 0.03], facecolor=axcolor)
height=(len(parnames)-1)*0.05
rXax = plt.axes([0.025, 0.5-height/2, 0.1, height], facecolor=axcolor)
rXax.text(0.9, 0.95, "X", transform=rXax.transAxes, fontsize=12,
          verticalalignment='top',horizontalalignment='right')
rYax = plt.axes([0.15, 0.5-height/2, 0.1, height], facecolor=axcolor)
rYax.text(0.9, 0.95, "Y", transform=rYax.transAxes, fontsize=12,
          verticalalignment='top',horizontalalignment='right')
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])

ilogS=(math.log10(N*dSdN))
ilogSamps=int(math.log10(N))
print("ilogS=",ilogS)
sstart = Slider(axstart, 'log-Start', 2, ilogS, valinit=s0)
sdens = Slider(axdens, 'log-Density', 1, ilogSamps, valinit=d0)
radioX = RadioButtons(rXax, parnames, active=cx0)
radioY = RadioButtons(rYax, parnames, active=cy0)
for circle in radioX.circles: # adjust radius here. The default is 0.05
    circle.set_radius(0.03)
for circle in radioY.circles: # adjust radius here. The default is 0.05
    circle.set_radius(0.03)
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

sstart.on_changed(update)
sdens.on_changed(update)
radioX.on_clicked(update)
radioY.on_clicked(update)
button.on_clicked(reset)

plt.show()
