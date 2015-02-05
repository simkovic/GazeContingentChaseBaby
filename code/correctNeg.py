## The MIT License (MIT)
##
## Copyright (c) <2015> <Matus Simkovic>
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
## THE SOFTWARE.

from psychopy import visual, core, event,gui
from psychopy.misc import pix2deg
import numpy as np
import pylab as plt
import random, os
from Settings import Q
from Constants import *

# these routines were used to correct detection response where
# the subject clicked incorrect agent because a distractor overlapped
# or crossed with a target.

wind=Q.initDisplay()
# some constants
CHASEE=0
CHASER=1
X=0;Y=1;PHI=2

def exportFrame(trajectories,f,a1,a2,fn):
    nrframes=trajectories.shape[0]
    cond=trajectories.shape[1]
    clrs=np.ones((cond,3))
    clrs[CHASEE]=(0,1,0)
    clrs[CHASER]=(1,0,0)
    if a1>1:
        clrs[a1]=(0,0,1)
    else:
        clrs[a2]=(0,0,1)
    elem=visual.ElementArrayStim(wind,fieldShape='sqr',
            nElements=cond, sizes=Q.agentSize,
            elementMask='circle',elementTex=None,colors=clrs)
    elem.setXYs(trajectories[f,:,[X,Y]].transpose())
    elem.draw()
    wind.flip()
    wind.getMovieFrame()
    wind.saveMovieFrames(fn+'.png')


DATDIR='/home/matus/Desktop/pylink/behavioralOutput/'
INPDIR='/home/matus/Desktop/pylink/input'
SCRDIR='/home/matus/Desktop/pylink/screenshot/'
VP=0;TR=2;B=1;ID=3;RT=6;A1=7;A2=9;ACC=11
out=[]
for vp in vpn:
    dat = np.loadtxt(DATDIR+'vp%03d.res'%vp)
    i=0
    for d in dat:
        i+=1
        if (d[A1]==0 or d[A1]==1 or d[A2]==0 or d[A2]==1) and not d[ACC]:
            #print d[TR]
            s=INPDIR+'/vp%03d/vp%03db%dtrial%03d.npy'%(vp,vp,d[B],d[ID])
            trial=np.load(s)
            fr= round(d[RT]*100.0)
            s=SCRDIR+'vp%03db%dt%03df%04d'%(vp,d[B],d[TR],fr)
            exportFrame(trial,fr,d[A1],d[A2],s)
            out.append([vp,d[B],d[TR],fr])
    print np.sum(1-dat[dat[:,A1]>-1,ACC])
    #break
np.savetxt(SCRDIR+'coding.txt',np.array(out),fmt='%d')    
wind.close()
