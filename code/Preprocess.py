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

import numpy as np
import pylab as plt
from Settings import *
import os,pickle
from ETData import ETData, interpRange
plt.ion()

##########################################################
# helper functions
def _isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def _discardInvalidTrials(data):
    bb=range(len(data))
    bb.reverse()
    for i in bb:
        if data[i].ts<0: data.pop(i)
    return data
def _reformat(trial,tstart,Qexp):
    if len(trial)==0: return np.zeros((0,7))
    trial=np.array(trial)
    ms=np.array(Qexp.monitor.getSizePix())/2.0
    if type(trial) is type( () ): print 'error in readEdf'
    trial[:,0]-=tstart
    trial[:,1]=Qexp.pix2deg(trial[:,1]-ms[0])
    trial[:,2]=-Qexp.pix2deg(trial[:,2]-ms[1])
    if trial.shape[1]>4:
        trial[:,4]=Qexp.pix2deg(trial[:,4]-ms[0])
        trial[:,5]=-Qexp.pix2deg(trial[:,5]-ms[1])
    return trial
  
def readTobii(vp,block,path,lagged=False,verbose=False):
    '''
        reads Tobii controller output of subject VP and block BLOCK
        the file should be at <PATH>/VP<VP>B<BLOCK>.csv
        requires the corresponding input files on the input path
        vp - subject id
        block - experiment block
        path - path to the eyetracking data
        lagged - log time stamp when the data was made available
            (ca. 30 ms time lag), useful for replay
        verbose - print info
        returns ETData instance
        
        Each trial starts with line '[time]\tTrial\t[nr]'
        and ends with line '[time]\tOmission'  
    '''
    from Settings import Qexp
    if verbose: print 'Reading Tobii Data'
    #path = os.getcwd()
    #path = path.rstrip('code')
    f=open(path+'VP%03dB%d.csv'%(vp,block),'r')
    #Qexp=Settings.load(Q.inputPath+'vp%03d'%vp+Q.delim+'SettingsExp.pkl' )
           
    #f=open('tobiiOutput/VP%03dB%d.csv'%(vp,block),'r')
    ms= Qexp.monitor.getSizePix()
    try:
        data=[];trial=[]; theta=[];t=0;msgs=[]; t0=[0,0,0];reward=[]
        on=False
        while True:
            words=f.readline()
            if len(words)==0: break
            words=words.strip('\n').strip('\r').split('\t')
            if len(words)==2: # collect header information
                if words[0]=='Recording time:':
                    recTime=words[1]; t0[0]=0; on=True
                if words[0]=='Subject: ':on=False
                if words[0]=='Recording refresh rate: ':
                    hz=float(words[1])
            elif len(words)==4 and words[2]=='Trial':
                t0[1]=trial[-1][0] # perturb
            elif len(words)==4 and words[2]=='Phase':
                phase=int(words[3])
            elif len(words)>=11 and on: # record data
                # we check whether the data gaze position is on the screen
                xleft=float(words[2]); yleft=float(words[3])
                if xleft>ms[0] or xleft<0 or yleft>ms[1] or yleft<0:
                    xleft=np.nan; yleft=np.nan;
                xright=float(words[5]); yright=float(words[6])
                if xright>ms[0] or xright<0 or yright<0 or yright>ms[1]:
                    xright=np.nan; yright=np.nan;
                if lagged: tm =float(words[0])+float(words[8]);ff=int(words[1])
                else: tm=float(words[0]);ff=int(words[1])-2
                tdata=(tm,xleft,yleft,float(words[9]),
                    xright,yright,float(words[10]),ff)
                trial.append(tdata)
            elif len(words)>2 and (words[2]=='Detection' or words[2]=='Omission'):
                # we have all data for this trial, transform to deg and append
                on=False;t0[2]=trial[-1][0]
                trial=np.array(trial)
                trial[trial==-1]=np.nan # TODO consider validity instead of coordinates
                trial=_reformat(trial,t0[0],Qexp)
                #print t0, trial.shape, trial[0,0]
                et=ETData(trial[:,:-1],[],t0,
                    [vp,block,t,hz,'BOTH'],fs=np.array([np.nan,np.nan]),recTime=recTime,msgs=msgs)
                fs=trial[et.ts:et.te,[-1,0]]
                fs[:,1]-=t0[1]
                for fff in range(fs.shape[0]-2,-1,-1):
                    if fs[fff+1,0]<fs[fff,0]:
                        fs[fff,0]=fs[fff+1,0]
                et.fs=np.zeros((fs[-1,0],3))
                et.fs[:,0]=range(int(fs[-1,0]))
                et.fs[:,1]=interpRange(fs[:,0],fs[:,1],et.fs[:,0])
                et.fs[:,2]=interpRange(fs[:,1],range(fs.shape[0]),et.fs[:,1])
                et.fs[:,2]=np.round(et.fs[:,2])
                for msg in et.msgs:
                    if msg[2]=='Omission': msg[0]=float(msg[0]);msg[1]=float(msg[1])
                    if msg[2]=='Drift Correction':
                        msg[1]=int(round(msg[0]*75/1000.))
                        #msg.append(msg[0]*et.hz/1000.)
                    elif msg[1]-et.fs.shape[0]>0:
                        val=(msg[1]-et.fs.shape[0])*75/et.hz+et.fs[-1,2]
                        #msg.append(int(round(val)))
                    #else: msg.append(int(et.fs[int(msg[1]),2]))
                #et.extractBasicEvents(trial[:,:-1]);
                et.phase=phase;et.reward=reward
                data.append(et)
                t+=1;trial=[];msgs=[];reward=[]
            elif on and len(words)==6:
                msgs.append([float(words[0])-t0[1],words[2]+' '+words[5]]) 
            elif on and len(words)>2:
                msgs.append([float(words[0])-t0[1],int(words[1]),words[2]])
                if words[2]=='Reward On': reward=[float(words[0])-t0[1]]
                if words[2]=='Reward Off': reward.append(float(words[0])-t0[1])
    except: f.close(); print 'Words: ',words; raise
    f.close()
    if verbose:print 'Finished Reading Data'
    return data
    
    
if __name__ == '__main__':
##    for vp in range(1,5):
##        saveETinfo(vp=vp)
##        saveTrackedAgs(vp=vp)
    vpn=range(170,185)
    D=[]
    for vp in vpn:
        data=readTobii(vp,0)
        for trl in data:
            trl.extractBasicEvents()
            for b in trl.bev:
                g=trl.getGaze()
                if b[1]+1<g.shape[0] and b[0]-1>0:
                    dist=np.linalg.norm(g[b[1]+1,[7,8]]-g[b[0]-1,[7,8]])
                    D.append([b[1]-b[0],dist])
    D=np.array(D) 
            
    #print np.int32(np.isnan(data[0].gaze[:1000,1]))
    #data[0].extractBasicEvents()
    #for dat in data: dat.extractBasicEvents()
