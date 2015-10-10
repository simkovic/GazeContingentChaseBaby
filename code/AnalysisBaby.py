# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import pylab as plt
import matplotlib as mpl
import os,pystan
from ETData import tseries2eventlist
from Preprocess import readTobii
from matustools.matusplotlib import *
LBLS=[['Blink','Gaze','E3train','E3test','E4train','E4test'],
	['Infant','Adult','Random'],['Observed','Shuffled']]
BEV,AEV,CEV,FEV=range(4)
ETHZ=60
DPI=300

def initPath(adult=False,shuffled=False):
    outpath=os.getcwd().rstrip('code')+'evaluation'+os.path.sep+'Baby'+os.path.sep
    etdatapath=os.getcwd().rstrip('code')+'tobiiOutput'+os.path.sep  
    figpath=os.getcwd().rstrip('code')+'figures'+os.path.sep+'Baby'+os.path.sep
    if adult:
        etdatapath+= 'adultcontrol'+os.path.sep
        figpath+= 'ac'+os.path.sep
    if adult: allvpn=[range(100,110),range(120,130),range(172,180)]
    else: allvpn=[range(100,120),range(120,140),range(143,164),range(170,185)]
    if shuffled:
        for vpn in allvpn:
            for vp in vpn:
                vp+=100
    suffix=['B','A'][int(adult)]+['O','R'][int(shuffled)]
    pth={'outpath':outpath,'etdatapath':etdatapath,
         'figpath':figpath,'allvpn':allvpn,'suf':suffix}
    return pth
def loadSubjectMetadata(path=None):
    ''' return Nx3 matrix with one row for each subject
        first column - subject id
        second column - male (1) or female (0)
        third column - age in days
    '''
    from datetime import date
    if path is None: path=initPath()
    f=open(path['etdatapath']+'vpinfo.csv','r')
    D=[]
    for line in f.readlines():
        els=line[:-1].rsplit(' ')
        els=filter(len,els)
        d,m,y=map(int,els[1].rsplit('.'))
        expdate=date(2000+y,m,d)
        d,m,y=map(int,els[3].rsplit('.'))
        birthdate=date(2000+y,m,d)
        age=expdate-birthdate
        vp=int(els[0])
        if vp<120: cond=0
        elif vp<140: cond=1
        elif vp<170: cond=2
        elif vp<200: cond=3
        D.append([vp,int(els[2]),age.days,cond])
    f.close()
    D=np.array(D)
    print 'Age range is [%d,%d] days'%(np.min(D[:,2]),np.max(D[:,2]))
    print '%d boys, %d girls'%(D[:,1].sum(),(D[:,1]==0).sum())
    print '%.1f percent boys'% (D[:,1].mean()*100)
    print '%d subjects in total'% D.shape[0]
    return D
def plotBasicEvents(adult=False):
    path=initPath(adult,0)
    k=0
    for vpn in path['allvpn']:
        plt.close()
        plt.figure(0,figsize=(100,10))
        for vp in vpn:
            print vp
            data=readTobii(vp,0,path['etdatapath'])
            st=0
            for trl in data:
                trl.extractBasicEvents()
                #trl.opur=trl.cpur=np.zeros(trl.te-trl.ts)
                #trl.events=[]
                #trl.extractComplexEvents()
                trl.computeMsgTime()
                st=trl.plotMsgs(st)
        plt.ylim([vpn[0],vpn[-1]+1])
        plt.savefig(path['figpath']+'datOverview%d.png'%k);k+=1

def removeShortEvs(tsin,md):
    """ >>> ts=np.array([1,1,1,0,1,1,1,0,0,1,1,1,0,0,0,1,1,1,
                0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,1,1,0,0,1,0,1,0,1])
        >>> print ts
        >>> print removeShortEvs(ts==1,2,3) 
    """
    evs=[]
    if not np.any(tsin): return np.int32(tsin)
    if np.all(tsin): return np.int32(tsin)
    tser=np.copy(tsin)
    ton = np.bitwise_and(tser,
        np.bitwise_not(np.roll(tser,1))).nonzero()[0].tolist()
    toff=np.bitwise_and(np.roll(tser,1),
        np.bitwise_not(tser)).nonzero()[0].tolist()
    if ton[-1]>toff[-1]:toff.append(tser.shape[0])
    if ton[0]>toff[0]:ton.insert(0,0)
    assert len(ton)==len(toff)
    #print np.int32(np.bitwise_and(tser,np.bitwise_not(np.roll(tser,1))))
    #print np.int32(np.bitwise_and(np.roll(tser,1),np.bitwise_not(tser)))
    for f in range(len(ton)):
        ts=ton[f];te=toff[f];dur=te-ts
        #print ts, te,dur
        if  dur<md: tsin[ts:te]-=1
    #tsin -= temp[:,val]
    return np.int32(tsin)

def plotNogazeDuration():
    plt.figure(figsize=(12,12))
    for vp in range(100,120):
        print vp
        plt.subplot(5,4,vp-99)
        plt.ion()
        data=readTobii(vp,0,ETDATAPATH);
        datT=[];datF=[]
        for trl in data:
            trl.extractBasicEvents()
            miss=np.int32(np.logical_and(np.isnan(trl.gaze[:,7]),
                    np.isnan(trl.gaze[:,8])))
            miss=removeShortEvs(miss,2*60)
            miss=1-removeShortEvs(1-miss,1*60)
            datT+=map(lambda x: (x[1]-x[0])/60.,tseries2eventlist(miss))
            datF+=map(lambda x: (x[1]-x[0])/60.,tseries2eventlist(1-miss))
        
        x=np.linspace(0,10,21);h=x[-1]/float(x.size-1)
        a=np.histogram(datT,bins=x, normed=True)
        plt.barh(x[:-1],-a[0],ec='k',fc='k',height=h,lw=0)
        a=np.histogram(datF,bins=x, normed=True)
        plt.barh(x[:-1],a[0],ec='g',fc='g',height=h,lw=0)
        plt.xlim([-0.7,0.7]);
        plt.gca().set_yticks(range(0,10,2))
        plt.ylim([0,10]);
        #plt.grid(False,axis='y')
        if vp==10:plt.legend(['blikn','gaze'])
        
def computeBabyEvents(vp=113,path=None,verbose=False):
    ''' returns baby events a nested lists with subjects
        at the first level 

    '''
    if path is None: path=initPath()
    data=readTobii(vp,0,path['etdatapath'])
    flickerEv=[];trial=-1;allEvs=[]
    TOL=20 # max allowed time gap in microsec between
    # the end and start of the consecutive events
    for trl in data:
        trial+=1
        chaseEv=[];flickerEv.append([])
        saccount=-1
        t0=trl.t0[0]
        maxf=trl.t0[1]-t0#trl.te-trl.ts
        #print trl.phase
        assert trl.phase<3
        trl.extractBasicEvents()
        trl.computeMsgTime() 
        for msg in trl.msgs:
            temp=msg[2].split('th ')
            # E3 chase
            if len(temp)==2:
                si=int(temp[0])
                if si==1:
                    chaseEv.append([msg[3],msg[4]])
                    saccount=1
                elif si-saccount==1:
                    saccount+=1
                    chaseEv[-1][1]=min(msg[4],maxf)
                else: # should never happen
                    print si, saccount
                    raise
            # E4 flicker
            if msg[2]=='Reward On':
                flickerEv[-1].extend([msg[0]])
            elif msg[2]=='Reward Off':
                assert len(flickerEv[-1])==1
                if saccount == [3,5,5][trl.phase]:flickerEv[-1]=[]
                else: flickerEv[-1].append(msg[0])
        if len(flickerEv[-1])==1:
            flickerEv[-1].append(maxf)
        # E1 no gaze
        miss=np.int32(np.logical_and(np.isnan(trl.gaze[:,1]),
            np.isnan(trl.gaze[:,4])))
        miss=np.concatenate([miss,np.ones(ETHZ*2,dtype=np.int32)*miss[-1]])
        miss=removeShortEvs(miss,2*ETHZ)
        miss=1-removeShortEvs(1-miss,0.5*ETHZ)
        nogazeEv=tseries2eventlist(miss[trl.ts:trl.te])
        # translate to time
        ngeTime=[]
        for ev in nogazeEv:
            if ev[0]==ev[1]: continue
            ngeTime.append([trl.gaze[ev[0]+trl.ts,0]-t0,
                trl.gaze[ev[1]+trl.ts,0]-t0])
        nogazeEv=ngeTime
        # print info
        if verbose: print chaseEv
        if verbose: print nogazeEv
        if verbose: print flickerEv
        # combine event streams
        evs=[]
        #if len(nogazeEv) and nogazeEv[0][0]>0:
        #    evs.append([AEV,0,min(nogazeEv[0][0],chaseEv[0][0])-1])
        while True:
            if not len(nogazeEv) and not len(chaseEv): break
            if (not len(nogazeEv) or len(chaseEv) and
                chaseEv[0][0]<nogazeEv[0][0]):
                ev=chaseEv.pop(0);evid=CEV; 
            elif not len(chaseEv) or chaseEv[0][0]>=nogazeEv[0][0]:
                ev=nogazeEv.pop(0);evid=BEV;
            else:
                print chaseEv[0],nogazeEv[0]
                raise
            if not len(evs):
                if ev[0]>0: evs.append([AEV,0,ev[0],False])
            elif ev[0]-evs[-1][2]>TOL:
                evs.append([AEV,evs[-1][2],ev[0],False])
            elif ev[0]-evs[-1][2]<=TOL:
                if evid==BEV and evs[-1][0]==CEV:
                    evs[-1][2]=ev[0]
                elif evid==CEV  and evs[-1][0]==BEV:
                    print 'warning',vp,trial,evs[-1],ev
                    continue
                elif evid==CEV and evs[-1][0]==CEV:
                    # happens rarely
                    evs[-1][2]=ev[1]
                    continue
            try: assert ev[1]-ev[0]>0
            except:
                print trial,evid,ev
                raise ValueError
            evs.append([evid,ev[0],ev[1],False])
        if not len(evs): evs.append([AEV,0,maxf,True])
        elif maxf-evs[-1][2]>TOL:
            evs.append([AEV,evs[-1][2],maxf,True])
        # add E4 flicker events
        for ei in range(len(evs)):
            if (len(flickerEv[-1]) and evs[ei][1]<flickerEv[-1][0]
                and evs[ei][2]>flickerEv[-1][0] and evs[ei][0]==CEV):
                flickerEv[-1][1]=evs[ei][2]
                evs[ei][2]=flickerEv[-1][0];evs[ei][3]=True
                evs.insert(ei+1,[FEV]+flickerEv[-1]+[False])
                #min(flickerEv[-1][1]
        evs[-1][3]=True
        if verbose: print evs
        # check event stream
        for ei in range(len(evs)):
            evs[ei].append(trl.phase<2)
            assert len(evs[ei])==5
            assert evs[ei][1]<=evs[ei][2]
            if (evs[ei][0]==CEV or evs[ei][0]==FEV): assert  evs[ei][4]>-1
            if ei:
                assert evs[ei][0]!=evs[ei-1][0]
                assert evs[ei][1]-evs[ei-1][2]==0
                if evs[ei][0]==FEV:
                    assert evs[ei-1][0]==CEV 
                    if ei+1<len(evs): assert evs[ei+1]!=CEV
        allEvs.append(evs)
    return allEvs
             
def plotBabyEvents(vp,path,evs=None):
    if evs==None: evs=computeBabyEvents(vp,path=path)
    plt.close('all')
    ax=plt.gca();dur=0
    for trial in range(len(evs)):
        for ev in evs[trial]:
            s=ev[1];e=ev[2]
            r=mpl.patches.Rectangle((s,trial+0.25),e-s,
                0.5,color=['k','y','g','r'][ev[0]])
            ax.add_patch(r)
        dur=max(dur,ev[2])
    plt.xlim([0,dur])
    plt.ylim([0,trial+1])
    plt.savefig(path['figpath']+'vp%d.png'%vp)
    

def computeAvgDS(evs):
    M=np.zeros((4,4));count=np.zeros((4,4))
    for ev in evs:
        for ei in range(1,len(ev)):
            #count[ev[ei][0],ev[ei][0]]+=1
            
            if not ev[ei][3]:
                M[ev[ei][0],ev[ei][0]]+= (ev[ei][2]-ev[ei][1])/1000.
                count[ev[ei][0],ev[ei][0]]+=1
            count[:,ev[ei-1][0]]+=1
            count[ev[ei-1][0],ev[ei-1][0]]-=1
            M[ev[ei][0],ev[ei-1][0]]+= 1
    M/=count
    return M

def plotVpDS(vp,bns=np.linspace(0,14,15)):
    plt.close('all')
    evs=computeBabyEvents(vp)
    R=[[],[],[],[]];S=[[],[],[],[]]
    for ev in evs:
        for ei in range(1,len(ev)):
            if ev[ei][3]: S[ev[ei][0]].append((ev[ei][2]-ev[ei][1]))
            else: R[ev[ei][0]].append((ev[ei][2]-ev[ei][1]))
    for i in range(4):
        plt.subplot(2,4,i+1)
        plt.grid(False);plt.ylim([0,1])
        if not i: plt.ylabel('valid')
        plt.title(['nogaze','gaze','chase','flicker'][i])
        if len(R[i]): plt.hist(R[i],normed=True,bins=bns)
        
        plt.subplot(2,4,i+5)
        plt.grid(False);plt.ylim([0,1])
        if not i: plt.ylabel('to impute')
        if len(S[i]): plt.hist(S[i],normed=True,bins=bns)
    plt.savefig(FIGPATH+'dsvp%d.png'%vp)

def plotAvgDS():
    i=-1
    for vpn in [range(100,120),range(120,140),range(143,163),range(170,185)]:
        D=[];i+=1
        for vp in vpn:
            evs=computeBabyEvents(vp,verbose=False)
            #plotBabyEvents(evs)
            #plt.savefig(FIGPATH+'vp%d.png'%vp)
            D.append(computeAvgDS(evs))
        D=np.array(D)
        np.save('D%d.npy'%i,D)
        D=np.load('D%d.npy'%i)
        for eid in range(4):
            plt.subplot(4,4,4*i+eid+1)
            plt.grid(False)
            if not i:plt.title(['nogaze','gaze','chase','flicker'][eid])
            #print i,eid, np.min(D[~np.isnan(D[:,eid,eid]),eid,eid])
            if np.all(np.isnan(D[:,eid,eid])): continue
            plt.hist(D[~np.isnan(D[:,eid,eid]),eid,eid],
                     normed=True,bins=np.linspace(0,14,15))
            plt.ylim([0,1])
    plt.savefig(FIGPATH+'dsavg.png')

def evs2stan(PLOT=False):
    ''' 2nd dim of dur is blink-train,blink-test, watch-train,watch-test,
            chase-train,chase-test,flicker-train,flicker-test
        2nd dim of arivt is CEV->CEV, CEVtr-> FEV,CEVte->FEV,
            FEVtr-> FEV,FEVte->FEV,'''
    def _helpFun(DD,JJ,CC,PP,PLOT):
        temp=computeBabyEvents(CC,path=PP)
        if PLOT: plotBabyEvents(CC,PP,evs=temp)
        at=np.array([np.nan,np.nan]);trid=[-1,-1]
        for j in range(len(temp)):
            for k in range(len(temp[j])):
                a,b,c,d,e=temp[j][k];e=1-e
                #print np.int32(temp[j][k])
                dur=(c-b)/1000.
                if dur ==0: 
                    print CC[i,0],j,k
                    bla
                DD[a*2+e,d,int(DD[a*2+e,d,-1])]=dur
                DD[a*2+e,d,-1]+=1
                # compute arrival times
                if a==AEV: at+=dur
                elif a==CEV:
                    if trid[0]>-1:
                        JJ[trid[0],0,int(JJ[trid[0],0,-1])]=at[0]
                        JJ[trid[0],0,-1]+=1
                    trid[0]=0
                    at[0]=0; at[1]+=dur
                elif a==FEV:
                    if trid[1]>-1:
                        JJ[trid[1],0,int(JJ[trid[1],0,-1])]=at[1];
                        JJ[trid[1],0,-1]+=1
                    at[:]=0;trid[0]=int(e)+1;trid[1]=int(e)+3
        for gg in range(2):
            if at[gg]>0 and trid[gg]>-1:
                JJ[trid[gg],1,int(JJ[trid[gg],1,-1])]=at[gg]
                JJ[trid[gg],1,-1]+=1
        return DD,JJ
    #pool=Pool(processes=8)
    for adult in [0,1]:#todo
        for shuffle in [0,1]:
            path=initPath(adult,shuffle)
            if not shuffle: C=loadSubjectMetadata(path=path)
            N=C.shape[0]
            D=-np.ones((N,8,2,301))# event duration
            D[:,:,:,-1]=0#np.zeros((N,8,2))# event count
            J=-np.ones((N,5,2,301))# arrival times
            J[:,:,:,-1]=0
            for i in range(N):
                DD=np.copy(D[i,:,:,:])
                JJ=np.copy(J[i,:,:,:])
                CC=C[i,0]+shuffle*100
                PP=path
                DD,JJ=_helpFun(DD,JJ,CC,PP,PLOT)
                #DD,JJ=pool.apply_async(_helpFun,[DD,JJ,CC,PP]
                D[i,:,:,:]=DD
                J[i,:,:,:]=JJ
            np.save(path['outpath']+'duration'+path['suf'],D)
            np.save(path['outpath']+'arivt'+path['suf'],J)
            np.save(path['outpath']+'group'+path['suf'],C[:,3]+1)

def stanAll(NCORE=6):
    '''dst=0 estimates duration, dst=1 estimate arival times'''
    model= """
    data {
        int<lower=0> N; // nr of subjects
        int<lower=0> E; // max event number
        real<lower=-1> duration[N,2,E]; //  in seconds
        int<lower=0> maxE[N,2];
        int<lower=0,upper=4> cond[N];// experimental condition
    }
    parameters {
        real<lower=-20,upper=20> mus[N];
        real<lower=0,upper=5> sigmas;
        real<lower=-10,upper=10> hmus[4];
        real<lower=0,upper=5> hsigmas[4];
    }

    model {
        //for (g in 1:4) hsigmas[g]~cauchy(0,1);
        for (n in 1:N){ // iterate over subject
            mus[n]~normal(hmus[cond[n]],hsigmas[cond[n]]);
        for (e in 1:maxE[n,1])
            duration[n,1,e]~lognormal(mus[n],sigmas);
        for (e in 1:maxE[n,2]){
            increment_log_prob(lognormal_ccdf_log(
                duration[n,2,e],mus[n],sigmas));
            }  
        }
    }
    """
    smDur = pystan.StanModel(model_code=model)
    def _hlp(adult,shuffle,f):
        print adult,shuffle,f
        path=initPath(adult,shuffle)
        fn=['duration','arivt'][dst];pref=['Dur','Timeto'][dst]
        dur=np.load(path['outpath']+fn+path['suf']+'.npy')
        count=np.int32(dur[:,:,:,-1])
        cond=np.load(path['outpath']+'group'+path['suf']+'.npy')
        dat={'duration':dur[:,f,:,:-1],'N':dur.shape[0],'E':dur.shape[3]-1,
             'maxE':count[:,f,:],'cond':cond}
        fitDur=smDur.sampling(data=dat,iter=5000,warmup=2000,chains=NCORE,
                seed=4,thin=5,n_jobs=NCORE)
        saveStanFit(fitDur,path['outpath']+pref+path['suf']+str(f)+'.stanfit')
        if np.all(fitDur.summary()['summary'][:,-1]<1.05): print 'OK'
        else: print 'FAIL'
        f=open(path['outpath']+pref+path['suf']+str(f)+'.check','w')
        print >> f,fitDur
        f.close()
    for dst in [0,1]:
        for adult in [0,1]:
            for shuffle in [[0,1],[0]][adult]:
                for f in [[4,5,6,7],range(5)][dst]:
                    _hlp(adult,shuffle,f)

def plotSuppDur(clrs=['r','b','g']):
    figure(size=3,aspect=0.6)
    contrast=np.zeros((3600,4,2))
    for shuffle in range(2):
        for adult in [0,1]:
            if shuffle and adult: continue
            for f in [4,5,6,7]:
                sel=[range(4),[0,1,3]][adult];x=[[3,2,1,4],[3,2,4]][adult]
                path=initPath(adult,shuffle)
                fitDur=loadStanFit(path['outpath']+'Dur'+path['suf']+str(f)+'.stanfit')
                cond=np.load(path['outpath']+'group'+path['suf']+'.npy')-1
                #if f==4 and shuffle==1: continue
                ii=[1,3,2,4][f-4]
                subplot(2,2,ii)
                #if not shuffle: plt.title(LBLS[0][i])
                errorbar(np.exp(fitDur['hmus'][:,sel]),
                    x=np.array(x)+2*4*shuffle+4*adult,clr=clrs[adult+2*shuffle])
                plt.xlim([0.5,12.5])
                plt.ylim([0,[1,10][f>5]])
                plt.grid(False,axis='x')
                ax=plt.gca()
                ax.set_xticks(range(1,13))
                #if not ii%2:plt.ylabel('Median Duration in Seconds')
                
                if ii<3:
                    ax.set_xticklabels(['G1','G2','G3','G4']*3)
                    plt.title(['$\mathrm{E}_1$','$\mathrm{E}_2$'][ii==2])
                else:ax.set_xticklabels([])
                if ii%2: plt.ylabel(['Train','Test'][ii==3])
                if not shuffle and not adult: subplotAnnotate(loc='ne')

    subplot(2,2,2)
    handles=[]
    for i in range(len(clrs)):
        handles.append(mpl.patches.Patch(color=clrs[i],label=LBLS[1][i]))
    lg=plt.legend(handles=handles,loc=0,frameon=True)
    lg.get_frame().set_facecolor('#dddddd')
    lg.get_frame().set_linewidth(0.)
    plt.savefig(path['figpath']+'stan'+os.path.sep+'SuppDur.png',dpi=DPI)


def stanTrainTest(NCORE=6):
    ''' required to compute '''
    model="""
    data {
        int<lower=0> N; // nr of subjects
        int<lower=0> E; // max event number
        real<lower=-1> duration[N,2,E]; //  in seconds
        int<lower=0> maxE[N,2];
    }
    parameters {
        real<lower=-50,upper=50> mus[N];
        real<lower=0,upper=100> sigmas;
        real<lower=-50,upper=50> hmus;
        real<lower=0,upper=100> hsigmas;
    }
    model {
        //for (g in 1:3) hsigmas[g]~cauchy(0,5); sigmas[g]~cauchy(0,5);
        for (n in 1:N){ // iterate over subject
            mus[n]~normal(hmus,hsigmas);
            for (e in 1:maxE[n,1])
                duration[n,1,e]~lognormal(mus[n],sigmas);
            for (e in 1:maxE[n,2]){
                increment_log_prob(lognormal_ccdf_log(
                        duration[n,2,e],mus[n],sigmas));
    }}}
    """
    smDur = pystan.StanModel(model_code=model)
    def _hlp(adult,shuffle,f):
        print adult,shuffle,f
        path=initPath(adult,shuffle)
        fn=['duration','arivt'][dst];pref=['TTDur','TTTimeto'][dst]
        dur=np.load(path['outpath']+fn+path['suf']+'.npy')
        count=np.int32(dur[:,:,:,-1])
        cond=np.load(path['outpath']+'group'+path['suf']+'.npy')
        dat={'duration':dur[:,f,:,:-1],'N':dur.shape[0],'E':dur.shape[3]-1,
             'maxE':count[:,f,:],'cond':cond}
        fitDur=smDur.sampling(data=dat,iter=5000,warmup=2000,chains=NCORE,
                seed=4,thin=5,n_jobs=NCORE)
        saveStanFit(fitDur,path['outpath']+pref+path['suf']+str(f)+'.stanfit')
        if np.all(fitDur.summary()['summary'][:,-1]<1.05): print 'OK'
        else: print 'FAIL'
        f=open(path['outpath']+pref+path['suf']+str(f)+'.check','w')
        print >> f,fitDur
        f.close()
    for dst in [0,1]:
        for adult in [0]:
            for shuffle in [1]:
                for f in [[6,7],[1,2]][dst]:
                    _hlp(adult,shuffle,f)
def plotDur():
    figure(size=3,aspect=0.4)
    nm='Dur';legloc=0;clrs=['r','b','g'];ylims=[1,10]
    contrast=np.zeros((3600,4,2))
    for f in [6,7]:#[4,5,6,7]:
        for adult in [1,0]:
            sel=[range(4),[0,1,3]][adult];x=[[3,2,1,4],[3,2,4]][adult]
            path=initPath(adult,0)
            fitDur=loadStanFit(path['outpath']+nm+path['suf']+str(f)+'.stanfit')
            cond=np.load(path['outpath']+'group'+path['suf']+'.npy')-1
            #ii=[1,3,2,4][f-4]
            subplot(1,2,f-5)
            #if not shuffle: plt.title(LBLS[0][i])
            errorbar(np.exp(fitDur['hmus'][:,sel]),
                x=np.array(x)+4*adult,clr=clrs[adult])
            if f<7:plt.ylabel('$\mathrm{E}_2$ Median Duration in Seconds')
        path=initPath(0,1)
        fitRnd=loadStanFit(path['outpath']+'TT'+nm+path['suf']+str(f)+'.stanfit')
        errorbar(np.exp(fitRnd['hmus']),x=[9],clr=clrs[2])
        plt.xlim([0.5,9.5])
        plt.ylim([0,6])
        plt.grid(False,axis='x')
        ax=plt.gca()
        ax.set_xticks(range(1,9))
        ax.set_xticklabels(['G1','G2','G3','G4']*2)
        plt.title(['Train','Test'][f-6])
        subplotAnnotate(loc='ne')
    subplot(1,2,1);handles=[];
    for i in range(len(clrs)):
        handles.append(mpl.patches.Patch(color=clrs[i],label=LBLS[1][i]))
    lg=plt.legend(handles=handles,loc=2,frameon=True)
    lg.get_frame().set_facecolor('#dddddd')
    lg.get_frame().set_linewidth(0.);
    plt.savefig(path['figpath']+'stan'+os.path.sep+nm+'.png',dpi=DPI)    

def plotSuppInterev():
    nm='Timeto';clrs=['r','b','g'];legloc=2;nmout='SuppInterev'
    lbls=[r'$\mathrm{E}_1 \rightarrow \mathrm{E}_1$',
          r'$\mathrm{E}_2^\mathrm{Tr}\rightarrow \mathrm{E}_1$',
          r'$\mathrm{E}_2^\mathrm{Te}\rightarrow \mathrm{E}_1$',
          r'$\mathrm{E}_2^\mathrm{Tr}\rightarrow \mathrm{E}_2$',
          r'$\mathrm{E}_2^\mathrm{Te}\rightarrow \mathrm{E}_2$',]
    figure(size=3,aspect=0.6)
    contrast=np.zeros((3600,4,2))
    for shuffle in range(2):
        for adult in [0,1]:
            if shuffle and adult: continue
            for f in range(5):
                sel=[range(4),[0,1,3]][adult];x=[[3,2,1,4],[3,2,4]][adult]
                path=initPath(adult,shuffle)
                fitDur=loadStanFit(path['outpath']+nm+path['suf']+str(f)+'.stanfit')
                cond=np.load(path['outpath']+'group'+path['suf']+'.npy')-1
                ii=[2,0,1,3,4][f]
                subplot(2,3,ii+1)
                plt.title(lbls[f])
                errorbar(np.exp(fitDur['hmus'][:,sel]),
                    x=np.array(x)+2*4*shuffle+4*adult,clr=clrs[adult+2*shuffle])
                plt.xlim([0.5,12.5])
                plt.ylim([0,[14,120][ii>2]])
                plt.grid(False,axis='x')
                ax=plt.gca()
                ax.set_xticks(range(1,13))
                #if ii<3:ax.set_xticklabels(['G1','G2','G3','G4']*3)
                #else:
                ax.set_xticklabels([])
                if ii%3!=0: ax.set_yticklabels([])
                if not shuffle and not adult: subplotAnnotate(loc='nw')
    subplot(2,3,5)
    handles=[]
    for i in range(len(clrs)):
        handles.append(mpl.patches.Patch(color=clrs[i],label=LBLS[1][i]))
    lg=plt.legend(handles=handles,loc=3,bbox_to_anchor=(1.3, 0.5),frameon=True)
    lg.get_frame().set_facecolor('#dddddd')
    lg.get_frame().set_linewidth(0.)
    subplot(2,3,1)
    plt.text(-2.7,10,'Median Duration in Seconds',size=12,rotation='vertical')
    plt.savefig(path['figpath']+'stan'+os.path.sep+nmout+'.png',dpi=DPI)

def plotInterev():
    nm='Timeto';clrs=['r','b','g'];legloc=2;nmout='Interev'
    lbls=[r'$\mathrm{E}_2^\mathrm{Tr}\rightarrow \mathrm{E}_1$',
          r'$\mathrm{E}_2^\mathrm{Te}\rightarrow \mathrm{E}_1$']
    figure(size=3,aspect=0.4)
    contrast=np.zeros((3600,4,2))
    for f in [1,2]:
        for adult in [0,1]:
            sel=[range(4),[0,1,3]][adult];x=[[3,2,1,4],[3,2,4]][adult]
            path=initPath(adult,0)
            fitDur=loadStanFit(path['outpath']+nm+path['suf']+str(f)+'.stanfit')
            cond=np.load(path['outpath']+'group'+path['suf']+'.npy')-1
            subplot(1,2,f)
            errorbar(np.exp(fitDur['hmus'][:,sel]),
                x=np.array(x)+4*adult,clr=clrs[adult])
        path=initPath(0,1)
        fitRnd=loadStanFit(path['outpath']+'TT'+nm+path['suf']+str(f)+'.stanfit')
        errorbar(np.exp(fitRnd['hmus']),x=[9],clr=clrs[2])
        plt.title(lbls[f-1])
        plt.xlim([0.5,9.5])
        plt.ylim([0,7])
        plt.grid(False,axis='x')
        ax=plt.gca()
        ax.set_xticks(range(1,9))
        ax.set_xticklabels(['G1','G2','G3','G4']*2)
        subplotAnnotate(loc='nw')

    subplot(1,2,1)
    #plt.text(-3,7,'Median Duration in Seconds',size=12,rotation='vertical')
    handles=[]
    for i in range(len(clrs)):
        handles.append(mpl.patches.Patch(color=clrs[i],label=LBLS[1][i]))
    lg=plt.legend(handles=handles,loc=9,bbox_to_anchor=(0.7, 1),frameon=True)
    lg.get_frame().set_facecolor('#dddddd')
    lg.get_frame().set_linewidth(0.)
    #subplot(2,3,1)
    #plt.text(-2.7,10,'Median Duration in Seconds',size=12,rotation='vertical')
    plt.savefig(path['figpath']+'stan'+os.path.sep+nmout+'.png',dpi=DPI)


def stanContrast(NCORE=6):
    model="""
    data {
        int<lower=0> N; // nr of subjects
        int<lower=0> E; // max event number
        real<lower=-1> duration[N,2,2,E]; //  in seconds
        int<lower=0> maxE[N,2,2];
        int<lower=1,upper=4> cond[N];// experimental condition
    }
    parameters {
        vector<lower=-50,upper=50>[2] mus[N];
        real<lower=0,upper=100> sigmas[3];
        real<lower=-50,upper=50> hmus[3];
        real<lower=0,upper=100> hsigmas[3];
        real<lower=-1,upper=1> kappa;
    }
    transformed parameters {
        vector[2] vmu; matrix[2,2] S;
        vmu[1]<-hmus[1];vmu[2]<-hmus[2];
        for (i in 1:2) S[i,i]<-square(hsigmas[i]);
        S[1,2]<-hsigmas[1]*kappa*hsigmas[2];
        S[2,1]<-S[1,2];
    }

    model {
        real sgm;
        //for (g in 1:3) hsigmas[g]~cauchy(0,5); sigmas[g]~cauchy(0,5);
        int s; int k;
        for (n in 1:N){ // iterate over subject
            if (cond[n]==4) mus[n,1]~normal(hmus[3],hsigmas[3]);
            else mus[n]~multi_normal(vmu,S);
        for (f in 1:2){ // leave this state
            if (cond[n]==4){s<-1; sgm<-sigmas[3];} 
            else {s<-f; sgm<-sigmas[f];}
            for (e in 1:maxE[n,f,1])
                duration[n,f,1,e]~lognormal(mus[n,s],sgm);
            for (e in 1:maxE[n,f,2]){
                increment_log_prob(lognormal_ccdf_log(
                        duration[n,f,2,e],mus[n,s],sgm));
            }  
        }}
    }
    """
    smContr = pystan.StanModel(model_code=model)
    for adult in [0,1]:
        for dst in range(3):
            path=initPath(adult,0)
            fn=['duration','arivt'][dst>0];
            sel=[[6,7],[1,2],[3,4]][dst]
            dur=np.load(path['outpath']+fn+path['suf']+'.npy')
            count=np.int32(dur[:,:,:,-1])
            cond=np.load(path['outpath']+'group'+path['suf']+'.npy')
            dat={'duration':dur[:,sel,:,:-1],'N':dur.shape[0],'E':dur.shape[3]-1,
                'maxE':count[:,sel,:],'cond':cond}
            fitContr=smContr.sampling(data=dat,iter=5000,warmup=2000,chains=NCORE,
                    seed=4,thin=5,n_jobs=NCORE)
            fp=path['outpath']+fn+str(dst)+'Contr'+path['suf']
            saveStanFit(fitContr,fp+'.stanfit')
            f=open(fp+'.check','w')
            print >> f,fitContr
            f.close()  

def plotContr():
    def pairwiseComparison(stf):
        for i in range(len(stf)):
            for j in range(i+1,len(stf)):
                printCI(np.exp(stf[i]-stf[j]))
        
    figure(size=3,aspect=0.4)
    tmp=[];res=[];ylims=[3.5,6,90]
    for h in range(2):
        subplot(1,2,h+1)
        for adult in [1,0]:
            path=initPath(adult,0)
            nm=['duration','arivt'][h>0]+str(h)
            w=loadStanFit(path['outpath']+nm+'Contr'+path['suf']+'.stanfit')
            errorbar(np.exp(w['hmus']),x=np.arange(3)+adult*3,clr=['r','b'][adult])
            print ['Infant','Adult'][adult]
            if not adult: tmp.append(w['hmus']/w['hsigmas'])
            stf=[w['hmus'][:,0],w['hmus'][:,1],w['hmus'][:,2]]
            if False:#h==1:
                w=loadStanFit(path['outpath']+'arivt0'+'Contr'+path['suf']+'.stanfit')
                errorbar(np.exp(w['hmus']),x=[3+adult*4],clr=['r','b'][adult])
                stf.append(w['hmus'])
            pairwiseComparison(stf)
        ax=plt.gca()
        plt.xlim([-0.5,5.5])
        ax.set_xticks(range(6))
        ax.set_xticklabels(['Tr','Te','Co']*2)
        #if not h:plt.title(['Infant','Adult'][adult])
        if not h: plt.ylabel('Median duration in seconds')
        plt.ylim([0,ylims[h]]);
        plt.title(['$\mathrm{E}_2$',r'$\mathrm{E}_2\rightarrow \mathrm{E}_1$'][h])
        #subplotAnnotate(loc='ne')
    plt.savefig(path['figpath']+'stan'+os.path.sep+'Contr.png',dpi=DPI)
    eh=[0,0]
    for i in range(2):
        eh[0]+=([1,-1][i]*(2*tmp[i][:,0]-tmp[i][:,1]-tmp[i][:,2])
                -np.abs(tmp[i][:,1]-tmp[i][:,2]))
        eh[1]+=([1,-1][i]*(-2*tmp[i][:,2]+tmp[i][:,0]+tmp[i][:,1])
                -np.abs(tmp[i][:,1]-tmp[i][:,0]))
    p1=np.exp(eh[0])/(np.exp(eh[1])+np.exp(eh[0]))
    p2=np.square(eh[0])/(np.square(eh[1])+np.square(eh[0]))
    #L=p/(1-p)
    printCI(p1)
    printCI(p2)
    print np.mean(p1),np.mean(p2)

def stanOneGroup(NCORE=6):
    model="""
    data {
        int<lower=0> N; // nr of subjects
        int<lower=0> E; // max event number
        real<lower=-1> arivt[N,2,2,E]; //  in seconds
        int<lower=0> maxE[N,2,2];
    }
    parameters {
        real<lower=-50,upper=50> mus[N];
        real<lower=0,upper=100> sigmas;
        real<lower=-50,upper=50> hmus;
        real<lower=0,upper=100> hsigmas;
    }
    model {
        //for (g in 1:3) hsigmas[g]~cauchy(0,5); sigmas[g]~cauchy(0,5);
        for (n in 1:N){ // iterate over subject
        for (k in 1:2){
            mus[n]~normal(hmus,hsigmas);
            for (e in 1:maxE[n,k,1])
                arivt[n,k,1,e]~lognormal(mus[n],sigmas);
            for (e in 1:maxE[n,k,2]){
                increment_log_prob(lognormal_ccdf_log(
                        arivt[n,k,2,e],mus[n],sigmas));
    }}}}
    """
    smContr = pystan.StanModel(model_code=model)
    for shuffle in range(2):
        for adult in [0,1]:
            path=initPath(adult,shuffle)
            if adult and shuffle: continue
            timeto=np.load(path['outpath']+'duration'+path['suf']+'.npy')
            count=np.load(path['outpath']+'count'+path['suf']+'.npy')
            dat={'arivt':timeto[:,[4,5],:,:-1],'N':timeto.shape[0],'E':timeto.shape[3]-1,
                'maxE':np.int32(timeto[:,:,:,-1][:,[4,5],:])}
            fitContr=smContr.sampling(data=dat,iter=5000,warmup=2000,chains=NCORE,
                    seed=4,thin=5,n_jobs=NCORE)
            fp=path['outpath']+'OneGroup'+path['suf']
            saveStanFit(fitContr,fp+'.stanfit')
            f=open(fp+'.check','w')
            print >> f,fitContr
            f.close()

def plotRandom():
    figure(size=2,aspect=0.6)
    tmp=[];res=[];ylims=[0.8,10]
    for h in range(2):
        subplot(1,2,h+1);
        for x in range(3):
            pars=[[0,0],[1,0],[0,1]][x]
            path=initPath(pars[0],pars[1])
            nm=['OneGroup','arivt0Contr'][h]
            w=loadStanFit(path['outpath']+nm+path['suf']+'.stanfit')
            errorbar(np.exp(w['hmus']),x=[x],clr=['r','b','g'][x])
            printCI(np.exp(w['hmus']))
        ax=plt.gca()
        plt.xlim([-0.5,2.5])
        ax.set_xticks(range(3))
        ax.set_xticklabels(['Infant','Adult','Random'])
        #if not h:plt.title(['Infant','Adult'][adult])
        if not h: plt.ylabel('Median duration in seconds')
        plt.ylim([0,ylims[h]]);
        plt.title(['$\mathrm{E}_1$',r'$\mathrm{E}_1\rightarrow \mathrm{E}_1$'][h])
        #subplotAnnotate(loc='ne')
    #plt.savefig(path['figpath']+'stan'+os.path.sep+'Random.png',dpi=DPI)
       
if __name__=='__main__':
    pass
    #plotBasicEvents(adult=True)
    #plotBasicEvents()

##    path=initPath(0,0)
##    for vpn in path['allvpn']:
##        for vp in vpn:
##            try:plotBabyEvents(vp,path)
##            except IOError: print 'missing', vp
    #evs2stan(True)
    #stanAll()
    #stanContrast()
    #stanTrainTest()
    #stanOneGroup()

    #plotSuppDur()
    #plotDur()
    #plotSuppInterev()
    plotInterev()
    #plotContr()

    

