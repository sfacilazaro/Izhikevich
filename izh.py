#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 20:27:11 2023

@author: sergio
"""

#if __name__ == "__main__":
#    main() 

import matplotlib.pyplot as plt
import numpy as np

#network parameters
N = 400            #size of the network
I = 16             #number of modules

#statistics
nSTAT = 5          #number of cultures for statistics

alpha = 1.0        #connection probability
alphaDEL = 0.05 
alphaMIN = 0.00 
alphaMAX = 1.00 
alphas = np.linspace(alphaMIN,alphaMAX,int((alphaMAX-alphaMIN)/alphaDEL)+1)

beta = 1.0         #jump probability
betaDEL = 0.05 
betaMIN = 0.00 
betaMAX = 1.00 
betas = np.linspace(betaMIN,betaMAX,int((betaMAX-betaMIN)/betaDEL)+1)

#dynamical parameter
veq = 0            #membrane potential in the equilibrium should it exist
a = 7.5            #quadratic growth voltage factor
b = 0.5            #sensitivity to subthreshold fluctuations - b>0 resonator | b<0 integrator
c = 0.6            #reset condition for the membrane potential
d = 3.3            #reset condition for the inhibitory current
p = 6              #peak value for the membrane potential
r = 0              #resting potential
t = 1              #threshold potential
g = 3              #synaptic current strength
gs = 0.5           #shot noise strength/mini current strength
gw = 1.5           #white noise strength, std
eta = 0.75         #white noise strength, mean
lm = 0             #minimum frequency for mini currents
lp = 1             #maximum frequency for mini currents
la = 0.5           #average frequency for mini currents

#simulation parameters
tsERM = 1000       #termalization time in time steps
tsSIM = 5000       #simulation time in time steps
dt = 0.02          #time step
tsp = 1            #precision for recording time steps

#global variables


#functions
def A2L(AA):
    LL = []
    for i in range(len(AA)):
        arr = np.nonzero(AA[:,i])[0]
        LL.append(arr.tolist())
    return LL

def whitenoise():
    return eta + np.fabs(gw*np.random.normal(0,1))
    
 
def shotnoise(LL):
    ss =  np.zeros(N, dtype=float)
    snprob = dt * np.random.uniform(lm,lp,N)  #shot noise probability lx*dt probability that a shot noise event happens in each time step
    ran = np.random.uniform(0,1,N)
    act = np.where(ran < snprob)[0]
    for nn in range(N):
        ss[nn] = gs * len(np.intersect1d(act, LL[nn]))
    return ss

def fv(vv,uu,ww,ss,LL):
    act = np.where(vv > 1)[0]
    II = np.empty(N, dtype=float)
    for nn in range(N):
        II[nn] = g * len(np.intersect1d(act, LL[nn]))
    return a * vv * (vv - 1) - uu + II + ww + ss

def fu(vv,uu):
    return b * vv - uu

def ctev(vv,uu,ww,ss,LL):
    aux = vv
    k1 = fv(aux, uu, ww, ss, LL)
    
    aux = vv + k1 * dt / 2
    k2 = fv(aux, uu, ww, ss, LL)
    
    aux = vv + k2 * dt / 2
    k3 = fv(aux, uu, ww, ss, LL)
    
    aux = vv + k3 * dt
    k4 = fv(aux, uu, ww, ss, LL)
    return (k1 + 2*k2 + 2*k3 + k4)/6

def cteu(vv,uu):
    aux = uu
    k1 = fu(vv, aux)
    
    aux = uu + k1 * dt / 2
    k2 = fu(vv, aux)
    
    aux = uu + k2 * dt / 2
    k3 = fu(vv, aux)
    
    aux = uu + k3 * dt
    k4 = fu(vv, aux)
    return (k1 + 2*k2 + 2*k3 + k4)/6


def izhRK4(vo,uo,ww,ss,LL):
    vn = vo + dt * ctev(vo,uo,ww,ss,LL)
    un = uo + dt * cteu(vo,uo)
    ind = np.where(vn > p)[0]
    if len(ind):
        vn[ind] = c
        un[ind] += d
    return vn, un
 
        
def izhRP():
    #LOOPS alpha/neta/nstat/nit
        
    #variables
    #v =  np.zeros(N, dtype=float)           #membrane potential
    vOLD = np.random.uniform(size = N)      #np.random.uniform(size = N) np.zeros(N, dtype=float) 
    #u =  np.zeros(N, dtype=float)           #inhibitory current
    uOLD = b*vOLD                           #b*vOLD                      
    active = np.full(N, False)              #flag for active neurons
    #sn = np.empty(N, dtype=float)           #shot noise
    #wn = whitenoise()                       #white noise
    
    #name = f"A-a{alpha:0.2f}-b{beta:0.2f}-nst{nstat:02d}.txt"    
    #A = np.loadtxt(name, dtype='int', delimiter=' ')
    A = np.loadtxt('M.txt', dtype='int', delimiter=' ') 
    L = A2L(A)
    
    file = open('RP.txt','w')
    
    for ts in range(tsERM+tsSIM):
        #write active neurons in raster plot
        active[np.where(vOLD > 1)[0]] = True
        if not(ts % tsp):
            pulse = np.where(active)[0]
            for n in range(len(pulse)):
                file.write('%0.3f\t%03d\n' % (dt * (ts-tsERM), pulse[n]))
            active = np.full(N, False)    
        #advance
        wn = whitenoise()
        sn = shotnoise(L)
        v, u = izhRK4(vOLD, uOLD, wn, sn, L)
        #if max(v)>1:
        #    print([ts, max(v)])
        vOLD = v
        uOLD = u
    
    file.close()    
    
izhRP()
RP = np.loadtxt('RP.txt', dtype='float', delimiter='\t')
plt.plot(RP[:,0],RP[:,1],'.')