import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import QuantileTransformer,MinMaxScaler
import sympy as sp
import math
import os
import argparse
parser = argparse.ArgumentParser(prog='BEAR model inference',description='training BEAR model with input data')
parser.add_argument('--nfeatures', type=int, help='number of priors used in BEAR',required=True, default=3)
parser.add_argument('--lr',type=float,help='learning rate',required=True)
parser.add_argument('--nclusters',type=int,help='number of clusters of the BEAR',required=False,default="./")
parser.add_argument('--data_file',type=str,help='prefix of input dataset',required=True,default=None)
parser.add_argument('--eps',type=float,help='eps used for convergence',required=False,default=1e-3)
parser.add_argument('--nepochs',type=int,help='number of epochs of running',required=True,default=1e-3)
parser.add_argument('--output',type=str,help="output file",required=True)
args = parser.parse_args()

LR=args.lr
EPS=args.eps
N=args.nfeatures
FILE=args.data_file
NEPOCHS=args.nepochs
HPARAMS={'a0':1,'b0':1,'c0':1,'d0':1,'p0':10,'q0':2,'r0':1,'s0':1,'v0':1,'k0':2,'muk0':1,'lam0':10,'j':1,'l0':10,'m0':-5.0}
OUTPUT=args.output
LOG=args.log

def sigmoid(x1):
    return 1.0/(1+ np.exp(-1*x1.astype(float)))

def lambda_epi(x):
    return 0.5*(sigmoid(x)-0.5)/x

def dxy(a1,a2,p0,p,q,lr=LR, eps=EPS):
    x, y = sp.symbols('x y',real=True)
    i=0
    fx=a1*(x**2+x)/y**2+(a2+y)*x/y+(p0-x)*(sp.digamma(x,evaluate=False)-sp.log(y))
    print(fx)
    valm=0
    val = fx.subs({x: p, y: q}).evalf(4)
    #dp,dq=diff(f,x,1).subs({x:p,y:q}).evalf(4),diff(f,y,1).subs({x:p,y:q}).evalf(4)
    while abs(val-valm)>=EPS:
        dp, dq = sp.diff(fx, x, 1).subs({x: p, y: q}).evalf(4), sp.diff(fx, y, 1).subs({x: p, y: q}).evalf(4)
        p=p+lr*dp
        q=q+lr*dq
        valm=val
        val=fx.subs({x:p,y:q}).evalf(4)
        if abs(val-valm)<=eps:
            print(val)
            break
    return p, q



class sig_inf():
    def __init__(self, input, features,nclusters=N,HPARAMS=HPARAMS,lr=LR):
        self.K_input = input.shape[1]
        self.features=features
        self.input=input
        initialize(HPARAMS)
        #self.qn=self.q0
    def initialize(self,HPARAMS):
        self.K_features= N_FEATURES
        assert self.K_features==features.shape[1]
        self.nclusters=nclusters
        self.N=self.input.shape[0]
        self.a0,self.b0,self.c0,self.d0=HPARAMS['a0'],HPARAMS['b0'],HPARAMS['c0'],HPARMS['d0']
        self.p0,self.q0.self.r0,self.s0=HPARAMS['p0'],HPARAMS['q0'],HPARAMS['r0'],HPARMS['s0']
        self.v0,self.k0,self.muk0,self.lam0=HPARAMS['v0'],HPARAMS['k0'],HPARAMS['muk0'],HPARMS['lam0']
        self.cn=self.c0+self.N / 2
        self.pn=self.p0+1
        self.qn=self.q0
        self.u=self.pn / self.qn
        self.tau=self.c0/ self.d0
        self.H0=np.eye(self.K_features)
        self.mu0, self.w = np.repeat(HPARAMS['h0'],self.K_features)[np.newaxis],  np.repeat(HPARAMS['w'],self.K_features)[np.newaxis]
        self.lam=self.lam0 + 1
        self.v=self.v0 + 1
        self.k=self.k0 + 1
        self.epi, h=np.random.uniform(size=(self.N,1)), np.random.uniform(size=(self.N,1))
        self.fi=sigmoid(h)
        self.w2=self.w.dot(self.w.T)
        self.mm=HPARAMS['m0']
        self.j=HPARAMS['j']+0.5
        self.l0, self.l=HPARAMS['l0'], HPARAMS['l0']
        
    def update(self):

        features=self.features
        input=self.input
        a,sw,epi,f,mk=[],[],[],[],[]

        self.mu=(self.mu0*self.k0+self.w)/self.k
        self.H = np.linalg.inv(np.linalg.inv(self.H0) + self.k0 * self.mu0.dot(self.mu0.T) + self.w2)

        for i,feature in enumerate(features):
            feature=feature.reshape(-1,1)
            epi.append(np.sqrt(feature.T.dot(self.w2).dot(feature)))
            sw.append(2*lambda_epi(epi[-1])*feature.dot(feature.T))

        self.epi=np.asarray(epi).reshape(-1,1)
        self.sw=np.linalg.inv(np.asarray(sw).sum(axis=0)+self.v*self.H)
        self.mw = self.sw.dot(np.sum((self.fi - 0.5) * features, axis=0).reshape(-1,1)+self.v*self.H.dot(self.mu))
        self.w=self.mw
        self.w2=self.mw.dot(self.mw.T)+self.sw              #W


        self.sigma = 1 / (self.j / self.l + self.N * self.tau)
        self.mm=self.sigma * (input - self.fi * self.u).sum(0) * self.tau+ 0.5* self.sigma* self.j/self.l
        self.l=self.l0+0.5*(self.mm**2+self.sigma-self.mm)
        self.sn=self.s0+self.u
        self.eq=self.r0/self.sn

        self.a1=self.tau*self.fi.sum()*-0.5
        self.a2=self.tau*((input-self.mm)*self.fi).sum()-self.eq
        self.pn,self.qn=dxy(self.a1,self.a2,self.p0,self.pn,self.qn)
        self.u = self.pn / self.qn
        self.u2 = (self.pn ** 2 + self.pn) / (self.qn ** 2)

        self.dn = self.d0 + 1 / 2 * (input ** 2 + self.fi * self.u2 - 2 *(input*(self.mm + self.fi * self.u) + 2 * self.mm * self.u * self.fi)).sum() + 1 / 2 * self.N * (self.mm ** 2 + self.sigma)
        self.tau = (self.cn / self.dn).item()

        for i, feature in enumerate(features):
            feature = feature.reshape(-1, 1)
            inp = input[i, :]
            f.append(self.w.T.dot(feature) + ((inp - self.mm) * self.u - self.u2 / 2) * self.tau)
            a.append(self.w.T.dot(feature))
        self.f = np.asarray(f).reshape(-1, 1)
        self.a = np.asarray(a).reshape(-1, 1)

        self.fi = sigmoid(self.f)
        #self.mk=self.sk*(self.muk*self.an/self.bn+(self.fi*(self.input-self.mm)).sum()*self.tau)
        #self.muk = (self.lam0 * self.muk0 + self.mk) / self.lam
        #self.u2 = 2 / (self.qn ** 2)

        #self.u=self.mk

        #self.bn = self.b0 + 0.5 * (self.sk + self.mk ** 2 + self.muk0**2-2*self.muk0*self.mk)*self.lam0/self.lam
        #self.k = self.k0 + 0.5 * (self.mm ** 2 + self.sigma)
        #self.u2 = self.mk ** 2 + self.sk
        #self.bn=self.b0+1/self.qn
        #self.eq=self.an/self.bn

        return self.w,self.u
    
    def obj(self):
        t1=-0.5*np.log(np.trace(self.sw))
        t2=-0.5*self.mw.T().dot(np.linalg.inv(self.sw)).dot(self.mw)
        t3=0
        for i in range(self.N):
            t3=t3-np.log(sigmoid(self.epi[i]))-0.5*self.epi[i]+lambda_epi(self.epi[i])*self.epi[i]**2
        t4=-1.0*np.sum((self.mm**2 +1)/(2*self.sigma**2))
        t5=np.sum((-0.5*np.add.outer(self.input**2, self.m**2+1) + np.outer(self.input,self.m)).self.fi)
        t6=-1.0*np.sum(np.log(self.fi))
        return t1+t2+t3+t4+t5+t6


    def train(self,iters,eps=EPS):
        yw,yu,lln = [],[],[]
        neww,newu=self.update1()
        yw.append(neww)
        yu.append(newu)
        lln.append(self.obj())
        for i in range(iters):
            neww, newu = self.update()
            yw.append(neww)
            yu.append(newu)
            lln.appedn(self.obj())
            #if abs(yu[-1]- yu[-2]) < eps:
                #break
            if abs(lln[-1]-lln[-2])<eps:
                break
        return lln, yw,yu


def main():
    
    variants=pd.read_table(FILE,sep="\t",header=0,index_col=0)
    prior_features_index=[i+2 for i in range(N)]
    test_features_index=[1]
    x=variants.iloc[:,prior_features]
    y=variants.iloc[:,test_features]
    x=x.div(x.max(),1)
    x['bias']=1.0
    x=np.asarray(x)
    id=variants.index
    input=abs(np.asarray(y)).reshape(-1,1)
    if LOG:
        input=np.log2(input+EPS)

    Inf=sig_inf(input=input,features=x,nclusters=2)
    
    Inf.train(NSTEPS)
    
    scaled_fi=pd.DataFrame(MinMaxScaler().fit(Inf.fi).transform(Inf.fi),index=id)

    out=pd.concat((variants,fi,scaled_fi),axis=1)
    out.to_csv(OUTPUT,sep="\t",index=True)

if __name__ == '__main__':
    main()
