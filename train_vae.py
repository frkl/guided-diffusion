
#Python2,3 compatible headers
from __future__ import unicode_literals,division
from builtins import int
from builtins import range

#System packages
import torch
from torch.autograd import Variable,grad
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy
import scipy
import scipy.misc
import math
import time
import random
import pandas
import argparse
import sys
import os
import re
import copy
import importlib
import json
from collections import namedtuple
from collections import OrderedDict
import itertools

import os
import numpy as np
import copy
import torch

import warnings

import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms
BICUBIC = transforms.InterpolationMode.BICUBIC
import torchvision.datasets
from functools import partial

import util.video
import util.smartparse as smartparse
import util.session_manager as session_manager

# Training settings
def default_params():
    params=smartparse.obj();
    params.data='/work/dataset/fgvc-aircraft'
    params.root='/work/dataset/fgvc-aircraft/fgvc-aircraft-2013b/data/images'
    #model
    params.arch='arch.vae';
    params.nh=32;
    
    params.length=9; #video length
    params.sz=128; #video size
    params.load='';
    #Training
    params.batch_size=16;
    params.lr=3e-4;
    #MISC
    params.session_dir=None;
    return params


window_size=9
crop_size=15

params = smartparse.parse()
params = smartparse.merge(params, default_params())
params.argv=sys.argv;
session=session_manager.create_session(params);


#Video data loader class
class Dataset:
    def __init__(self,fnames,root=''):
        self.root=root
        self.fnames=fnames
        print('Loaded %d images'%(len(fnames)))
        self.transform=transforms.Compose([lambda image: image.convert("RGB"),transforms.Resize((224,224)),transforms.CenterCrop(224),transforms.ToTensor()]);
    
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self,i):
        fname=os.path.join(self.root,self.fnames[i])
        frame=torchvision.datasets.folder.default_loader(fname);
        frame=self.transform(frame);
        return frame


data_train=torch.load(os.path.join(params.data,'data_fsl_train.pt'))
data_test=torch.load(os.path.join(params.data,'data_fsl_val.pt'))

dataset_train=Dataset(data_train['table_ann']['imname'],root=params.root)
dataset_test=Dataset(data_test['table_ann']['imname'],root=params.root)

data_train = DataLoader(dataset_train,batch_size=params.batch_size,shuffle=True,num_workers=8,drop_last=True);
data_test = DataLoader(dataset_test,batch_size=params.batch_size,shuffle=False,num_workers=8);

#Prepare network
arch=importlib.import_module(params.arch);
net=arch.new(nh=params.nh,imsz=224).cuda();
divider=torch.Tensor(1).fill_(0).cuda().requires_grad_();

if not (params.load is None or params.load==''):
    checkpoint=torch.load(params.load);
    net.load_state_dict(checkpoint,strict=False)


t0=time.time()
opt=optim.AdamW(list(net.parameters())+[divider],lr=params.lr,weight_decay=1e-4)


for epoch in range(1000):
    if (epoch+1)%10==0:
        torch.save(net.state_dict(),session.file('model','%d.pt'%epoch))
        net.eval()
        tracker=session_manager.loss_tracker();
        with torch.no_grad():
            loss=[];
            for i,v in enumerate(data_test):
                v=v.cuda();
                recon,kl=net(v);
                s=torch.exp((divider*30).clamp(max=20,min=-20)) # std for likelihood
                diff=((recon-v).abs()+1/255)*255 # Add a 1 pixel error. This is to prevent likelihood from collapsing for easy-to-predict futures
                nll=torch.log(s)+(diff/s)**2/2
                nll=nll.sum(dim=-3,keepdim=True) # Sum over channels. KL is computed in hidden space where nh=32. Likelihood is computed in pixel space where C=3. We sum over nh and C respectively to make KL and likelihood match
                
                nll=nll.mean()
                kl=kl.mean()
                loss_vae=nll+kl
                
                #Monitor reconstruction pixel error
                err=(recon-v).abs().sum(dim=-3,keepdim=True).mean() #Reconstruction absolute pixel error, summed over RGB channels
                tracker.add(loss_vae=loss_vae,nll=nll,kl=kl,err=err,s=s)
                
                if i==0:
                    gen=net.generate(10)
                    print(gen.shape,gen.max(),gen.min())
                    util.video.write_video(gen,'debug.avi',fps=30)
                
                print('eval %d, %d/%d, %s         '%(epoch,i,len(data_test),tracker.str()),end='\r');
            
            session.log('eval %d, %s'%(epoch,tracker.str()))
    
    tracker=session_manager.loss_tracker();
    net.train()
    for i,vtrain in enumerate(data_train):
        vtrain=vtrain.cuda();
        opt.zero_grad()
        recon,kl=net(vtrain);
        
        s=torch.exp((divider*30).clamp(max=20,min=-20)) # std for likelihood
        diff=((recon-vtrain).abs()+1/255)*255 # Add a 1 pixel error. This is to prevent likelihood from collapsing for easy-to-predict futures
        nll=torch.log(s)+(diff/s)**2/2
        nll=nll.sum(dim=-3,keepdim=True) # Sum over channels. KL is computed in hidden space where nh=32. Likelihood is computed in pixel space where C=3. We sum over nh and C respectively to make KL and likelihood match
        
        nll=nll.mean()
        kl=kl.mean()
        
        #Monitor reconstruction pixel error
        err=(recon-vtrain).abs().sum(dim=1,keepdim=True).mean(); #Reconstruction absolute pixel error, summed over RGB channels
        
        loss_vae=nll+kl
        loss_vae.backward()
        opt.step();
        
        tracker.add(loss_vae=loss_vae,nll=nll,kl=kl,err=err,s=s)
        
        print('train %d, %d/%d, %s, time %.2f           '%(epoch,i,len(data_train),tracker.str(),time.time()-t0),end='\r');
        if (i+1)%300==0:
            session.log('train %d/%d, %s, time %.2f           '%(i,len(data_train),tracker.str(),time.time()-t0));
            #break
    
    
    session.log('Epoch %d, %s, time %.2f          '%(epoch,tracker.str(),time.time()-t0))
    

