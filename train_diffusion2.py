
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
    params.data='/work2/dataset/fgvc'
    #params.root='/work/dataset/fgvc-aircraft/fgvc-aircraft-2013b/data/images'
    #model
    params.arch='arch.diffusion';
    params.nh=32;
    
    params.length=9; #video length
    params.imsz=128; #video size
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
        self.transform=transforms.Compose([lambda image: image.convert("RGB"),transforms.Resize((params.imsz,params.imsz)),transforms.CenterCrop(params.imsz),transforms.ToTensor()]);
    
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self,i):
        fname=os.path.join(self.root,self.fnames[i])
        frame=torchvision.datasets.folder.default_loader(fname);
        frame=self.transform(frame);
        return frame

data_train=pandas.read_csv(os.path.join(params.data,'train.txt'))
data_test=pandas.read_csv(os.path.join(params.data,'test.txt'))


#data_train=torch.load(os.path.join(params.data,'data_fsl_train.pt'))
#data_test=torch.load(os.path.join(params.data,'data_fsl_val.pt'))

dataset_train=Dataset(list(data_train['LQ']),root=params.data)
dataset_test=Dataset(list(data_test['LQ']),root=params.data)

data_train = DataLoader(dataset_train,batch_size=params.batch_size,shuffle=True,num_workers=8,drop_last=True);
data_test = DataLoader(dataset_test,batch_size=params.batch_size,shuffle=False,num_workers=8);

#Prepare network
import arch.diffusion as arch
net = arch.new(imsz=params.imsz).cuda()


if not (params.load is None or params.load==''):
    checkpoint=torch.load(params.load);
    net.load_state_dict(checkpoint,strict=False)


t0=time.time()
opt=optim.AdamW(list(net.parameters()),lr=params.lr,weight_decay=1e-4)


for epoch in range(1000):
    if (epoch+0)%10==0:
        torch.save(net.state_dict(),session.file('model','%d.pt'%epoch))
        net.eval()
        tracker=session_manager.loss_tracker();
        with torch.no_grad():
            loss=[];
            for i,v in enumerate(data_test):
                v=v.cuda();
                loss = net(v)
                tracker.add(loss=loss)
                
                if i==0:
                    gen = net.generate(N = 24)
                    print(gen.shape,gen.max(),gen.min())
                    util.video.write_video(gen,'debug_diffusion2.avi',fps=3)
                
                print('eval %d, %d/%d, %s         '%(epoch,i,len(data_test),tracker.str()),end='\r');
            
            session.log('eval %d, %s'%(epoch,tracker.str()))
    
    tracker=session_manager.loss_tracker();
    net.train()
    for i,vtrain in enumerate(data_train):
        vtrain=vtrain.cuda();
        opt.zero_grad()
        
        loss = net(vtrain)
        loss.backward()
        
        opt.step();
        tracker.add(loss=loss)
        
        print('train %d, %d/%d, %s, time %.2f           '%(epoch,i,len(data_train),tracker.str(),time.time()-t0),end='\r');
        if (i+1)%300==0:
            session.log('train %d/%d, %s, time %.2f           '%(i,len(data_train),tracker.str(),time.time()-t0));
            #break
    
    
    session.log('Epoch %d, %s, time %.2f          '%(epoch,tracker.str(),time.time()-t0))
    







import torch


diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000    # number of steps
)

training_images = torch.rand(8, 3, 128, 128) # images are normalized from 0 to 1
loss = diffusion(training_images)
loss.backward()

# after a lot of training

sampled_images = diffusion.sample(batch_size = 4)
sampled_images.shape # (4, 3, 128, 128)
