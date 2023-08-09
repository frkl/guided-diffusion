
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import random



class new(nn.Module):
    def __init__(self,nh=64,imsz=224):
        super().__init__()
        self.encoder=smp.Unet(encoder_name="resnet18",in_channels=3,classes=2*nh)
        self.decoder=smp.Unet(encoder_name="resnet18",in_channels=nh,classes=3)
        self.t=nn.Conv2d(4*nh,4*nh,1)
        self.m=nn.Parameter(torch.Tensor(1,2*nh,imsz,imsz).fill_(0))
        self.s=nn.Parameter(torch.Tensor(1,2*nh,imsz,imsz).fill_(0))
        self.m2=nn.Parameter(torch.Tensor(1,2*nh,imsz,imsz).fill_(0))
        self.s2=nn.Parameter(torch.Tensor(1,2*nh,imsz,imsz).fill_(0))
        
        self.nh=nh
        self.imsz=imsz
        return;
    
    def encode(self,v):
        h=self.encoder(v)
        h=torch.fft.fft2(h)
        h=(h-self.m)*torch.exp(self.s.clamp(-20,20))
        h=torch.cat((h.real,h.imag),dim=-3)
        h=self.t(h)
        
        hmean=h[:,::2,:,:]
        hvar=h[:,1::2,:,:]**2
        hstd=h[:,1::2,:,:].abs()
        return hmean,hvar,hstd
    
    def decode(self,h_gen):
        h_gen=(h_gen+self.m2)*torch.exp(self.s2.clamp(-20,20))
        h_gen=torch.complex(h_gen[:,::2,:,:],h_gen[:,1::2,:,:])
        h_gen=torch.fft.ifft2(h_gen)
        h_gen=h_gen.real
        recon=self.decoder(h_gen)
        return recon
    
    
    def forward(self,vtrain,vtest):
        eps=1e-20
        hmean_train,hvar_train,hstd_train=self.encode(vtrain)
        hmean_test,hvar_test,hstd_test=self.encode(vtest)
        
        h_fsmean=hmean_train.mean(dim=-4,keepdim=True)
        h_fsvar=hvar_train.mean(dim=-4,keepdim=True)+hmean_train.var(dim=-4,keepdim=True)
        h_fsstd=(h_fsvar+eps)**0.5
        
        h_gen=hmean_test+hstd_test*hstd_test.data.clone().normal_()
        recon_test=self.decode(h_gen)
        
        kl=-torch.log((hvar_test+eps)/(h_fsvar+eps))/2+(hvar_test+(hmean_test-h_fsmean)**2)/2/(h_fsvar+eps)-0.5;
        kl=kl.sum(dim=-3,keepdim=True) #Sum KL over channel dimension, now NT1HW
        
        return recon,kl
    
    def generate_fs(self,vtrain,N):
        eps=1e-20
        hmean_train,hvar_train,hstd_train=self.encode(vtrain)
        h_fsmean=hmean_train.mean(dim=-4,keepdim=True)
        h_fsvar=hvar_train.mean(dim=-4,keepdim=True)+hmean_train.var(dim=-4,keepdim=True)
        h_fsstd=(h_fsvar+eps)**0.5
        
        
        h_fsmean=h_fsmean.repeat(N,1,1,1)
        h_fsstd=h_fsstd.repeat(N,1,1,1)
        
        h_gen=h_fsmean+h_fsstd*h_fsstd.data.clone().normal_()
        gen=self.decode(h_gen)
        return gen
