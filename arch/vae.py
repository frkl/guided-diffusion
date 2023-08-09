
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
    
    def forward(self,v):
        eps=1e-20
        h=self.encoder(v)
        h=torch.fft.fft2(h)
        h=(h-self.m)*torch.exp(self.s.clamp(-20,20))
        h=torch.cat((h.real,h.imag),dim=-3)
        h=self.t(h)
        
        hmean=h[:,::2,:,:]
        hvar=h[:,1::2,:,:]**2
        hstd=h[:,1::2,:,:].abs()
        
        h_gen=hmean+hstd*hstd.data.clone().normal_()
        h_gen=(h_gen+self.m2)*torch.exp(self.s2.clamp(-20,20))
        h_gen=torch.complex(h_gen[:,::2,:,:],h_gen[:,1::2,:,:])
        h_gen=torch.fft.ifft2(h_gen)
        h_gen=h_gen.real
        recon=self.decoder(h_gen)
        
        
        kl=-torch.log(hvar+eps)/2+(hvar+hmean**2)/2-0.5
        kl=kl.sum(dim=-3,keepdim=True)
        
        return recon,kl
    
    def generate(self,N):
        eps=1e-20
        h_gen=torch.Tensor(N,2*self.nh,self.imsz,self.imsz).to(self.m.device).normal_()
        h_gen=(h_gen+self.m2)*torch.exp(self.s2.clamp(-20,20))
        h_gen=torch.complex(h_gen[:,::2,:,:],h_gen[:,1::2,:,:])
        h_gen=torch.fft.ifft2(h_gen)
        h_gen=h_gen.real
        gen=self.decoder(h_gen)
        return gen
