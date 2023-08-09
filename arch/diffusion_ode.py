
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import random
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

'''
def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
'''

def p_sample_loop(self, shape, return_all_timesteps = False):
    batch, device = shape[0], self.device

    img = torch.randn(shape, device = device)
    imgs = [img]

    x_start = None

    for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
        self_cond = x_start if self.self_condition else None
        img, x_start = self.p_sample(img, t, self_cond)
        imgs.append(img)

    ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

    ret = self.unnormalize(ret)
    return ret



class new(nn.Module):
    def __init__(self,nh=64,imsz=128,T=1000, s=0.008):
        super().__init__()
        
        beta = cosine_beta_schedule(T)
        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        
        self.beta=beta
        self.alpha=alpha
        self.alpha_bar=alpha_bar
        
        self.encoder = Unet(dim = 64,dim_mults = (1, 2, 4, 8),flash_attn = True)
        self.x = nn.Parameter(torch.Tensor(1).fill_(0))
        
        self.T=T
        self.nh=nh
        self.imsz=imsz
        return;
    
    def normalize(self,v):
        return v*2-1
    
    def denormalize(self,v):
        return (v+1)/2
    
    def forward(self,v):
        v=self.normalize(v)
        eps=v.data.clone().normal_()
        T=torch.LongTensor(v.shape[0]).random_(0,self.T)
        x=self.alpha_bar[T].to(v.device).view(-1,1,1,1)**0.5*v+(1-self.alpha_bar[T].view(-1,1,1,1).to(v.device))**0.5*eps
        x=x.float()
        eps_pred = self.encoder(x, T.to(v.device))
        loss=((eps-eps_pred)**2).sum(dim=-3).mean()
        return loss
    
    
    def generate(self,N):
        x=torch.Tensor(N,3,self.imsz,self.imsz).normal_().to(self.x.device)
        for i in range(self.T-1,-1,-1):
            eps=x.data.clone().normal_().to(x.device)
            if i==0:
                eps=eps*0
            
            T=torch.LongTensor(N).fill_(i).to(x.device)
            pred_eps=self.encoder(x,T)
            
            pred_x0=1/self.alpha_bar[i]**0.5 *x - ((1-self.alpha_bar[i])/self.alpha_bar[i])**0.5 *pred_eps
            pred_x0=pred_x0.clamp(-1,1)
            
            w0=(self.alpha_bar[i]/self.alpha[i])**0.5 * self.beta[i] / (1-self.alpha_bar[i])
            w1=(self.alpha[i]-self.alpha_bar[i])/self.alpha[i]**0.5 / (1-self.alpha_bar[i])
            x=w0*pred_x0 + w1*x
            beta=(self.alpha[i]-self.alpha_bar[i])/self.alpha[i]/(1-self.alpha_bar[i]) * self.beta[i]
            x=x+ beta**0.5*eps
            x=x.float()
        
        x=self.denormalize(x)
        return x
