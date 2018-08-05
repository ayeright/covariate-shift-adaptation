# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 20:24:07 2018

@author: Pichau
"""

import numpy as np
import torch


def pairwise_distances_squared(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)
    

def gaussian_kernel(D, 
                    sigma):
    """
    Applies Gaussian kernel element-wise.
    
    Computes exp(-d / (2 * (sigma ^ 2))) for each element d in D.
    
    Parameters
    ----------
    
    D: torch tensor
        
    sigma: scalar
        Gaussian kernel width.
        
    Returns
    -------
    
    torch tensor
        Result of applying Gaussian kernel to D element-wise.          
    """
    
    return (-D / (2 * (sigma ** 2))).exp()  