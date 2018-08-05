# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 20:21:09 2018

@author: Pichau
"""

import torch
from torch_general_utils import pairwise_distances_squared

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LFDA(object):
    """Local Fisher Discriminant Analysis"""
    
    
    def __init__(self):
        
        self.T_ = None
    
    
    def fit(self,
            X_tr,
            X_te):
        
        with torch.no_grad():
            
            # convert training and test data to torch tensors
            X_tr = torch.from_numpy(X_tr).float().to(DEVICE)
            X_te = torch.from_numpy(X_te).float().to(DEVICE)
        
            # define some useful variables
            n_tr, n_te = X_tr.size(0), X_te.size(0)
            
            # get the scaling factor for each x in X_tr and X_te,
            # which is the distance to the 7th nearest neighbour
            print("Finding nearest neighbours for X_train...")
            tau_tr, _ = self.kth_nn(X_tr, 7) # shape (n_tr,) 
            print("Finding nearest neighbours for X_test...")
            tau_te, _ = self.kth_nn(X_te, 7) # shape (n_te,) 

            # compute affinity matrices for X_tr and X_te
            print("Computing affinity matrices...")
            A_tr = self.affinity(X_tr, tau_tr) # shape (n_tr, n_tr) 
            A_te = self.affinity(X_te, tau_te) # shape (n_te, n_te)
            
            # compute the local within-class scatter matrix
            print("Computing local within-class scatter...")
            X_tr = X_tr.t() # shape (d, n_tr) 
            X_te = X_te.t() # shape (d, n_te) 
            G_tr = self.affinity_weights(X_tr, A_tr) # shape (d, d) 
            G_te = self.affinity_weights(X_te, A_te) # shape (d, d) 
            S_lw = self.within_class_scatter(G_tr, n_tr, G_te, n_te) # shape (d, d)
            
            # compute the local between-class scatter matrix
            print("Computing local between-class scatter...")
            S_lb = self.between_class_scatter(X_tr, G_tr, n_tr, X_te, G_te, n_te) # shape (d, d)
            
            # compute generalised eigenvalues and eigenvectors
            print("Computing generalised eigenvectors...")
            _, psi = self.generalised_eig(S_lb, S_lw)
            
            # compute orthonormal basis of the eigenvectors
            print("Computing orthonormal basis of eigenvectors...")
            self.T_, _, _ = psi.svd() # shape (d, d)
            print("Done!")
            
            
    def transform(self,
                  X,
                  m=None):
        
        X = torch.from_numpy(X).float().to(DEVICE)
        
        if m is None:
            m = self.T_.size(0)
            
        return X.mm(self.T_[:, :m]).cpu().numpy()
        
        
    
    
    def kth_nn(self,
               X, 
               k): 
        """
        
        """
        
        # compute euclidean distance between every pair of elements in X
        D = pairwise_distances_squared(X)
        
        # get the indices which sort each row in ascending order
        _, idx = D.sort(dim=1)
        
        # get k-th nearest neighbour for each row
        idx = idx[:, k]
        
        # get the corresponding distance
        D = torch.dist(X, X[idx])
        
        return D, idx
        
    
    def affinity(self,
                 X, 
                 tau): 
        """
        
        """
        
        D = pairwise_distances_squared(X)
        tau = tau.view(-1, 1)
        return torch.exp(-D / tau.mm(tau.t()))
    
    
    def affinity_weights(self,
                         X, # shape (d, n)
                         A, # shape (n, n)
                         ):
        """
        
        """
    
        n = A.size(0)
        diag_idx = torch.cat((torch.range(0, n-1).view(1, -1).long(), torch.range(0, n-1).view(1, -1).long()))
        diag_vals = A.mm(torch.ones((n, 1), device=DEVICE)).squeeze()    
        diag_sparse = torch.sparse.FloatTensor(diag_idx, diag_vals.cpu(), torch.Size([n, n])).to(DEVICE) # sparse (n, n)
        return diag_sparse.t().mm(X.t()).t().mm(X.t()) - X.mm(A).mm(X.t()) # shape (d, d)
    
    
    def within_class_scatter(self,
                             G_tr, # shape (d, d)
                             n_tr, 
                             G_te, # shape (d, d) 
                             n_te):
        """
        
        """
        
        return (G_tr / n_tr) + (G_te / n_te)


    def between_class_scatter(self,
                              X_tr, # shape (d, n) 
                              G_tr, # shape (d, d) 
                              n_tr, 
                              X_te, # shape (d, n) 
                              G_te, # shape (d, d) 
                              n_te):
        """
        
        """
        
        n = n_tr + n_te
        ones_tr = torch.ones((n_tr, 1), device=DEVICE)
        ones_te = torch.ones((n_te, 1), device=DEVICE)
        S1 = ((1 / n) - (1 / n_tr)) * G_tr + ((1 / n) - (1 / n_te)) * G_te # shape (d, d)
        S2 = (n_te / n) * X_tr.mm(X_tr.t()) + (n_tr / n) * X_te.mm(X_te.t()) # shape (d, d)
        S3 = -((1 / n) * X_tr.mm(ones_tr).mm(X_te.mm(ones_te).t())) # shape (d, d)
        S4 = -((1 / n) * X_te.mm(ones_te).mm(X_tr.mm(ones_tr).t())) # shape (d, d)
        return S1 + S2 + S3 + S4 # shape (d, d)
    
    
    def generalised_eig(self,
                        A, 
                        B):
        """
        
        """
        
        """
        # compute generalised eigenvalues and eigenvectors
        eta, psi = B.inverse().mm(A).eig(eigenvectors=True)
        eta = eta[:, 0]
        
        # sort them from largest to smallest eigenvalue
        eta, sorted_idx = eta.sort(descending=True)
        psi = psi[:, sorted_idx] # shape (d, d)
        
        return eta, psi
        """
        
        eig_vals, eig_vectors = linalg.eig(A, B, left=False, right=True)
        eig_vals = torch.from_numpy(eig_vals.real).float().to(DEVICE)
        eig_vectors = torch.from_numpy(eig_vectors.real).float().to(DEVICE)
        mask_valid = (~torch.isnan(eig_vals)) & (eig_vals < np.inf) &  (eig_vals > -np.inf)
        eig_vals = eig_vals[mask_valid]
        eig_vectors = eig_vectors[:, mask_valid]
        eig_vals, sorted_idx = eig_vals.sort(descending=True)
        eig_vectors = eig_vectors[:, sorted_idx]
        
        return eig_vals, eig_vectors