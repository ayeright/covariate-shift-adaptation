# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 16:18:13 2018

@author: Pichau
"""

import numpy as np
import torch
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import collections

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PCIF(object):
    """
    Probabilistic Classifier Importance Fitting.
    
    Trains a probabilistic classifier to distinguish between samples from 
    training and test distributions. Then given a feature vector x, we can use 
    the trained classifier along with Bayes' rule to estimate the probability 
    density ratio w(x) as follows:
        
    w(x) = n_tr * p(test|x) / n_te * p(train|x),
    
    where n_tr and n_te are the number of training and test samples used to
    fit the model respectively, and p(train|x) and p(test|x) are the 
    probabilities that x was sampled from the training and test distributions
    respectively, as predicted by the trained classifier.
    
    Attributes
    ----------
    
    n_tr_: integer
        Number of samples from training distribution used to fit the model.
           
    n_te_: integer
        Number of samples from test distribution used to fit the model.
           
    estimator_: estimator with scikit-learn interface
        Fitted probabilistic classifier.
    """

    def __init__(self):
        
        # attributes
        self.n_tr_ = None
        self.n_te_ = None
        self.estimator_ = None
        
    
    def fit(self,
            estimator,
            X_tr,
            X_te):
        """
        Fits a probabilistic classifier to the input training and test data
        to predict p(test|x).
        
        - If an estimator with the scikit-learn interface is provided,
        this estimator is fit to the data.
        
        - If scikit-learn GridSearchCV or RandomizedSearchCV object provided,
        model selection is run and the best estimator is subsequently fit to 
        all the data.
        
        Parameters
        ----------
        
        estimator: estimator or sklearn.model_selection.GridSearchCV/RandomizedSearchCV
            If estimator, assumed to implement the scikit-learn estimator interface.
        
        X_tr: numpy array
            Input data from training distribution, where each row is a feature vector.
            
        X_te: numpy array
            Input data from test distribution, where each row is a feature vector.
        """
        
        # construct the target (1 if test, 0 if train)
        self.n_tr_ = X_tr.shape[0]
        self.n_te_ = X_te.shape[0]
        n = self.n_tr_ + self.n_te_
        y = np.concatenate((np.zeros(self.n_tr_), np.ones(self.n_te_)))
    
        # stack and shuffle features and target
        X = np.vstack((X_tr, X_te))
        i_shuffle = np.random.choice(n, n, replace=False)
        X = X[i_shuffle]
        y = y[i_shuffle]
        
        # fit estimator
        if isinstance(estimator, GridSearchCV) or isinstance(estimator, RandomizedSearchCV):
            print("Running model selection...")
            estimator.refit = True
            estimator.fit(X, y)
            print("Done!")
            print("Best score = {}".format(estimator.best_score_))
            print("Using best estimator.") 
            self.estimator_ = estimator.best_estimator_
        else:
            print("Fitting estimator...")
            self.estimator_ = estimator.fit(X, y)
            print("Done!")
               
    
    def predict(self,
                X):
        """
        Estimates importance weights for input data.
        
        For each feature vector x, the trained probabilistic classifier 
        is used to estimate the probability density ratio
            
        w(x) = n_tr * p(test|x) / n_te * p(train|x),
        
        where n_tr and n_te are the number of training and test samples used to
        train the model respectively, and p(train|x) and p(test|x) are the
        probabilities that x was sampled from the training and test distributions
        respectively, as predicted by the trained classifier.
        
        Parameters
        ----------
        
        X: numpy array
            Input data, where each row is a feature vector.
            
        Returns
        -------
        
        w: numpy vector of shape (len(X),)
            Estimated importance weight for each input. 
            w[i] corresponds to importance weight of X[i]
        """
        
        assert self.estimator_ is not None, "Need to run fit method before calling predict!"
               
        p = self.estimator_.predict_proba(X)
        if len(p.shape) == 1:
            w = (self.n_tr_ / self.n_te_) * (p / (1 - p))
        else:
            w = (self.n_tr_ / self.n_te_) * (p[:, 1] / p[:, 0])
            
        return w
    
    
class NNIF(object):
    """
    Neural Network Importance Fitting.
    
    Trains a pytorch neural network to distinguish between samples from 
    training and test distributions. Then given a feature vector x, we can use 
    the trained network along with Bayes' rule to estimate the probability 
    density ratio w(x) as follows:
        
    w(x) = n_tr * p(test|x) / n_te * p(train|x),
    
    where n_tr and n_te are the number of training and test samples used to
    fit the model respectively, and p(train|x) and p(test|x) are the 
    probabilities that x was sampled from the training and test distributions
    respectively, as predicted by the trained network.
    
    Attributes
    ----------
    
    n_tr_: integer
        Number of samples from training distribution used to fit the model.
           
    n_te_: integer
        Number of samples from test distribution used to fit the model.
           
    estimator_: pytorch model
        Fitted neural network.
    """

    def __init__(self):
        
        # attributes
        self.n_tr_ = None
        self.n_te_ = None
        self.estimator_ = None
        
    
    def fit(self,
            estimator,
            X_tr,
            X_te):
        """
        Fits a pytorch neural network to the input training and test data
        to predict p(test|x).
        
        Parameters
        ----------
        
        estimator: pytorch model
        
        X_tr: numpy array
            Input data from training distribution, where each row is a feature vector.
            
        X_te: numpy array
            Input data from test distribution, where each row is a feature vector.
        """
        
        pass
               
    
    def predict(self,
                X):
        """
        Estimates importance weights for input data.
        
        For each feature vector x, the trained neural network 
        is used to estimate the probability density ratio
            
        w(x) = n_tr * p(test|x) / n_te * p(train|x),
        
        where n_tr and n_te are the number of training and test samples used to
        train the model respectively, and p(train|x) and p(test|x) are the
        probabilities that x was sampled from the training and test distributions
        respectively, as predicted by the trained network.
        
        Parameters
        ----------
        
        X: numpy array
            Input data, where each row is a feature vector.
            
        Returns
        -------
        
        w: numpy vector of shape (len(X),)
            Estimated importance weight for each input. 
            w[i] corresponds to importance weight of X[i]
        """
        
        pass

    
class uLSIF(object):
    """
    Unconstrained Least Squares Importance Fitting (uLSIF).
    
    Implementation of uLSIF algorithm as described in 
    Machine Learning in Non-Stationary Environments - Introduction to Covariate Shift Adaption,
    M. Sugiyama and M. Kawanabe, 2012.
    
    Gaussian kernel basis functions are fit to samples from training and test
    distributions to approximate the probability density ratio
    
    w(x) = p_te(x) / p_tr(x),
    
    where p_tr(x) and p_te(x) are the probabilities that the feature vector x
    comes from the training and test distributions respectively. The fitting 
    is done through minimisation of the squared-loss between the model and 
    the true probability density ratio function.
    
    Once fitted the model can be used to estimate the probability density
    ratio, or importance, of any x. 
    
    Parameters
    ----------
    
    n_kernels: integer (default=100)
        Number of Guassian kernels to use in the model.
        
    Attributes
    ----------
    
    C_: torch tensor
        Kernel centres, where each row is a randomly chosen sample from the 
        test distribution.
        
    alpha_: torch tensor
        Coefficients of fitted model.
    
    sigma_: scalar
        Kernel width of fitted model.   
    """

    def __init__(self,
                n_kernels=100):
        
        # parameters
        self.n_kernels = n_kernels
        
        # attributes
        self.C_ = None
        self.alpha_ = None
        self.sigma_ = None
        
    
    def fit(self,
            X_tr,
            X_te,
            sigma,
            lam,
            random_seed=42):
        """
        Fits the model to the input training and test data.
        
        Gaussian kernel basis functions are fit to the data by minimising
        the squared-loss between the model and the true probability density 
        ratio function.
        
        - If scalars provided for both kernel width (sigma) and regularisation 
        strength (lam), the model with these hyperparameters is fit to the data.
        
        - If more than one value provided for either of the hyperparameters, 
        a hyperparameter search is performed via leave-on-out cross-validation
        and the best parameters are used to fit the model.
        
        Parameters
        ----------
        
        X_tr: numpy array
            Input data from training distribution, where each row is a feature vector.
            
        X_te: numpy array
            Input data from test distribution, where each row is a feature vector.
            
        sigma: scalar or iterable
            Gaussian kernel width. If iterable, hyperparameter search will be run.
            
        lam: scalar or iterable
            Regularisation strength. If iterable, hyperparameter search will be run.
            
        random_seed: integer (default=42)
            Numpy random seed.
        """
        
        with torch.no_grad():
        
            np.random.seed(random_seed)
            
            # convert training and test data to torch tensors
            X_tr = torch.from_numpy(X_tr).float().to(DEVICE)
            X_te = torch.from_numpy(X_te).float().to(DEVICE)
            
            # randomly choose kernel centres from X_te without replacement
            n_te = X_te.size(0)
            t = min(self.n_kernels, X_te.size(0)) 
            self.C_ = X_te[np.random.choice(n_te, t, replace=False)] # shape (t, d)
            
            # compute the squared l2-norm of the difference between 
            # every point in X_tr and every point in C,
            # element (l, i) should contain the squared l2-norm
            # between C[l] and X_tr[i]
            print("Computing distance matrix for X_train...")
            D_tr = self.pairwise_euclidean_distance_squared(self.C_, X_tr) # shape (t, n_tr)
        
            # do the same for X_te
            print("Computing distance matrix for X_test...")
            D_te = self.pairwise_euclidean_distance_squared(self.C_, X_te) # shape (t, n_te)        
            
            # check if we need to run a hyperparameter search
            search_sigma = isinstance(sigma, (collections.Sequence, np.ndarray)) and \
                            (len(sigma) > 1)
            search_lam = isinstance(lam, (collections.Sequence, np.ndarray)) and \
                            (len(lam) > 1)
            if search_sigma | search_lam:
                print("Running hyperparameter search...")
                sigma, lam = self.loocv(X_tr, D_tr, X_te, D_te, sigma, lam)
            else:
                if isinstance(sigma, (collections.Sequence, np.ndarray)):
                    sigma = sigma[0]
                if isinstance(lam, (collections.Sequence, np.ndarray)):
                    lam = lam[0]
                    
            print("Computing optimal solution...")    
            X_tr = self.gaussian_kernel(D_tr, sigma)  # shape (t, n_tr)
            X_te = self.gaussian_kernel(D_te, sigma) # shape (t, n_te)
            H, h = self.kernel_arrays(X_tr, X_te) # shapes (t, t) and (t, 1)
            alpha = (H + (lam * torch.eye(t)).to(DEVICE)).inverse().mm(h) # shape (t, 1)
            self.alpha_ = torch.max(torch.zeros(1).to(DEVICE), alpha) # shape (t, 1)
            self.sigma_ = sigma
            print("Done!")
        
    
    def predict(self,
                X):
        """
        Estimates importance weights for input data.
        
        For each feature vector x, uses the fitted model to estimate the 
        probability density ratio
        
        w(x) = p_te(x) / p_tr(x),
        
        where p_tr is the probability density of the training distribution and
        p_te is the probability density of the test distribution.
        
        Parameters
        ----------
        
        X: numpy array
            Input data from training distribution, where each row is a feature vector.
            
        Returns
        -------
        
        w: numpy vector of shape (len(X),)
            Estimated importance weight for each input. 
            w[i] corresponds to importance weight of X[i]
        """
        
        with torch.no_grad():
        
            assert self.alpha_ is not None, "Need to run fit method before calling predict!"
            
            # convert data to torch tensors
            X = torch.from_numpy(X).float().to(DEVICE)
            
            # compute the squared l2-norm of the difference between 
            # every point in X and every point in C,
            # element (l, i) should contain the squared l2-norm
            # between C[l] and X[i]
            D = self.pairwise_euclidean_distance_squared(self.C_, X) # shape (t, n)
            
            # compute gaussian kernel
            X = self.gaussian_kernel(D, self.sigma_)  # shape (t, n_tr)
            
            # compute importance weights
            w = self.alpha_.t().mm(X).squeeze().cpu().numpy() # shape (n_tr,) 
            
        return w
                
    
    def loocv(self,
              X_tr,
              D_tr,
              X_te,
              D_te,
              sigma_range,
              lam_range):
        """
        Runs hyperprameter search via leave-one-out cross-validation (LOOCV).
        
        Computes LOOCV squared-loss for every combination of the Guassian kernel 
        width and regularisation strength and returns the parameters which 
        correspond to the smallest loss. 
        
        Parameters
        ----------
        
        X_tr: torch tensor
            Input data from training distribution, where each row is a feature vector.
            
        D_tr: torch tensor
            Squared l2-norm of the difference between every kernel centre
            and every row in X_tr.
            Element (l, i) should contain the squared l2-norm between 
            the l-th kernel centre and X_tr[i]
            
        X_te: torch tensor
            Input data from test distribution, where each row is a feature vector.
            
        D_te: torch tensor
            Squared l2-norm of the difference between every kernel centre
            and every point in X_te.
            Element (l, i) should contain the squared l2-norm between 
            the l-th kernel centre and X_te[i]
            
        sigma_range: scalar or iterable
            Guassian kernel width. If scalar will be converted to list.
            
        lam_range: scalar or iterable
            Regularisation strength. If scalar will be converted to list.
            
        Returns
        -------
        
        sigma_hat: scalar
            Guassian kernel width corresponding to lowest LOOCV loss.
            
        lam_hat: scalar
            Regularisation strength corresponding to lowest LOOCV loss.            
        """
        
        with torch.no_grad():
        
            # make sure hyperparameter ranges are iterables
            if not isinstance(sigma_range, (collections.Sequence, np.ndarray)):
                sigma_range = [sigma_range]
            if not isinstance(lam_range, (collections.Sequence, np.ndarray)):
                lam_range = [lam_range]
                
            # define some useful variables
            n_tr, d = X_tr.size()
            n_te = X_te.size(0)
            n = min(n_tr, n_te)
            t = min(self.n_kernels, n_te) 
            ones_t = torch.ones((t, 1), device=DEVICE)
            ones_n = torch.ones((n, 1), device=DEVICE)
            diag_n_idx = torch.cat((torch.range(0, n-1).view(1, -1).long(), torch.range(0, n-1).view(1, -1).long()))
            losses = np.zeros((len(sigma_range), len(lam_range)))
            
            # for each candidate of Gaussian kernel width...
            for sigma_idx, sigma in enumerate(sigma_range):
                
                # apply the Guassian kernel function to the elements of D_tr and D_te
                # reuse variables X_tr and X_te as we won't need the originals again
                X_tr = self.gaussian_kernel(D_tr, sigma) # shape (t, n_tr)
                X_te = self.gaussian_kernel(D_te, sigma)  # shape (t, n_te)
                
                # compute kernel arrays
                H, h = self.kernel_arrays(X_tr, X_te) # shapes (t, t) and (t, 1)
                
                # for what follows X_tr and X_te must have the same shape,
                # so choose n points randomly from each
                X_tr = X_tr[:, np.random.choice(n_tr, n, replace=False)] # shape (t, n)
                X_te = X_te[:, np.random.choice(n_te, n, replace=False)] # shape (t, n)
                
                # for each candidate of regularisation parameter...
                for lam_idx, lam in enumerate(lam_range):
                    
                    # compute the t x t matrix B
                    B = H + torch.eye(t, device=DEVICE) * (lam * (n_tr - 1)) / n_tr # shape (t, t)
                    
                    # compute the t x n matrix B_0           
                    B_inv = B.inverse() # shape (t, t)
                    B_inv_X_tr = B_inv.mm(X_tr) # shape (t, n)
                    diag_num = h.t().mm(B_inv_X_tr).squeeze() # shape (n,)
                    diag_denom = (n_tr * ones_n.t() - ones_t.t().mm(X_tr * B_inv_X_tr)).squeeze() # shape (n,) 
                    diag_sparse = torch.sparse.FloatTensor(diag_n_idx, (diag_num / diag_denom).cpu(), torch.Size([n, n])).to(DEVICE) # sparse (n, n)
                    B_0 = B_inv.mm(h).mm(ones_n.t()) + (diag_sparse.t().mm(B_inv_X_tr.t())).t() # shape (t, n)
                    
                    # compute B_1
                    diag_num = ones_t.t().mm(X_te * B_inv_X_tr).squeeze() # shape (n,)
                    diag_sparse = torch.sparse.FloatTensor(diag_n_idx, (diag_num / diag_denom).cpu(), torch.Size([n, n])).to(DEVICE) # sparse (n, n)
                    B_1 = B_inv.mm(X_te) + (diag_sparse.t().mm(B_inv_X_tr.t())).t() # shape (t, n)
        
                    # compute B_2
                    B_2 = ((n_tr - 1) / (n_tr * (n_te - 1))) * (n_te * B_0 - B_1) # shape (t, n)              
                    B_2 = torch.max(torch.zeros(1).to(DEVICE), B_2) # shape (t, n) 
                    
                    # compute leave-one-out CV loss
                    loss_1 = ((X_tr * B_2).t().mm(ones_t).pow(2).sum() / (2 * n)).item()
                    loss_2 = (ones_t.t().mm(X_te * B_2).mm(ones_n) / n).item()          
                    losses[sigma_idx, lam_idx] = loss_1 - loss_2        
                    print("sigma = {:0.5f}, lambda = {:0.5f}, loss = {:0.5f}".format(
                            sigma, lam, losses[sigma_idx, lam_idx]))
                    
            # get best hyperparameters        
            sigma_idx, lam_idx = np.unravel_index(np.argmin(losses), losses.shape)
            sigma_hat, lam_hat = sigma_range[sigma_idx], lam_range[lam_idx]   
            print("\nbest loss = {:0.5f} for sigma = {:0.5f} and lambda = {:0.5f}".format(
                    losses[sigma_idx, lam_idx], sigma_hat, lam_hat))
        
        return sigma_hat, lam_hat
    
    
    def pairwise_euclidean_distance_squared(self,
                                            A, 
                                            B):
        
        """
        Computes pairwise squared l2-norm between rows of two matrices.
        
        Computes the squared l2-norm of the difference between every row in C
        with every row in X. Element (l, i) of output should contain the 
        squared l2-norm between C[l] and X[i].
        
        - Number of columns must be same in both matrices.
        
        Parameters
        ----------
        
        A: torch tensor
            
        B: torch tensor
            
        Returns
        -------
        
        D: torch tensor
            Has shape (len(A), len(B)), where D[i, j] is equal to the 
            squared l2-norm between A[i] and B[j].          
        """
        
        # define some useful variables
        array_size_limit = 1e8
        t, d = A.size()
        n = B.size(0)
        
        # reshape the matrices for broadcasting
        A = A.view(t, d, 1)
        B = B.t()
        
        if n * d * t <= array_size_limit:
            # do in one go
            D = (A - B).pow(2).sum(dim=1)
        else:
            # do in chunks to avoid memory error
            D = torch.zeros((t, n)).to(DEVICE)
            chunk_size = array_size_limit / (d * t)
            num_chunks = int(np.ceil(n / chunk_size))
            for chunk in range(num_chunks):
                first_col = int(chunk * chunk_size)
                last_col = min(n, int((chunk + 1) * chunk_size))
                D[:, first_col:last_col] = (A - B[:, first_col:last_col]).pow(2).sum(dim=1) 
                
        return D 
    
    
    def gaussian_kernel(self, 
                        D, 
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
    
    
    def kernel_arrays(self,
                      X_tr,
                      X_te):
        """
        Computes kernel matrix H and vector h from algorithm.
        
        H[l, l'] is equal to the sum over i=1:n_tr of 
        
        exp(-(||x_i - c_l|| ^ 2 + -||x_i - c_l'|| ^ 2) / (2 * (sigma ^ 2)))
        
        where n_tr is the number of samples from the training distribution,
        x_i is the i-th sample from the training distribution and 
        c_l is the l-th kernel centre.
        
        h[l] is equal to the sum over i=1:n_te of 
        
        exp(-(||x_i - c_l|| ^ 2) / (2 * (sigma ^ 2)))
        
        where n_te is the number of samples from the test distribution,
        x_i is the i-th sample from the test distribution and 
        c_l is the l-th kernel centre.
        
        Parameters
        ----------
        
        X_tr: torch tensor
            X_tr[l, i] is equal to the Gaussian kernel of the squared l2-norm
            of the difference between the l-th kernel centre and the
            i-th sample from the training distribution:
            
                exp(-(||x_i - c_l|| ^ 2) / (2 * (sigma ^ 2))).
            
        X_te: torch tensor
            X_te[l, i] is equal to the Gaussian kernel of the squared l2-norm
            of the difference between the l-th kernel centre and the
            i-th sample from the test distribution:
                
                exp(-(||x_i - c_l|| ^ 2) / (2 * (sigma ^ 2))).
            
        Returns
        -------
        
        H: torch tensor
            H[l, l'] is equal to the sum over i=1:X_tr.size(1) 
            of X_tr[l, i] * X_tr[l', i] / X_tr.size(1)
            
        h: torch tensor
            h[l] is equal to the sum over i=1:X_te.size(1) 
            of X_te[l, i] / X_te.size(1)          
        """
    
        # compute H
        n_tr = X_tr.size(1)
        H = X_tr.mm(X_tr.t()) / n_tr # shape (t, t)
            
        # compute h
        n_te = X_te.size(1)
        h = X_te.sum(dim=1, keepdim=True) / n_te # shape (t, 1)
        
        return H, h
