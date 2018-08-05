# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 08:40:17 2018

@author: Pichau
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats
import numpy as np
import pandas as pd
import copy
from os.path import join
from datetime import datetime
import sys

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model,
               criterion,
               optimiser,
               X_train,
               y_train,
               X_val=None,
               y_val=None,
               batch_size=32,
               num_epochs=100,
               patience=10,
               verbose=False):
    """
    Trains a pytorch model.
    
    For the particular model, uses the given optimiser to minimise the
    criterion on the given training data.
    
    - Training is done in batches, with the number of training examples in
    each batch specified by the batch size.
    
    - If no validation data is provided, the model is trained for the specified 
    number of epochs.
    
    - If validation data is provided, the validation criterion is computed after
    every epoch and training is stopped when (if) no improvement to the 
    criterion is observed after a set number of epochs, defined by the patience.
    
    Parameters
    ----------
    
    model: torch.nn.Module
        A pytorch model which inherits from the nn.Module class.
        
    criterion: valid loss function from torch.nn
        The loss between the model outputs and the true labels 
        which the model is trained to minimise.
        e.g. nn.CrossEntropyLoss()
        
    optimiser: valid optimiser from torch.optim
        Optimiser which attempts to minimise the criterion on the training data
        by updating the parameters specified by the optimisers "params" parameter.
        e.g. optim.SGD(model.parameters())
        
    X_train: torch tensor
        Training features with shape (# training examples, # features).
        
    y_train: torch tensor
        Training labels where y_train[i] is the label of X_train[i].
    
    X_val: None or torch tensor
        Validation features with shape (# validation examples, # features).
        default=None
    
    y_val: None or torch tensor
        Validation labels  where y_val[i] is the label of X_val[i].
        default=None
    
    batch_size: integer
        Number of examples in each batch during training and validation.
        default=32
    
    num_epochs: integer
        Maximum number of passes through full training set
        (possibly less if validation data provided and early stopping triggered).
        default=100
    
    patience: integer
        Only used if validation data provided.
        Number of epochs to wait without improvement to the best validation loss
        before training is stopped.
        default=10
    
    verbose: boolean
        Whether or no to print training and validation loss after every epoch.
        default=False
        
    Returns
    -------
    
    model: torch.nn.Module
        Same model as input but with trained parameters.
        If validation data provided, return model corresponding to best validation loss,
        else returns model after final training epoch
    
    history: dictionary
        Dictionary with following fields:
            train_loss: list with training loss after every epoch
            val_loss: list with validation loss after every epoch (only if validation data provided)
    """
    
    # compute initial training loss
    train_loss = evaluate_model(model,
                                X_train,
                                y_train, 
                                criterion,
                                batch_size)["loss"]
    history = {"train_loss": [train_loss]}
    
    if X_val is not None:
        # compute initial validation loss
        val_loss = evaluate_model(model,
                                  X_val,
                                  y_val,
                                  criterion,
                                  batch_size)["loss"]
        history["val_loss"] = [val_loss]
        best_loss = val_loss
        epochs_since_improvement = 0
        
        # make a copy of model weights
        best_model_weights = copy.deepcopy(model.state_dict())
    
    # create vector we will use to shuffle training data at the beginning of every epoch    
    num_train = X_train.size(0)
    i_shuffle = np.random.choice(num_train, num_train, replace=False)     

    # train in epochs
    num_epochs = int(num_epochs)
    for epoch in range(num_epochs):
        if verbose:      
            print("Epoch {}/{}".format(epoch + 1, num_epochs))
            
        # train with one pass through training set (with shuffling)
        i_shuffle = i_shuffle[np.random.choice(num_train, num_train, replace=False)]
        training_epoch(model,
                      criterion,
                      optimiser,
                      X_train,
                      y_train,
                      batch_size,
                      i_shuffle)
                
        # compute training loss
        train_loss = evaluate_model(model,
                                    X_train,
                                    y_train, 
                                    criterion,
                                    batch_size)["loss"]
        history["train_loss"].append(train_loss)
        if verbose:
            print("Train loss: {:.4f}".format(train_loss))
        
        if X_val is not None:
            # compute validation loss
            val_loss = evaluate_model(model,
                                     X_val,
                                     y_val,
                                     criterion,
                                     batch_size)["loss"]
            history["val_loss"].append(val_loss)
            if verbose:
                print("Val loss: {:.4f}".format(val_loss))
        
            # if validation loss decreased, record best loss and make a copy 
            # of model weights             
            if val_loss < best_loss:    
                best_loss = val_loss
                epochs_since_improvement = 0
                best_model_weights = copy.deepcopy(model.state_dict())              
            else:
                epochs_since_improvement += 1

            # stop training early?
            if epochs_since_improvement >= patience:
                # load best model weights and stop training
                model.load_state_dict(best_model_weights)
                return model, history 
            
        if verbose:
            print("-" * 20)
                
    return model, history     


def training_epoch(model,
                  criterion,
                  optimiser,
                  X,
                  y,
                  batch_size=32,
                  i_order=None):
    """
    Executes one training epoch.
    
    For the particular model, updates parameters by performing one pass
    through the training set, using the specified optimiser to minimise
    the specified critierion on the given training data.
    
    - Training is done in batches, with the number of training examples in
    each batch specified by the batch size.
    
    - Training examples can effectively be shuffled by specifying a random order 
    in which the examples are to appear in the batches.
    
    Parameters
    ----------
    
    model: torch.nn.Module
        A pytorch model which inherits from the nn.Module class.
        
    criterion: valid loss function from torch.nn
        The loss between the model predictions and the true labels 
        which the model is trained to minimise.
        e.g. nn.CrossEntropyLoss()
        
    optimiser: valid optimiser from torch.optim
        Optimiser which attempts to minimise the criterion on the training data
        by updating the parameters specified by the optimisers "params" parameter.
        e.g. optim.SGD(model.parameters())
        
    X: torch tensor
        Training features with shape (# training examples, # features).
        
    y: torch tensor
        Training labels where y[i] is the label of X[i].
    
    batch_size: integer
        Number of examples in each batch during training.
        default=32
    
    i_order: None or numpy vector
        If not None, needs to contain indices of training data rows,
        which specifies the order in which the examples are to appear 
        in the batches during training
        default=None
          
    Returns
    -------
    
    model: torch.nn.Module
        Same model as input but with trained parameters.
    """
    
    # we are in training mode
    model.train()
    
    # train in batches
    num_train = X.size(0)
    num_batches = int(np.ceil(num_train / batch_size))    
    for batch in range(num_batches):

        # zero the parameter gradients
        optimiser.zero_grad()

        # get data in this batch
        i_first = batch_size * batch
        i_last = batch_size * (batch + 1)  
        i_last = min(i_last, num_train)        
        if i_order is None:
            X_batch = X[i_first:i_last]
            y_batch = y[i_first:i_last]
        else:
            X_batch = X[i_order[i_first:i_last]]
            y_batch = y[i_order[i_first:i_last]]
            
        # forward pass
        out = model(X_batch).squeeze()

        # compute loss
        loss = criterion(out, y_batch)
    
        # backward pass + optimise
        loss.backward()
        optimiser.step()
        
    return model
    

def apply_model(model,
                X,
                batch_size=32):
    """
    Applies a pytorch model in batches.
    
    Uses the given model to predict labels for given features.
    
    - The features must correspond to the same features which were used
    to train the model.
    
    - The model is applied in batches, with the number of examples in
    each batch specified by the batch size.
    
    Parameters
    ----------
    
    model: torch.nn.Module
        A pytorch model which inherits from the nn.Module class.
        
    X: torch tensor
        Features with shape (# examples, # features).
        Must correspond to the same features which were used to train the model.

    batch_size: integer
        Number of examples in each batch during prediction.
        default=32
          
    Returns
    -------
    
    y_pred: pytorch tensor
        Model predictions with shape (# examples,).
    """
    
    # we are in inference mode
    model.eval()
    
    # create tensor for storing outputs
    num_points = X.size(0)
    output_dim = list(model._modules.items())[-1][1].out_features
    if output_dim == 1:
        out = torch.zeros(num_points).to(X.device)
    else:
        out = torch.zeros(num_points, output_dim).to(X.device)
    
    # apply in batches   
    num_batches = int(np.ceil(num_points / batch_size))      
    for batch in range(num_batches):

        # get data in batch
        i_first = batch_size * batch
        i_last = batch_size * (batch + 1)  
        i_last = max(i_last, num_points)                
        X_batch = X[i_first:i_last]

        # predict
        with torch.no_grad():           
            out[i_first:i_last] = model(X_batch).squeeze()

    return out


def evaluate_model(model, 
                  X,
                  y,
                  criterion=None,
                  batch_size=32,
                  metrics=None):
    """
    Evaluates a pytorch model.
    
    Uses the given model to predict labels for given features,
    and computes overall loss between predictions and given labels
    using the specified criterion. Also computes any additional specified
    metrics.
    
    - The features must correspond to the same features which were used
    to train the model.
    
    - The model is applied in batches, with the number of examples in
    each batch specified by the batch size.
    
    Parameters
    ----------
    
    model: torch.nn.Module
        A pytorch model which inherits from the nn.Module class.
        
    X: torch tensor
        Training features with shape (# examples, # features).
        
    y: torch tensor
        Training labels where y[i] is the label of X[i].
        
    criterion: None or valid loss function from torch.nn (default=None)
        The loss between the model predictions and the true labels 
        which the model is trained to minimise.
        e.g. nn.CrossEntropyLoss()
    
    batch_size: integer (default=32)
        Number of examples in each batch during prediction.
        
    metrics: None or list of tuples (name, callable) (default=None)
        Metrics to compute between model outputs and true labels.
          
    Returns
    -------
    
    scores: dictionary
        Keys are metric names and values are the corresponding scores.
    """
    
    # we are in inference mode
    model.eval()
    
    # save scores to dictionary
    scores = {}
    
    # apply model
    out = apply_model(model, X, batch_size)
        
    if criterion is not None:
        # compute the loss
        scores["loss"] = criterion(out, y).item()
    
    if metrics is not None:
        # compute other metrics
        for name, f in metrics:  
            scores[name] = f(y, out)

    return scores


def get_dout_dx(model,
                X,
                batch_size=32,
                proba=False):
    
    # we are in inference mode
    model.eval()
    
    # compute gradients in batches
    dout_dx = torch.zeros_like(X, dtype=torch.float, device=X.device)
    num_points = X.size(0)
    num_batches = int(np.ceil(num_points / batch_size))    
    for batch in range(num_batches):
        
        # zero gradients
        model.zero_grad()

        # get data in this batch
        i_first = batch_size * batch
        i_last = batch_size * (batch + 1)  
        i_last = min(i_last, num_points)        
        X_batch = torch.autograd.Variable(X[i_first:i_last], requires_grad=True)

        # forward pass
        out = model(X_batch).squeeze()
        
        # compute dout_dx
        out.backward(torch.ones_like(out, dtype=torch.float, device=X.device))
        dout_dx[i_first:i_last] = X_batch.grad
        
    return dout_dx


class FeedForwardNet(nn.Module):
    
    """
    Fully-connected feed forward neural network.
    
    Builds on the standard nn.Module class by adding functionality
    to easily set network hyperparameters and include features such as
    batch normalisation and categorical embeddings.
    
    As well as the initialisation method which constructs the network,
    there are two more methods for initialising layer weights
    and computing the forward pass.
    
    - If embedding groups provided, categorical variables are embedded
    as described in "Entity Embeddings of Categorical Variables" 
    (https://arxiv.org/abs/1604.06737).
    
    Parameters
    ----------
    
    input_dim: integer
        Number of input features.
        
    output_dim: integer
        Number of outputs.
        
    params: dictionary
        Network parameters. 
        See below for available parameters and default values.

    embedding_groups: None or list of lists
        If not None, each list specifies the indices of dummy columns
        corresponding to a single categorical variable.
        
    Attributes
    ----------
    
    total_embedding_dim_: integer
        Sum of the dimensions of all embedding layers.
    
    Network Layers
    --------------
    
    embeddings: torch.nn.ModuleList or None
        Embedding layers.
                
    hidden_layers: torch.nn.ModuleList
         Linear hidden layers.
    
    batch_norm_layers: torch.nn.ModuleList or None
        Batch normalisation layers.
    
    output_layer: torch.nn.Linear
        Linear output layer.    
    """
    
    def __init__(self,
                input_dim,  
                output_dim,
                params,
                embedding_groups=None):
        
        super(FeedForwardNet, self).__init__()
        
        self.embedding_groups = embedding_groups
        
        # define default network parameters
        self.params = {}
        self.params["num_hidden"] = 1
        self.params["hidden_dim"] = [input_dim]
        self.params["embed_dummies"] = False
        self.params["embedding_dim"] = [1]
        self.params["batch_norm"] = True
        self.params["weight_init"] = nn.init.xavier_uniform_
        self.params["hidden_activation"] = F.relu
        self.params["output_activation"] = None
   
        # replace provided parameters
        for key, value in params.items():
            self.params[key] = value
            
        # make sure integer parameters are correct type
        input_dim = int(input_dim)
        output_dim = int(output_dim)
        self.params["num_hidden"] = int(self.params["num_hidden"])
                
        # if params["hidden_dim"] scalar, convert to list with length equal
        # to the number of hidden dimensions
        if np.isscalar(self.params["hidden_dim"]):
            self.params["hidden_dim"] = [self.params["hidden_dim"]] * self.params["num_hidden"]
        self.params["hidden_dim"] = [int(x) for x in self.params["hidden_dim"]]
                
        if embedding_groups is not None:
        
            # if params["embedding_dim"] scalar, convert to list with length equal
            # to the number of groups of dummy variables that will be embedded
            if np.isscalar(self.params["embedding_dim"]):
                self.params["embedding_dim"] = [self.params["embedding_dim"]] * len(embedding_groups)
            self.params["embedding_dim"] = [int(x) for x in self.params["embedding_dim"]]
            
            # construct embedding for each specified group of dummy variables
            self.embeddings = nn.ModuleList()
            total_embedding_dim = 0
            total_dummies = 0
            for i, dummy_indices in enumerate(embedding_groups):
                num_dummies = len(dummy_indices)
                self.embeddings.append(nn.Embedding(num_dummies,
                                                   self.params["embedding_dim"][i]))
                total_dummies += num_dummies
                total_embedding_dim += self.params["embedding_dim"][i]                
                self.init_layer(self.embeddings[-1])
            
            # input dimension to first hidden layer is 
            # number of numeric variables + total embedding dimension
            input_dim = input_dim - total_dummies + total_embedding_dim
            
            # save total embedding dimension (we will need it for the forward pass)
            self.total_embedding_dim_ = total_embedding_dim
            
        else:
            self.embeddings = None
            self.total_embedding_dim_ = 0
        
        # define module lists for storing hidden layers
        self.hidden_layers = nn.ModuleList()
        if self.params["batch_norm"]:
            self.batch_norm_layers = nn.ModuleList()
        else:
            self.batch_norm_layers = None
        
        # construct hidden layers
        for i in range(self.params["num_hidden"]):
            
            self.hidden_layers.append(nn.Linear(input_dim, 
                                                self.params["hidden_dim"][i]))
            self.init_layer(self.hidden_layers[-1])
                
            if self.params["batch_norm"]:
                self.batch_norm_layers.append(nn.BatchNorm1d(self.params["hidden_dim"][i]))
                
            input_dim = self.params["hidden_dim"][i]
                
        # construct output layer
        self.output_layer = nn.Linear(input_dim, output_dim)
        self.init_layer(self.output_layer)
        
        
    def init_layer(self, layer):
        """
        Initialises network layer weights.
        
        Initialises the weights of the specified layer using the strategy 
        specified in self.params["weight_init"].
        
        - Available initialisation strategies are "xavier".
        
        - Default is uniform.
        
        Parameters
        ----------
        
        layer: torch layer
            Layer whose weights we want to initialise.        
        """
        
        return self.params["weight_init"](layer.weight)
            
                
    def forward(self, X):
        """
        Network forward pass.
        
        Computes the network output by performing the forward pass.
        
        - The number of features in the input data must be the same
        as the network input dimension
        
        Parameters
        ----------
        
        X: torch tensor
            Features with shape (# examples, # features).
            
        Returns
        -------
        
        out: torch tensor
            Output of final network layer.            
        """
        
        if self.embedding_groups is not None:
            # create tensor for storing all embeddings
            embeds = torch.zeros((X.size(0), self.total_embedding_dim_), 
                                 device=X.device)
            
            # loop through each group of dummy variables
            first_col = 0
            last_col = self.params["embedding_dim"][0]
            all_dummy_indices = []
            for i, dummy_indices in enumerate(self.embedding_groups):
                # get embeddings for this group of dummies
                embed_indices = torch.argmax(X[:, dummy_indices], 1)
                embeds[:, first_col:last_col] = self.embeddings[i](embed_indices)           
                first_col += self.params["embedding_dim"][i]
                last_col += self.params["embedding_dim"][i]
                all_dummy_indices += dummy_indices
                
            # concatenate numeric variables with embeddings
            numeric_indices = list(set(range(X.size(1))) - set(all_dummy_indices))
            X = torch.cat((X[:, numeric_indices], embeds), 1)
        
        # loop through hidden layers
        for i, layer in enumerate(self.hidden_layers):
            
            X = layer(X)
            
            # batch norm
            if self.params["batch_norm"]:
                X = self.batch_norm_layers[i](X)
            
            # activation
            if self.params["hidden_activation"] is not None:
                X = self.params["hidden_activation"](X)
        
        # output layer
        out = self.output_layer(X).squeeze()
        if self.params["output_activation"] is not None:
            out = self.params["output_activation"](out)
                   
        return out


class NNRandomSearch(object):
    """
    Neural network hyperparameter random search.
    
    Runs random hyperparameter search of neural network architecture and
    training parameters, where values are sampled from pre-defined
    distributions.
    
    Parameters
    ----------
    
    prediction_type: string, one of "binary_classification", "multi_classification" or "regression"
        Specifies the type of task.
        
    param_dists:  dictionary
        The dictionary key specifies the hyperparameter and the value
        specifies the distribution from which to draw samples. 
        The value is a tuple with  two elements.
        The first element defines the distribution, which can be a list
        or a any distribution from scipy.stats.
        The second element is a string which defines how the distribution 
        is sampled. Options are "set" (choose randomly from set),
        "uniform" (sample uniformly) and "exp" (sample uniformly from log
        domain before exponentiating).
        Examples can be found below for default distributions.
        
    criterion: valid loss function from torch.nn (default=None)
        The loss between the model predictions and the true labels 
        which the model is trained to minimise.
        e.g. nn.CrossEntropyLoss().
        
    embedding_groups: None or list of lists (default=None)
        If not None, each list specifies the indices of dummy columns
        corresponding to a single categorical variable.
        
    metrics: None or list of strings (default=None)
        Metrics to compute between model predictions and true labels.
        Binary classification options are "accuracy" and "auc".
        
    results_dir: string (default="")    
        Path to directory for saving results of hyperparameter search.
        
    refit: boolean or string (default=None)
        Refit an estimator using the best found parameters on the whole dataset.
                
    device: torch device (default=DEVICE)
        Device on which to run computation (CPU or GPU).
        
    Attributes
    ----------
    
    results_
    
    best_estimator_ 
    
    best_score_ 
    
    best_params_ 
    """
        
    def __init__(self,
                param_dists,
                output_dim=1,
                output_activation=None,
                criterion=nn.MSELoss(),
                embedding_groups=None,
                metrics=None,
                results_dir=None,
                refit=False):
        
        # define default parameter distributions
        self.param_dists = {}
        self.param_dists["num_hidden"] = ([1], "set")
        self.param_dists["hidden_dim"] = (stats.randint(10, 21), "uniform")
        self.param_dists["embed_dummies"] = ([False], "set")
        self.param_dists["embedding_dim"] = ([1], "set")
        self.param_dists["batch_norm"] = ([True], "set")
        self.param_dists["weight_init"] = ([nn.init.xavier_uniform_], "set")
        self.param_dists["hidden_activation"] = ([F.relu], "set")
        self.param_dists["optimiser"] = (["adam"], "set")
        self.param_dists["lr"] = (stats.uniform(np.log(0.0001), np.log(0.01) - np.log(0.0001)), "exp")
        
        # replace with provided distributions
        for key, value in param_dists.items():
            self.param_dists[key] = value
        
        # other parameters
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.criterion = criterion
        self.embedding_groups = embedding_groups
        self.metrics = metrics
        self.results_dir = results_dir
        self.refit = refit
        
        # attributes
        if isinstance(refit, tuple) and (len(refit) == 2):
            self.refit_metric_ = refit[0]
            self.refit_max_or_min_ = refit[1]
        else:
            self.refit_metric_ = "loss"
            self.refit_max_or_min_ = "min"
        
        self.results_ = pd.DataFrame()         
        self.best_estimator_ = None     
        self.best_score_ = None      
        self.best_params_ = None 
        
    
    def fit(self,
           X_train,
           y_train,
           X_val,
           y_val,
           num_experiments,
           num_trials,
           batch_size=32,
           num_epochs=100,
           patience=10,
           verbose=True):
        """
        Runs the random hyperparameter search.
        
        Parameters
        ----------
        
        X_train: torch tensor
            Training features with shape (# training examples, # features).
        
        y_train: torch tensor
            Training labels where y_train[i] is the label of X_train[i].
        
        X_val: torch tensor
            Validation features with shape (# validation examples, # features).
        
        y_val: torch tensor
            Validation labels where y_val[i] is the label of X_val[i].
            
        num_experiments: integer
            Number of experiments to run. 
            Each experiment can contain several trials, and the results of each
            experiment will be saved to a separate file.
            
        num_trials: integer
            Number of neural networks to train in each experiment.
            
        batch_size: integer (default=32)
            Number of examples in each batch during training and validation.       
        
        num_epochs: integer (default=100)
            Maximum number of passes through full training set
            (possibly less if validation data provided and early stopping triggered).
                    
        patience: integer (default=10)
            Only used if validation data provided.
            Number of epochs to wait without improvement to the best validation loss
            before training is stopped.
        
        verbose: boolean (default=True)
            Whether or not to print results of each trial.
        """
        
        # use same device as X_train
        device = X_train.device
        
        # get input dimension
        input_dim = X_train.size(1)
        
        # for each experiment...
        for experiment in range(num_experiments):
            
            # create dataframe for storing results of experiment
            experiment_results = pd.DataFrame()
            
            # for each trial...
            for trial in range(num_trials):
                if verbose:
                    print("Running trial {}/{} of experiment {}/{} ...".format(
                                                                   trial + 1, 
                                                                   num_trials,
                                                                   experiment + 1,
                                                                   num_experiments))                
                
                # sample hyperparameters
                params = self.sample_hyperparameters()                
                trial_results = params.copy()
                
                # build model
                params["output_activation"] = self.output_activation
                if params["embed_dummies"]:
                    embedding_groups = self.embedding_groups
                else:
                    embedding_groups = None                    
                model = FeedForwardNet(input_dim,
                                      self.output_dim, 
                                      params,
                                      embedding_groups)
                model.to(device)
                                    
                # define optimiser
                optimiser = self.get_optimiser(model, params)
                
                # train model
                model, history = train_model(model,
                                            self.criterion,
                                            optimiser,
                                            X_train,
                                            y_train,
                                            X_val,
                                            y_val,
                                            batch_size,
                                            num_epochs,
                                            patience,
                                            False)

                # get best epoch
                best_epoch = np.argmin(np.array(history["val_loss"]))
                trial_results["epochs"] = best_epoch
                if verbose:
                    print("Best epoch: {}".format(best_epoch))
                
                # compute training metrics
                metric_scores_train = evaluate_model(model,
                                                     X_train,
                                                     y_train,
                                                     self.criterion,
                                                     batch_size,
                                                     self.metrics)
                
                # compute validation metrics
                metric_scores_val = evaluate_model(model,
                                                   X_val,
                                                   y_val,
                                                   self.criterion,
                                                   batch_size,
                                                   self.metrics)
        
                for metric, score_train in metric_scores_train.items(): 
                    score_val = metric_scores_val[metric]
                    trial_results["train_" + metric] = score_train
                    trial_results["val_" + metric] = score_val
                    if verbose:
                        print("Train {}: {:.4f}, Val {}: {:.4f}".format(metric,
                                                                        score_train,
                                                                        metric,
                                                                        score_val,
                                                                        ))
                
                # append trial results
                self.results_ = self.results_.append(trial_results, ignore_index=True)
                experiment_results = experiment_results.append(trial_results, ignore_index=True)

                if verbose:
                    print("-" * 50)
            
            if self.results_dir is not None:
                # save experiment results
                self.save_results(experiment_results)
                
        if self.refit != False:
            if verbose:
                print("Refitting estimator with the best found parameters...")
            
            # get best parameters
            self.best_params_, self.best_score_ = self.get_best_params()

            # build model
            self.best_params_["output_activation"] = self.output_activation
            if self.best_params_["embed_dummies"]:
                embedding_groups = self.embedding_groups
            else:
                embedding_groups = None                    
            self.best_estimator_ = FeedForwardNet(input_dim,
                                                  self.output_dim, 
                                                  self.best_params_,
                                                  embedding_groups)
            self.best_estimator_.to(device)
                                
            # define optimiser
            optimiser = self.get_optimiser(self.best_estimator_, self.best_params_)
            
            # train model
            self.best_estimator_, _ = train_model(self.best_estimator_,
                                                  self.criterion,
                                                  optimiser,
                                                  torch.cat((X_train, X_val)),
                                                  torch.cat((y_train, y_val)),
                                                  None,
                                                  None,
                                                  batch_size,
                                                  self.best_params_["epochs"],
                                                  None,
                                                  False)
            
                
    def sample_hyperparameters(self):
        """
        Samples hyperparameters from pre-defined distributions.
        
        Returns
        -------
        
        params: dictionary
            Keys are parameter names and values are parameter values. 
        """
        
        params = {}
        for key, (dist, sampling) in self.param_dists.items():            
            
            if sampling == "uniform":
                params[key] = dist.rvs(size=1)[0]                
                
            if sampling == "exp":
                params[key] = np.exp(dist.rvs(size=1)[0])
                                
            if sampling == "set":
                params[key] = dist[stats.randint(0, len(dist)).rvs(size=1)[0]]
                
        return params
    
    
    def get_optimiser(self,
                     model, 
                     params):
        """
        Defines the optimiser using the specified parameters.
        
        Parameters
        ----------
        
        model: torch.nn.Module
            A pytorch model which inherits from the nn.Module class.
            
        params: dictionary
            Keys are parameter names and values are parameter values. 
        
        Returns
        -------
        
        optimiser: optimiser from torch.optim
            Optimiser defined by given parameters.
        """
        
        if params["optimiser"] == "adam":
            optimiser = optim.Adam(model.parameters(), lr=params["lr"])
            
        else:
            optimiser = optim.SGD(model.parameters(), lr=params["lr"])
            
        return optimiser
    
    
    def save_results(self, 
                     results):
        """
        Saves results of hyperparameter search to disk.
        
        Saves Pandas DataFrame to csv with datetime as filename. 
        
        Parameters
        ----------
        
        results: Pandas DataFrame
            Hyperparameter values and training and validation scores for each trial.
        """
        
        time_now = str(datetime.now())
        time_now = time_now.replace(':', '')
        time_now = time_now.replace('.', '')
        results_file = join(self.results_dir, time_now) + ".csv"
        results.to_csv(results_file, index=False)
        
        
    def get_best_params(self):
        
        # get all metric names
        if self.metrics is None: 
            all_metric_names = "loss"
        else:        
            all_metric_names = [x[0] for x in self.metrics]
        
        # get the validation metric we will use to determine best model 
        if self.refit_metric_ in all_metric_names:
            metric = "val_" + self.refit_metric_
        else:
            metric = "val_loss"
            
        # order the results, ascending or descending depending on whether
        # we want the max or min
        if self.refit_max_or_min_ == "min":
            ordered_results = self.results_.sort_values(metric)
        else:
            ordered_results = self.results_.sort_values(metric, ascending=False)
        
        # get the top parameters
        best_params = ordered_results.iloc[0, :].to_dict()
        
        # get corresponding score
        best_score = best_params[metric]
        
        return best_params, best_score
        
        



 
 
"""---------------------------------------ROUGH WORK BELOW---------------------------------"""
      
def to_sparse(X):
    n, m = X.size()
    rows = torch.range(0, n-1).view(-1, 1).long()
    cols = torch.range(0, m-1).view(1, -1).long()
    rows = rows.repeat(1, m).view(1, -1)
    cols = cols.repeat(1, n)
    idx = torch.cat((rows, cols))
    v = X.view(-1, 1).squeeze()
    
    X_typename = torch.typename(X).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, X_typename)
    
    return sparse_tensortype(idx, v.cpu(), X.size())      
        