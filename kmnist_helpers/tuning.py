# -*- coding: utf-8 -*-
from kmnist_helpers import set_seed, seed
from kmnist_helpers.model_selection import holdoutCV
import random
import numpy as np
import itertools


def RandomSearch(epochs, model, iterations, train_loader, validate_loader):
    '''Performs random search for weight decay and learn rate with holdout CV.
    
    Args:
        epochs (int): number of epochs
        model: pytorch nn.module model
        iterations: number of iterations to find optimal parameters
        train_loader: pytorch dataloader
        validate_loader: pytorch dataloader
    
    Returns:
        highest accuracy, list of optimal weight decay and learn rate
    '''
    max_acc = 0.0
    opt_wd = 0.0
    opt_lr = 0.0
  
    set_seed(seed)

    for i in range(iterations):

        # randomise our hyper parameters
        weight_decay = 0.0
        lr = 0.0 
        
        # Select from ranges of [1e-3 to 1e-7]
        while np.isclose(weight_decay, 0.0):
            power = int((random.random()*10)%6)+3
            weight_decay = (5.*round(random.random()*10./5.) * (1./10.**power))
    
        # Select from ranges of [1e-3 to 1e-6]
        while np.isclose(lr, 0.0):
            power = int((random.random()*10)%5)+3 
            lr = 5.*round(random.random()*10./5.) * (1./10.**power)
    
        print("Weight Decay: ", weight_decay)
        print("Learn Rate: ", lr)
        
        lloss, val_loss, val_acc = holdoutCV(epochs, weight_decay, lr,
                                             model, train_loader, validate_loader)
    
        if val_acc > max_acc:
            max_acc = val_acc
            opt_wd = weight_decay
            opt_lr = lr

    return max_acc, [opt_wd, opt_lr]


def GridSearch(epochs, model, rand_params, train_loader, validate_loader, pseudo=True):
    '''Performs grid search for weight decay and learn rate with holdout CV.
    
    Args:
        epochs (int): number of epochs
        model: pytorch nn.module model
        rand_params: takes in best parameters found from rand_params
        train_loader: pytorch dataloader
        validate_loader: pytorch dataloader
        pseudo (bool): only find contigous combinations
    
    Returns:
        best hyperparameters, livelossplots, losses, accuracies
    '''
    i = 0
    lloss_list, loss_list, acc_list = [], [], []

    if pseudo: # In grid search we may want to only find only the combinations that vary contiguously
        grid = _ParamDomain(rand_params, False)
        grid = np.array(grid)
        grid = grid.transpose()
        print("Total number of side by side searches: ", grid.shape[0])

        for comb in grid:
            wd = comb[0]
            lrt = comb[1]
            #mtm = comb[2]
      
            ll, l, a = holdoutCV(epochs, wd, lrt,
                                 model, train_loader, validate_loader)
            lloss_list.append(ll)
            loss_list.append(l)
            acc_list.append(a)
      
        lloss_list = np.array(lloss_list)
        loss_list = np.array(loss_list)
        acc_list = np.array(acc_list)

        best_comb = np.argmax(acc_list)
  
    else:  
        grid = _ParamDomain(rand_params)
        hyperpara_perm = list(itertools.product(*grid))
        combinations = len(hyperpara_perm)
        print("Total Number of Parameter Combinations that will be carried out: ", combinations)

        for comb in range(combinations):
            ll, l, a = _test_hyperpara(epochs, hyperpara_perm[i+1], model, 
                                       train_loader, validate_loader)
            lloss_list.append(ll)
            loss_list.append(l)
            acc_list.append(a)

        lloss_list = np.array(lloss_list)
        loss_list = np.array(loss_list)
        acc_list = np.array(acc_list)

        best_comb = np.argmax(acc_list)     

    if pseudo: print(" - Pseudo Grid Search - ")
    print("Best Accuracy Achieved: ", np.amax(acc_list))
    print("The combination for the best params: ", best_comb)
      
    return best_comb, lloss_list, loss_list, acc_list


def _ParamDomain(params, range3=True):
    grid_search = []
  
    for param in params:
        count = 0
        number = 1/param
        s = set(str(param))
        isFive = s.issuperset("5")
    
        while (number > 0):
          number = number//10
          count = count + 1
        if not isFive:
          count = count - 1
        num = (1.0/10.0**count)
        
        if range3:
          param_domain = []
          param_domain.append(param - num*1)
          param_domain.append(param)
          param_domain.append(param + num*1)
        else:
          param_domain = []
          param_domain.append(param - num*2)
          param_domain.append(param - num*1)
          param_domain.append(param)
          param_domain.append(param + num*1)
          param_domain.append(param + num*2)
        
        grid_search.append(param_domain)
        
    return grid_search


def _test_hyperpara(epochs, one_hyperpara, model, train_loader, validate_loader):
    wd, lrt = one_hyperpara
    lloss, loss, acc = holdoutCV(epochs, wd, lrt,
                                 model, train_loader, validate_loader)
    return lloss, loss, acc
