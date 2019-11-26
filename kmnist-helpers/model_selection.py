# -*- coding: utf-8 -*-
from kmnist_helpers import device, set_seed, seed
from kmnist_helpers.propagate import train, validate
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader
from livelossplot import PlotLosses
import torch
import torch.nn as nn

def holdoutCV(epochs, wd, lrt, model, train_loader, validate_loader):
    '''Holdout cross validation.
    
        Iterate over epochs to train model and simultaneously validate for 
        model diagnostics using liveloss plots.
        
        For gradient descent, we use the Adam algorithm.
    
    Args:
        epochs (int): number of epochs to train over
        wd (double): weight decay 
        lrt (double): learning rate
        model: pytorch nn.module model
        train_loader: pytorch dataloader to train model
        validate_loader: pytorch dataloader to validate model
    
    Returns:
        liveloss object, average validation loss, average validation accuracy
    '''
    set_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lrt, 
                                 betas=(0.9, 0.999), eps=1e-08, 
                                 weight_decay=wd, amsgrad=False)
    
    criterion = nn.CrossEntropyLoss()
      
    liveloss = PlotLosses()
    
    for epoch in range(epochs):
        logs = {}
        train_loss, train_accuracy = train(model, optimizer,
                                           criterion, train_loader)
        logs['' + 'log loss'] = train_loss.item()
        logs['' + 'accuracy'] = train_accuracy.item()

        validation_loss, validation_accuracy = validate(model, criterion,
                                                        validate_loader)
        logs['val_' + 'log loss'] = validation_loss.item()
        logs['val_' + 'accuracy'] = validation_accuracy.item()
        
        liveloss.update(logs)
        liveloss.draw()

    return liveloss, validation_loss, validation_accuracy


def holdout_loaders(X, y, cidataset, batch, testbatch, trans=True):
    '''Assemble train and validation loaders for holdout CV
    
    Args:
        X (array): feature set
        y (array): label set
        cidataset: pytorch custom dataset
        batch (int): batch size
        testbatch (int): test batch size
        trans (bool): apply custom transformation to X
        
    Returns:
        train loader, validation loader
    '''
    shuffler = StratifiedShuffleSplit(n_splits=1, test_size=0.1,
                                      random_state=seed).split(X, y)
    
    indices = [(t, v) for t, v in shuffler][0]
    
    X_train, y_train = X[indices[0]].astype(float), y[indices[0]]
    X_val, y_val = X[indices[1]].astype(float), y[indices[1]]

    X_train, y_train = torch.from_numpy(X_train).float(), torch.from_numpy(y_train)
    X_val, y_val =  torch.from_numpy(X_val).float(), torch.from_numpy(y_val)

    mean, std = torch.mean(X_train), torch.std(X_train)

    train_ds = cidataset(X_train, y_train.long(), transform=trans, mean=mean, std=std)
    val_ds = cidataset(X_val, y_val.long(), transform=False, mean=mean, std=std)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=testbatch, shuffle=False, num_workers=0)
    
    return train_loader, val_loader


def kfoldCV(epochs, wd, lrt, model, train_loaders, val_loaders):
    '''Kfold cross validation, must use kfold_loaders beforehand.
        
        Args:
            epochs (int): number of epochs
            wd (double): weight decay
            lrt (double): learning rate
            model: pytorch model
            train_loaders: list of pytorch loaders
            val_loaders: list of pytorch loaders

        Returns:
            livelosses per fold, average fold loss, average fold accuracy
    '''
    fold_liveloss = []
    fold_loss = 0.
    fold_acc = 0.
    
    for fold in range(len(train_loaders)):
        
        liveloss, val_loss, val_acc = holdoutCV(epochs, wd, lrt, model, 
                                                train_loaders[fold], 
                                                val_loaders[fold])

        fold_liveloss.append(liveloss)
        fold_loss += val_loss
        fold_acc += val_acc
        print("fold:", fold)

    print("Averaged Accuracy: ", (fold_acc/len(train_loaders))*100)
    return fold_liveloss, fold_loss, fold_acc


def kfold_loaders(k, X, y, cidataset, batch, testbatch, trans=True, v=False):
    '''Assemble train and validation loaders for Kfold CV
    
    Args:
        k (int): number of kfolds
        X (array): feature set
        y (array): label set
        cidataset: pytorch custom dataset
        batch (int): batch size
        testbatch (int): test batch size
        trans (bool): apply custom transformation to X
        v (bool): verbatim        

    Returns:
        kfold train loader, kfold val loader
    '''
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    fold_train_loader = [] # list of shuffled training dataloaders
    fold_validation_loader = [] # list of shuffled validation dataloaders

    for train_index, test_index in kf.split(X, y):
        if v: print("TRAIN:", train_index, "Validation:", test_index)
        
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]
    
        if v: print("train size:", X_train.shape, "test size:", X_val.shape)
    
        # Convert to pytensor
        X_train, y_train = torch.from_numpy(X_train).float(), torch.from_numpy(y_train)
        X_val, y_val = torch.from_numpy(X_val).float(), torch.from_numpy(y_val)
    
        # Find mean std
        mean, std = torch.mean(X_train), torch.std(X_train)
    
        # make custom set
        train_dataset = cidataset(X_train, y_train.long(), transform=trans, 
                                  mean=mean, std=std)
        validation_dataset = cidataset(X_val, y_val.long(), transform=False, 
                                       mean=mean, std=std)
    
        # initialize the data-loaders
        fold_train_loader.append(DataLoader(train_dataset, batch_size=batch, 
                                            shuffle=True, num_workers=4))
        fold_validation_loader.append(DataLoader(validation_dataset, 
                                                 batch_size=testbatch, 
                                                 shuffle=False, num_workers=0))

    return fold_train_loader, fold_validation_loader