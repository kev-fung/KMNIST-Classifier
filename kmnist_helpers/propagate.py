# -*- coding: utf-8 -*-
import numpy as np
import torch
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from kmnist_helpers import device


def train(model, optimizer, criterion, data_loader):
    '''Train model over minibatches in dataset
    
    Args:
        model: pytorch nn.module model 
        optimizer: pytorch optimizer object
        criterion: pytorch criterion object
        data_loader: pytorch dataloader

    Returns:
        average loss per batch, average accuracy per batch
    '''
    model.train()
    train_loss, train_accuracy = 0, 0
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        a2 = model(X.view(-1, 1, 28, 28).float()) 
        loss = criterion(a2, y)
        loss.backward()
        train_loss += loss*X.size(0)
        y_pred = F.log_softmax(a2, dim=1).max(1)[1]
        train_accuracy += accuracy_score(y.cpu().numpy(),\
                                         y_pred.detach().cpu().numpy())*X.size(0)
        optimizer.step()
        
    return train_loss/len(data_loader.dataset), train_accuracy/len(data_loader.dataset)


def validate(model, criterion, data_loader):
    '''Validate model over minibatches in dataset, only compute forward
        propagation. Used for diagnostics. 
    
    Args:
        model: pytorch nn.module model 
        criterion: pytorch criterion object
        data_loader: pytorch dataloader

    Returns:
        average loss per batch, average accuracy per batch
    '''
    model.eval()
    validation_loss, validation_accuracy = 0., 0.
    for X, y in data_loader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            a2 = model(X.view(-1, 1, 28, 28).float())
            loss = criterion(a2, y)
            validation_loss += loss*X.size(0)
            y_pred = F.log_softmax(a2, dim=1).max(1)[1]
            validation_accuracy += accuracy_score(y.cpu().numpy(),\
                                                  y_pred.cpu().numpy())*X.size(0)
            
    return validation_loss/len(data_loader.dataset), validation_accuracy/len(data_loader.dataset)


def evaluate(model, data_loader):
    '''Evaluate final model over minibatches in dataset, only compute forward
        propagation. 
    
    Args:
        model: pytorch nn.module model 
        data_loader: pytorch dataloader

    Returns:
        model predictions, true labels
    '''
    model.eval()
    ys, y_preds = [], []
    for X, y in data_loader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)

            a2 = model(X.view(-1, 1, 28, 28).float())   
            y_pred = F.log_softmax(a2, dim=1).max(1)[1]
            ys.append(y.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())
            
    return np.concatenate(y_preds, 0),  np.concatenate(ys, 0)
