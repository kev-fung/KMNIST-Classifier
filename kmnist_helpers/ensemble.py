# -*- coding: utf-8 -*-
from kmnist_helpers import device
import torch
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn.functional as F


def ensemble_validate(model_list, criterion, data_loader):
  """To be passed a list of pre-trained models sent to GPU
  RETURNS: validation accuracy of the ensembled model predictons"""
    
  for i in model_list:
      i.eval()

  validation_loss, validation_accuracy = 0., 0.
  
  a2 = 0

  for X, y in data_loader:
      with torch.no_grad():
          X, y = X.to(device), y.to(device)

          for model in model_list:
            # add the pre softmax values of all models
            a2 += model(X.view(-1, 1, 28, 28).float())

          a2 /= len(model_list)

          # predictions based on multiple model averages
          y_pred = F.log_softmax(a2, dim=1).max(1)[1]

          validation_accuracy += accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())*X.size(0)

  return validation_accuracy/len(data_loader.dataset)
  
  
def ensemble_evaluate(model_list, data_loader):
  """To be passed a list of pre-trained models sent to GPU
  RETURNS: list of predicted ys based on ensembled models"""
  for i in model_list:
    i.eval()

  a2 = 0
  ys, y_preds = [], []

  for X, y in data_loader:
      with torch.no_grad():
          X, y = X.to(device), y.to(device)

          for model in model_list:
            # add the pre softmax values of all models
            a2 += model(X.view(-1, 1, 28, 28).float())

          a2 /= len(model_list)

          y_pred = F.log_softmax(a2, dim=1).max(1)[1]
          ys.append(y.cpu().numpy())
          y_preds.append(y_pred.cpu().numpy())

  return np.concatenate(y_preds, 0)