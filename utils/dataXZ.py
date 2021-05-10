import pickle5 as pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.use('pdf')
import itertools
import numpy as np
from datetime import datetime
import torch
from torch import nn
from torch import optim
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MaxAbsScaler, QuantileTransformer

from utils.utilities import meter
from utils.utilities import cartesian_converter
from utils.utilities import make_model
from utils import make_histos
from utils import dataXZ

sys.path.insert(0,'/mnt/c/Users/rober/Dropbox/Bobby/Linux/classes/GAML/GAMLX/nflows/nflows')
from nflows.transforms.autoregressive import MaskedUMNNAutoregressiveTransform
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.distributions.normal import DiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation



#Create data class
class dataXZ:
  """
  read the data stored in pickle format
  the converting routine is at https://github.com/6862-2021SP-team3/hipo2pickle
  """
  def __init__(self, standard = False):
    with open('data/pi0.pkl', 'rb') as f:
        xz = np.array(pickle.load(f), dtype=np.float64)
        #xz = xz[:, 1:]
        # z = xz[:, 16:]
        x = cartesian_converter(xz)
        
        #x = x[:,[0,1,2,3]]
        
        
        #xwithoutPid = x[:, [0, 1, 4, 5, 8, 12, ]]
        # zwithoutPid = z[:, [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]]
        self.xz = xz
        self.qt = self.quant_tran(x)

        df_x = pd.DataFrame(self.qt.transform(x)) #Don't know how to do this without first making it a DF
        x_np = df_x.to_numpy() #And then converting back to numpy
        self.x = torch.from_numpy(np.array(x_np))

        # self.z = torch.from_numpy(np.array(z))
        # self.zwithoutPid = torch.from_numpy(zwithoutPid)

    if standard:
      self.standardize()

  def quant_tran(self,x):
    gauss_scaler = QuantileTransformer(output_distribution='normal').fit(x)
    return gauss_scaler

  def standardize(self):
    self.xMu = self.xwithoutPid.mean(0)
    self.xStd = self.xwithoutPid.std(0)
    self.zMu = self.zwithoutPid.mean(0)
    self.zStd = self.zwithoutPid.std(0)
    self.xwithoutPid = (self.xwithoutPid - self.xMu) / self.xStd
    self.zwithoutPid = (self.zwithoutPid - self.zMu) / self.zStd

  def restore(self, data, type = "x"):
    mu = self.xMu
    std = self.xStd
    if type == "z":
      mu = self.zMu
      std = self.zStd
    return data * std + mu

  def sample(self, n):
        randint = np.random.randint( self.xz.shape[0], size =n)
        xz = self.xz[randint]
        x = self.x[randint]
        # z = self.z[randint]
        #xwithoutPid = self.xwithoutPid[randint]
        # zwithoutPid = self.zwithoutPid[randint]
        # return {"xz":xz, "x": x, "z": z, "xwithoutPid": xwithoutPid, "zwithoutPid": zwithoutPid}
        return {"xz":xz, "x": x}

