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
  def __init__(self, standard = False, feature_subset = "all", test=False):
    #use if already converted to cartesian
    #with open('data/pi0_cartesian_train.pkl', 'rb') as f:
       #x = np.array(pickle.load(f), dtype=np.float32)



    #For building Quantile transforms
    qt_data = 'data/pi0_spherical_train.pkl'
    with open(qt_data, 'rb') as fname:
        qt_xz = np.array(pickle.load(fname), dtype=np.float32)
        qt_x = cartesian_converter(qt_xz,type='x')
        qt_z = cartesian_converter(qt_xz,type='z')

        if feature_subset != "all": 
          qt_x = qt_x[:,feature_subset]
          qt_z = qt_z[:,feature_subset]

    self.qt_x = self.quant_tran(qt_x)
    self.qt_z = self.quant_tran(qt_z)

    #Use if not already converted
    if test:
      print("Test flag is enabled")
    fname = 'data/pi0_spherical_test.pkl' if test else 'data/pi0_spherical_train.pkl'
    print(fname)

    with open(fname, 'rb') as f:
        xz = np.array(pickle.load(f), dtype=np.float32)
        x = cartesian_converter(xz,type='x')
        z = cartesian_converter(xz,type='z')
        

        if feature_subset != "all": 
          x = x[:,feature_subset]
          z = z[:,feature_subset]

        xwithoutPid = x


        #self.qt = self.quant_tran(x)

        #For use with quant trans.
        df_x = pd.DataFrame(self.qt_x.transform(x)) #Don't know how to do this without first making it a DF
        df_z = pd.DataFrame(self.qt_z.transform(z)) #Don't know how to do this without first making it a DF

        x_np = df_x.to_numpy() #And then converting back to numpy
        z_np = df_z.to_numpy() #And then converting back to numpy
        
        # #IF USING QT:
        self.x = torch.from_numpy(np.array(x_np))
        self.z = torch.from_numpy(np.array(z_np))


        # IF NOT USING QT:
        #self.x = torch.from_numpy(np.array(x))
        #self.z = torch.from_numpy(np.array(z))   


        self.xz = xz

        #Commented out because currently ton using Quant trans.
        # df_x = pd.DataFrame(self.qt.transform(x)) #Don't know how to do this without first making it a DF
        # x_np = df_x.to_numpy() #And then converting back to numpy
        # self.x = torch.from_numpy(np.array(x_np))



        # #Xommented out because trying to reimplement quant trans.
        # #self.xz = xz
        # self.x = torch.from_numpy(np.array(x))
        # self.xwithoutPid = torch.from_numpy(np.array(xwithoutPid))
        # self.z = torch.from_numpy(np.array(z))


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
        #xz = self.xz[randint]
        x = self.x[randint]
        z = self.z[randint]
        #xwithoutPid = self.xwithoutPid[randint]
        # zwithoutPid = self.zwithoutPid[randint]
        # return {"xz":xz, "x": x, "z": z, "xwithoutPid": xwithoutPid, "zwithoutPid": zwithoutPid}
        return {"x": x,"z": z}
