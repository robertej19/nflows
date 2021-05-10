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

from utils.utilities import meter
from utils.utilities import cartesian_converter
from utils.utilities import make_model

sys.path.insert(0,'/mnt/c/Users/rober/Dropbox/Bobby/Linux/classes/GAML/GAMLX/nflows/nflows')
from nflows.transforms.autoregressive import MaskedUMNNAutoregressiveTransform
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.distributions.normal import DiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation


# Define device to be used
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)
print(dev)

#Define hyperparameters
num_layers = 18
num_features = 16
num_hidden_features = 20
num_epoch = 40000
training_sample_size = 100

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
        xwithoutPid = x
        #xwithoutPid = x[:, [0, 1, 4, 5, 8, 12, ]]
        # zwithoutPid = z[:, [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]]
        self.xz = xz
        self.x = torch.from_numpy(np.array(x))
        # self.z = torch.from_numpy(np.array(z))
        self.xwithoutPid = torch.from_numpy(xwithoutPid)
        # self.zwithoutPid = torch.from_numpy(zwithoutPid)

    if standard:
      self.standardize()

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
        xwithoutPid = self.xwithoutPid[randint]
        # zwithoutPid = self.zwithoutPid[randint]
        # return {"xz":xz, "x": x, "z": z, "xwithoutPid": xwithoutPid, "zwithoutPid": zwithoutPid}
        return {"xz":xz, "x": x, "xwithoutPid": xwithoutPid}

#read the data, with the defined data class
xz = dataXZ()

#construct an nflow model
flow, optimizer = make_model(num_layers,num_features,num_hidden_features,device)
print("number of params: ", sum(p.numel() for p in flow.parameters()))


start = datetime.now()
start_time = start.strftime("%H:%M:%S")
print("Start Time =", start_time)
losses = []
for i in range(num_epoch):
    sampleDict = xz.sample(training_sample_size)
    x = sampleDict["x"][:, 0:num_features].to(device)
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x).mean()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if ((i+1)%10) == 0:
      now = datetime.now()
      elapsedTime = (now - start )
      print("On step {} - loss {:.2f}, Current Time = {}".format(i,loss.item(),now.strftime("%H:%M:%S")))
      print("Elapsed time is {}".format(elapsedTime))
      print("Rate is {} seconds per epoch".format(elapsedTime/i))
      print("Total estimated run time is {}".format(elapsedTime+elapsedTime/i*(num_epoch+1-i)))
      if ((i+1)%200) == 0:
        torch.save(flow.state_dict(), "models/TM_{}_{}_{}_{}_{}_{:.2f}.pt".format(num_features,
          num_layers,num_hidden_features,training_sample_size,i,loss.item()))

tm_name = "models/TM-Final_{}_{}_{}_{}_{:.2f}.pt".format(num_features,
          num_layers,num_hidden_features,training_sample_size,losses[-1])
torch.save(flow.state_dict(), tm_name)
print("trained model saved to {}".format(tm_name))


if email:
  from pytools import circle_emailer
  now = datetime.now()
  script_end_time = now.strftime("%H:%M:%S")
  s_name = os.path.basename(__file__)
  subject = "Completion of {}".format(s_name)
  body = "Your script {} finished running at {}".format(s_name,script_end_time)
  circle_emailer.send_email(subject,body)
