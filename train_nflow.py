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


# Define device to be used
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)
print(dev)

#Define hyperparameters
num_layers = 16
num_features = 16
num_hidden_features = 32
num_epoch = 10000
training_sample_size = 400


#read the data, with the defined data class
xz = dataXZ.dataXZ()

# QuantTran = xz.qt

# sampleDict = xz.sample(3)
# x = sampleDict["x"]
# qt_params = QuantTran.get_params(deep=True)
# print(qt_params)
# X = QuantTran.inverse_transform(x)




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
      if ((i+1)%100) == 0:
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
