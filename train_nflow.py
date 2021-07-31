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


outdir = "julypics/"
# Define device to be used
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
#dev = "cpu"
device = torch.device(dev)
print(dev)

#Define hyperparameters
#The number of featuers is just the length of the feature subset, or 16 if "all"
#feature_subset = [1,2,3,5,6,7,9,10,11,13,14,15] #Only 3 momenta (assuming PID is known)

#feature_subset = [1,2,3] #Just electron features
#feature_subset = [5,6,7] #Just proton features
feature_subset = [9,10,11] #Just photon 1 features
#feature_subset = [13,14,15] #Just photon 2 features
#feature_subset = "all" #All 16 features

#These are parameters for the Normalized Flow model
num_layers = 10
num_hidden_features = 10

#These are training parameters
num_epoch = 3000
training_sample_size = 800


if feature_subset == "all":
  num_features = 16
else:
  num_features = len(feature_subset)


#read the data, with the defined data class
xz = dataXZ.dataXZ(feature_subset=feature_subset)

# print("trying to sample")
# sampleDict = xz.sample(100000)
# x = sampleDict["x"][:, 0:num_features]
# z = sampleDict["z"][:, 0:num_features]
# x= x.detach().numpy()
# z = z.detach().numpy()
# print("trying to plot")

# bin_size = [100,100]


# plt.hist(x[:,0],color = "red", density=True,bins=100)
# plt.savefig(outdir+"feature0"+"_noQT")
# plt.close()

# plt.hist(x[:,1],color = "red", density=True,bins=100)
# plt.savefig(outdir+"feature1""_noQT")
# plt.close()


# plt.hist(x[:,2],color = "red", density=True,bins=100)
# plt.savefig(outdir+"feature2""_noQT")
# plt.close()


# fig, ax = plt.subplots(figsize =(5, 3)) 
# plt.hist2d(x[:,0], x[:,1],bins =bin_size,norm=mpl.colors.LogNorm())# cmap = plt.cm.nipy_spectral) 
# plt.colorbar()
# plt.show()


# fig, ax = plt.subplots(figsize =(5, 3)) 
# plt.hist2d(z[:,0], z[:,1],bins =bin_size,norm=mpl.colors.LogNorm())# cmap = plt.cm.nipy_spectral) 
# plt.colorbar()
# plt.show()

#sys.exit()


#construct an nflow model
flow, optimizer = make_model(num_layers,num_features,num_hidden_features,device)
print("number of params: ", sum(p.numel() for p in flow.parameters()))
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.98)


start = datetime.now()
start_time = start.strftime("%H:%M:%S")
print("Start Time =", start_time)
losses = []
for i in range(num_epoch):

    sampleDict = xz.sample(training_sample_size)
    x_train = sampleDict["x"][:, 0:num_features].to(device)
    z_train = sampleDict["z"][:, 0:num_features].to(device)

    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x_train,context=z_train).mean()

    loss.backward()
    optimizer.step()
    ## scheduler.step()
    #print('Epoch-{0} lr: {1}'.format(i, optimizer.param_groups[0]['lr']))
    losses.append(loss.item())
    print("Loss is {}".format(loss.item()))
    if ((i+1)%200) == 0:
      now = datetime.now()
      elapsedTime = (now - start )
      plt.scatter(np.arange(0,len(losses)),losses)
      plt.show()

      print("On step {} - loss {:.2f}, Current Time = {}".format(i,loss.item(),now.strftime("%H:%M:%S")))
      print("Elapsed time is {}".format(elapsedTime))
      print("Rate is {} seconds per epoch".format(elapsedTime/i))
      print("Total estimated run time is {}".format(elapsedTime+elapsedTime/i*(num_epoch+1-i)))
      if ((i+1)%100) == 0:
        torch.save(flow.state_dict(), "models/Cond/QT/INV/photon1/TM-UMNN_{}_{}_{}_{}_{}_{:.2f}.pt".format(num_features,
          num_layers,num_hidden_features,training_sample_size,i,loss.item()))

plt.scatter(np.arange(0,len(losses)),losses)
plt.show()

tm_name = "models/Cond/photon2/TM-Final-UMNN_{}_{}_{}_{}_{:.2f}.pt".format(num_features,
          num_layers,num_hidden_features,training_sample_size,losses[-1])
torch.save(flow.state_dict(), tm_name)
print("trained model saved to {}".format(tm_name))


email = True
if email:
  from pytools import circle_emailer
  now = datetime.now()
  script_end_time = now.strftime("%H:%M:%S")
  s_name = os.path.basename(__file__)
  subject = "Completion of {}".format(s_name)
  body = "Your script {} finished running at {}".format(s_name,script_end_time)
  circle_emailer.send_email(subject,body)
