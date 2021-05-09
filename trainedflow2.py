
import pickle5 as pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.use('pdf')
import sklearn.datasets as datasets
import itertools
import numpy as np

from datetime import datetime
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from scipy.spatial import distance

import torch
from torch import nn
from torch import optim

import pandas as pd

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.distributions.normal import DiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

import sys
sys.path.insert(0,'/mnt/c/Users/rober/Dropbox/Bobby/Linux/classes/GAML/GAMLX/nflows/nflows')

from nflows.transforms.autoregressive import MaskedUMNNAutoregressiveTransform

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
#dev = "cpu"
print(dev)
device = torch.device(dev)


num_layers = 18#12
num_features = 16
base_dist = StandardNormal(shape=[num_features])
#base_dist = DiagonalNormal(shape=[3])
transforms = []
for _ in range(num_layers):
    transforms.append(ReversePermutation(features=num_features))

    transforms.append(MaskedUMNNAutoregressiveTransform(features=num_features, 
                                                          hidden_features=20)) 
                                                          #context_features=len(in_columns)))
    #transforms.append(MaskedAffineAutoregressiveTransform(features=num_features, 
    #                                                      hidden_features=100))



transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist).to(device)

#model = MaskedUMNNAutoregressiveTransform()
flow.load_state_dict(torch.load("models/trainedmodel_799_100_-15.19.pkl"))
#flow.load_state_dict(torch.load("trainedmodel_99_100_-2.76.pkl"))
#flow.double().eval()

zs = []

start = datetime.now()
start_time = start.strftime("%H:%M:%S")
print("Start Time =", start_time)

max_range = 10
sample_size = 200
for i in range(1,max_range+1):
    print("On set {}".format(i))
    z0= flow.double().sample(sample_size).cpu().detach().numpy()
    zs.append(z0)
    now = datetime.now()
    elapsedTime = (now - start )
    print("Elapsed time is {}".format(elapsedTime))
    print("Total estimated run time is {}".format(elapsedTime+elapsedTime/i*(max_range+1-i)))

zX = np.concatenate(zs)
#zX = np.concatenate((z0,z1,z2,z3,z4,z5,z6,z7,z8,z9),axis=0)
#print(z)
#sys.exit()

dfz = pd.DataFrame(zX)
dfz.to_pickle("GenData.pkl")
#print(dfz)

print('gen z final done')

bin_size = [150,150]
fig, ax = plt.subplots(figsize =(10, 7)) 
plt.rcParams["font.size"] = "16"
ax.set_xlabel("Electron Momentum x")  
ax.set_ylabel("Electron Momentum y")
plt.title('NFlow Generated Distribution - Final Iteration')

plt.hist2d(zX[:,1], zX[:,2],bins =bin_size,norm=mpl.colors.LogNorm())# cmap = plt.cm.nipy_spectral) 
#plt.xlim([-2,2])
#plt.ylim([-2,2])
#plt.colorbar()
plotname = "finalplotname2.jpeg"

plt.show()
