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



dev = "cuda:0" if torch.cuda.is_available() else "cpu"
#dev = "cpu"
print(dev)
device = torch.device(dev)

#reonstruct an nflow model
model_path = "models/"
model_name = "TM_16_18_20_100_799_-15.19.pt"

params = model_name.split("_")
num_features = int(params[1])
num_layers = int(params[2])
num_hidden_features = int(params[3])
training_sample_size = int(params[4])
epoch_num = int(params[5])
training_loss = float((params[6]).split(".p")[0])
print(num_features,num_layers,num_hidden_features,training_sample_size,training_loss)

flow, optimizer = make_model(num_layers,num_features,num_hidden_features,device)
print("number of params: ", sum(p.numel() for p in flow.parameters()))
flow.load_state_dict(torch.load(model_path+model_name))
flow.eval()


zs = []
max_range = 20
sample_size = 200

start = datetime.now()
start_time = start.strftime("%H:%M:%S")
print("Start Time =", start_time)
for i in range(1,max_range+1):
    print("On set {}".format(i))
    z0= flow.double().sample(sample_size).cpu().detach().numpy()
    zs.append(z0)
    now = datetime.now()
    elapsedTime = (now - start )
    print("Current time is {}".format(now.strftime("%H:%M:%S")))
    print("Elapsed time is {}".format(elapsedTime))
    print("Total estimated run time is {}".format(elapsedTime+elapsedTime/i*(max_range+1-i)))

z = np.concatenate(zs)

df = pd.DataFrame(z)
df.to_pickle("gendata/GenData_{}_{}_{}_{}_{}.pkl".format(num_features,
          num_layers,num_hidden_features,training_sample_size,training_loss))