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
from matplotlib import interactive
from matplotlib.patches import Rectangle

from utils import make_histos
from utils.utilities import meter
from utils.utilities import cartesian_converter

sys.path.insert(0,'/mnt/c/Users/rober/Dropbox/Bobby/Linux/classes/GAML/GAMLX/nflows/nflows')
from nflows.transforms.autoregressive import MaskedUMNNAutoregressiveTransform
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.distributions.normal import DiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

from icecream import ic

df_16 = pd.read_pickle("16_feature_pion.pkl")
df_4 = pd.read_pickle("4_feature_pion.pkl")


if len(df_16) > len(df_4):
    df_16 = df_16.head(len(df_4))
else:
    df_4 = df_4.head(len(df_16))




bin_size = [100,100]

var_name = 'Mpi0'
xvals = df_16[var_name]
xvals = df_4['recon_Mpi0']
xvals2 = df_4['nf_Mpi0']
###############





#x_name = "Gamma-Gamma Invariant Mass (GeV)"
x_name = "Reconstructed Pion Mass (GeV)"
output_dir = "./"
#ranges = "none"
#ranges = [12,28,100]
ranges = [.02,.3,100]
print("PLOTTTING")
make_histos.plot_1dhist(xvals,[x_name,],ranges=ranges,second_x=xvals2,annotation="yes",
                    saveplot=False,pics_dir=output_dir,plot_title="Reconstructed Pion Mass, NF Sampled",
                    density=True,proton_line=0.135,first_color="green",xlabel_1="Physics Recon. Data",xlabel_2="4-Feature Model")
