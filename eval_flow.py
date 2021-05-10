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

from utils import make_histos
from utils.utilities import meter


sys.path.insert(0,'/mnt/c/Users/rober/Dropbox/Bobby/Linux/classes/GAML/GAMLX/nflows/nflows')
from nflows.transforms.autoregressive import MaskedUMNNAutoregressiveTransform
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.distributions.normal import DiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation


run_1 = True
if run_1:
    dfs = []
    for i in range(7):
        df_pkl_name = "gendata/16_18_20_100_-15.19/GenDataDouble_{}.pkl".format(i)
        df0 = pd.read_pickle(df_pkl_name)
        dfs.append(df0)

    dfx = pd.concat(dfs)
    #df.hist()

  
    e = 0
    dfx['emass'] = dfx[e]**2-dfx[e+1]**2-dfx[e+2]**2-dfx[e+3]**2

    e = 4
    dfx['pmass'] = dfx[e]**2-dfx[e+1]**2-dfx[e+2]**2-dfx[e+3]**2

    e = 8
    dfx['g1mass'] = dfx[e]**2-dfx[e+1]**2-dfx[e+2]**2-dfx[e+3]**2

    e = 12
    dfx['g2mass'] = dfx[e]**2-dfx[e+1]**2-dfx[e+2]**2-dfx[e+3]**2

    #dfx = dfx.head(3)
    print(dfx)
    dfx = dfx.query('emass>-0.25 and emass<0.25 and pmass>0.8 and pmass<1 and g1mass>-0.2 and g1mass<0.2 and g2mass>-0.2 and g2mass<0.2')

    print(dfx)

    #np.absolute()

    bin_size = [100,100]


    xvals = dfx[1]
    x_name = "px E"
    output_dir = "./"
    ranges = [-1,1,20]

    make_histos.plot_1dhist(xvals,[x_name,],ranges=ranges,second_x="none",
                    saveplot=False,pics_dir=output_dir,plot_title=x_name,density=False)

    x_data = dfx[1]
    y_data = dfx[2]
    var_names = ["E Px","E Py"]
    saveplots = False
    outdir = "."
    title = "Px vs Py"
    filename = title
    units = ["GeV","Gev"]
    ranges = [[-2,2,200],[-2,2,200]]

    from matplotlib import interactive
    make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
                                saveplot=saveplots,pics_dir=output_dir,plot_title=title.replace("/",""),
                                filename=filename,units=units)

                       