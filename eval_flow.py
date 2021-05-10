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


data_path = "gendata/16features/"
run_1 = False
run_2 = True

dfs = []
    
filenames = os.listdir(data_path)

for f in filenames:
    df0 = pd.read_pickle(data_path+f)
    dfs.append(df0)

dfx = pd.concat(dfs)

if run_1:
    
    #df.hist()

    x_data = dfx[1]
    y_data = dfx[2]
    var_names = ["E Px","E Py"]
    saveplots = False
    output_dir = "."
    title = "Px vs Py"
    filename = title
    units = ["GeV","Gev"]
    ranges = [[-2,2,100],[-2,2,100]]

    make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
                                saveplot=saveplots,pics_dir=output_dir,plot_title=title.replace("/",""),
                                filename=filename,units=units)

  
    e = 0
    dfx['emass'] = dfx[e]**2-dfx[e+1]**2-dfx[e+2]**2-dfx[e+3]**2

    e = 4
    dfx['pmass'] = dfx[e]**2-dfx[e+1]**2-dfx[e+2]**2-dfx[e+3]**2

    e = 8
    dfx['g1mass'] = dfx[e]**2-dfx[e+1]**2-dfx[e+2]**2-dfx[e+3]**2

    e = 12
    dfx['g2mass'] = dfx[e]**2-dfx[e+1]**2-dfx[e+2]**2-dfx[e+3]**2

    dfx['Etot'] = dfx[0]**2-dfx[e+1]**2-dfx[e+2]**2-dfx[e+3]**2


    #dfx = dfx.head(3)
    print(dfx)
    #dfx = dfx.query('emass>-0.05 and emass<0.05 and pmass>0.86 and pmass<0.9 and g1mass>-0.05 and g1mass<0.05 and g2mass>-0.01 and g2mass<0.01')

    print(dfx)

    #np.absolute()

    bin_size = [100,100]


    xvals = dfx[8]
    x_name = "px E"
    output_dir = "./"
    ranges = "none"
    #ranges = [-1,1,20]

    make_histos.plot_1dhist(xvals,[x_name,],ranges=ranges,second_x="none",
                    saveplot=False,pics_dir=output_dir,plot_title=x_name,density=False)


    x_data = dfx[1]
    y_data = dfx[2]
    var_names = ["E Px","E Py"]
    saveplots = False
    output_dir = "."
    title = "Px vs Py"
    filename = title
    units = ["GeV","Gev"]
    ranges = [[-2,2,100],[-2,2,100]]

    make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
                                saveplot=saveplots,pics_dir=output_dir,plot_title=title.replace("/",""),
                                filename=filename,units=units)



if run_2:
    dfd = pd.read_pickle("data/pi0_100k.pkl")

    #df0 = pd.read_pickle("GenData0.pkl")
    #df1 = pd.read_pickle("GenData1.pkl")
    #dfX = [df0,df1]


    #dfXX = pd.concat(dfX)
    
    #zX = dfXX.to_numpy()

    dfXX = dfx
    output_dir = "gausspics/"

    import itertools
    parts = ["E","P","G1","G2"]
    feat = ["Energy","Px","Py","Pz"]
    a = parts
    b = feat
    #names = map(''.join, itertools.chain(itertools.product(list1, list2), itertools.product(list2, list1)))
    names = [r for r in itertools.product(a, b)]#: print r[0] + r[1]

    #names = [p for p in zip(parts,feat)]
    print(names)
    #sys.exit()
    vals = np.arange(0,16)
    for ind,x_key in enumerate(vals):
            name = names[ind]
            x_name = "{} {}".format(name[0],name[1])
            print("Creating 1 D Histogram for: {} ".format(x_key))
            xvals = dfd[x_key]
    #            make_histos.plot_1dhist(xvals,[x_name,],ranges="none",second_x="none",
    #                   saveplot=False,pics_dir=output_dir,plot_title=x_name)
            make_histos.plot_1dhist(xvals,[x_name,],ranges="none",second_x=dfXX[ind],
                    saveplot=False,pics_dir=output_dir,plot_title=x_name,density=True)


    x_data = dfd[1]
    y_data = dfd[2]
    var_names = ["E Px","E Py"]
    saveplots = False
    outdir = "."
    title = "Px vs Py"
    filename = title
    units = ["GeV","Gev"]
    ranges = [[-2,2,200],[-2,2,200]]

    from matplotlib import interactive
    interactive(True)
    make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
                                saveplot=saveplots,pics_dir=output_dir,plot_title=title.replace("/",""),
                                filename=filename,units=units)


    x_data = dfXX[1]
    y_data = dfXX[2]
    make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
                                saveplot=saveplots,pics_dir=output_dir,plot_title=title.replace("/",""),
                                filename=filename,units=units)


    interactive(False)
    plt.show()