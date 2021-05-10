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
gen_all_emd = False
run_3 = True

dfs = []
    
filenames = os.listdir(data_path)

for f in filenames:
    df0 = pd.read_pickle(data_path+f)
    dfs.append(df0)

df_nflow_data = pd.concat(dfs)
nflow_data_len = len(df_nflow_data.index)
print("The Generated dataset has {} events".format(nflow_data_len))
df_test_data_all = pd.read_pickle("data/pi0_cartesian_test.pkl")
df_test_data = df_test_data_all.sample(n=nflow_data_len)

if run_1:
    df = df_nflow_data
    e = 0
    df['emass2'] = df[e]**2-df[e+1]**2-df[e+2]**2-df[e+3]**2

    e = 4
    df['pmass2'] = df[e]**2-df[e+1]**2-df[e+2]**2-df[e+3]**2

    e = 8
    df['g1mass2'] = df[e]**2-df[e+1]**2-df[e+2]**2-df[e+3]**2

    e = 12
    df['g2mass2'] = df[e]**2-df[e+1]**2-df[e+2]**2-df[e+3]**2

    e = 0
    df['protonE'] = df[4]
    df['Etot'] = df[e] + df[e+4]+df[e+8]+df[e+12]
    e = 1
    df['pxtot'] = df[e] + df[e+4]+df[e+8]+df[e+12]
    e = 2
    df['pytot'] = df[e] + df[e+4]+df[e+8]+df[e+12]
    e = 3
    df['pztot'] = df[e] + df[e+4]+df[e+8]+df[e+12]
    df['NetE'] = np.sqrt(df['Etot']**2 - df['pxtot']**2 - df['pytot']**2 - df['pztot']**2)

    epsilon = 0.05
    #df2 = df.query("NetE>(4.556-{}) and NetE<(4.556+{})".format(epsilon,epsilon))
    df2 = df.query("protonE<1.475")#.format(epsilon,epsilon))
    print(df)
    print(df2)
    #sys.exit()

    bin_size = [100,100]
    xvals = df2[4]
    x_name = "px E"
    output_dir = "./"
    #ranges = "none"
    ranges = [1.4,1.6,30]

    make_histos.plot_1dhist(xvals,[x_name,],ranges=ranges,second_x="none",
                    saveplot=False,pics_dir=output_dir,plot_title=x_name,density=False)
    x_data = df[1]
    y_data = df[2]
    var_names = ["E Px","E Py"]
    saveplots = False
    output_dir = "."
    title = "Px vs Py"
    filename = title
    units = ["GeV","Gev"]
    ranges = [[-2,2,100],[-2,2,100]]

    interactive(True)
    make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
                                saveplot=saveplots,pics_dir=output_dir,plot_title=title.replace("/",""),
                                filename=filename,units=units)

    x_data = df2[1]
    y_data = df2[2]
    make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
                            saveplot=saveplots,pics_dir=output_dir,plot_title=title.replace("/",""),
                            filename=filename,units=units)
    interactive(False)
    plt.show()

    sys.exit()

if gen_all_emd:

    num_vars = len(df_nflow_data.columns)
    num_iters = 40
    nflow_set = []
    test_set = []
    nflow_vals = np.empty(shape=(num_vars,num_iters))
    test_vals = np.empty(shape=(num_vars,num_iters))
    for i in range(num_iters):
        df_test_data_0 = df_test_data_all.sample(n=nflow_data_len)
        df_test_data_1 = df_test_data_all.sample(n=nflow_data_len)
        for feature_num in range(num_vars):
            emd_nflow, _, _ = meter(df_test_data_0.to_numpy(),df_nflow_data.to_numpy(),feature_num)
            emd_test_data, _, _ = meter(df_test_data_0.to_numpy(),df_test_data_1.to_numpy(),feature_num)
            #print("EMD for feature {} is {:.4f} vs. {:.4f} from test data, a {:.3f} ratio".format(feature_num,emd_nflow,emd_test_data,emd_nflow/emd_test_data))
            nflow_vals[feature_num][i] = emd_nflow
            test_vals[feature_num][i] = emd_test_data
        # nflow_set.append(nflow_vals)
        # test_set.append(test_vals)
    #print(nflow_vals)
    nflow_mean = np.mean(nflow_vals,axis=1)
    nflow_std = np.std(nflow_vals,axis=1)
    test_mean = np.mean(test_vals,axis=1)
    test_std = np.std(test_vals,axis=1)
    ratio_2 = np.mean(nflow_vals/test_vals,axis=1)

    ratio_std_2 = np.std(nflow_vals/test_vals,axis=1)

    x = np.arange(len(ratio_2))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "20"

    fig, ax = plt.subplots(figsize =(10, 7)) 
    
    ax.set_xlabel("Feature Number")  
    ax.set_ylabel("Earth Mover's Distance Ratio")
    #plt.tick_params(labelsize=24)

    emd_plot = plt.errorbar(x,ratio_2,yerr=ratio_std_2,fmt='o',label="EMD Ratio",linewidth=2)
    ideal_line = plt.axhline(y = 1, color = 'r', linestyle = '-',label="Ideal Case",linewidth=2)
    plt.title("Earth Mover's Distance Ratio, All Features")
    #ax.legend([emd_plot, ideal_line])
    plt.legend(loc='upper center')

    plt.xlim([-.5,15.5])
    plt.ylim([0,10])
    #plt.show()
    plotname = "nflow_emd.png"
    plt.savefig(plotname)
    plt.close()

if run_3:
    output_dir = "hists_1D/"

    import itertools
    parts = ["Electron","Proton","Photon 1","Photon 2"]
    feat = ["Energy","X-Momentum","Y-Momentum","Z-Momentum"]
    a = parts
    b = feat
    #names = map(''.join, itertools.chain(itertools.product(list1, list2), itertools.product(list2, list1)))
    names = [r for r in itertools.product(a, b)]#: print r[0] + r[1]



    for feature_ind in range(16):
            name = names[feature_ind]
            x_name = "{} {}".format(name[0],name[1])
            print("Creating 1 D Histogram for: {} ".format(name))
            emd_nflow, _, _ = meter(df_test_data.to_numpy(),df_nflow_data.to_numpy(),feature_ind)
            xvals_1 = df_test_data[feature_ind]
            xvals_2 = df_nflow_data[feature_ind]
            make_histos.plot_1dhist(xvals_1,[x_name,],ranges="none",second_x=xvals_2,
                    saveplot=False,pics_dir=output_dir,plot_title=x_name,density=True,annotation=emd_nflow)


    x_data = df_test_data[5]
    y_data = df_test_data[6]
    var_names = ["E Px","E Py"]
    saveplots = False
    outdir = "."
    title = "Px vs Py"
    filename = title
    units = ["GeV","Gev"]
    #ranges = [[-2,2,100],[-2,2,100]]
    ranges = [[-1,1,100],[-1,1,100]]

    interactive(True)
    make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
                                saveplot=saveplots,pics_dir=output_dir,plot_title=title.replace("/",""),
                                filename=filename,units=units)


    x_data = df_test_data[5]
    y_data = df_test_data[6]
    make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
                                saveplot=saveplots,pics_dir=output_dir,plot_title=title.replace("/",""),
                                filename=filename,units=units)


    interactive(False)
    plt.show()