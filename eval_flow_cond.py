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


#data_path = "gendata/4features/" #Just electorn features
#data_path = "gendata/16features/" #All 16 features
#data_path = "gendata/Cond/16features/maaf/"
#data_path = "gendata/Cond/16features/UMNN/"
data_path = "gendata/Cond/proton/UMNN/"


physics_cuts = False
gen_all_emd = False
gen_1d_histos = True
gen_emd_comp = False

dfs = []
    
filenames = os.listdir(data_path)

for f in filenames:
    df0 = pd.read_pickle(data_path+f)
    dfs.append(df0)

df_nflow_data = pd.concat(dfs)
nflow_data_len = len(df_nflow_data.index)
print("The Generated dataset has {} events".format(nflow_data_len))


with open('data/pi0.pkl', 'rb') as f:
    xz = np.array(pickle.load(f), dtype=np.float32)
    x = cartesian_converter(xz,type='x')
    z = cartesian_converter(xz,type='z')
        

df_test_data = pd.DataFrame(x)
df_test_data_z = pd.DataFrame(z)


if len(df_nflow_data) > len(df_test_data):
    df_nflow_data = df_nflow_data.sample(n=len(df_test_data))
else:
    df_test_data = df_test_data.sample(n=len(df_nflow_data))
    df_test_data_z = df_test_data_z.sample(n=len(df_nflow_data))

df_test_data = df_test_data.drop(columns=[0,1,2,3,8,9,10,11,12,13,14,15])
df_test_data.columns = [0,1,2,3]
df_test_data_z = df_test_data_z.drop(columns=[0,1,2,3,8,9,10,11,12,13,14,15])
df_test_data_z.columns = [0,1,2,3]
print(df_test_data)


if physics_cuts:



    dvpi0p = df_nflow_data
    #dvpi0p = df_test_data

    e=4
    dvpi0p.loc[:,'pmass'] = np.sqrt(dvpi0p[e]**2-dvpi0p[e+1]**2-dvpi0p[e+2]**2-dvpi0p[e+3]**2)/0.938


   
    dvpi0p.loc[:, "Gpx"] = dvpi0p.loc[:, 9]
    dvpi0p.loc[:, "Gpy"] = dvpi0p.loc[:, 10]
    dvpi0p.loc[:, "Gpz"] = dvpi0p.loc[:, 11]
    dvpi0p.loc[:, "Gpx2"] = dvpi0p.loc[:, 13]
    dvpi0p.loc[:, "Gpy2"] = dvpi0p.loc[:, 14]
    dvpi0p.loc[:, "Gpz2"] = dvpi0p.loc[:, 15]
    gam1 = [dvpi0p['Gpx'], dvpi0p['Gpy'], dvpi0p['Gpz']]
    gam2 = [dvpi0p['Gpx2'], dvpi0p['Gpy2'], dvpi0p['Gpz2']]


    

    pi0 = [dvpi0p['Gpx']+dvpi0p['Gpx2'], dvpi0p['Gpy']+dvpi0p['Gpy2'], dvpi0p['Gpz']+dvpi0p['Gpz2']]
    def dot(vec1, vec2):
        # dot product of two 3d vectors
        return vec1[0]*vec2[0]+vec1[1]*vec2[1]+vec1[2]*vec2[2]
    def mag(vec1):
        # L2 norm of vector
        return np.sqrt(dot(vec1, vec1))

    dvpi0p.loc[:, "Mpi0"] = np.sqrt((mag(gam1)+mag(gam2))**2 - mag(pi0)**2)
    
    

    df = dvpi0p



    e = 0
    df['protonE'] = df[4]
    df['Etot'] = df[e] + df[e+4]+df[e+8]+df[e+12]
    e = 1
    df['pxtot'] = df[e] + df[e+4]+df[e+8]+df[e+12]
    e = 2
    df['pytot'] = df[e] + df[e+4]+df[e+8]+df[e+12]
    e = 3
    df['pztot'] = df[e] + df[e+4]+df[e+8]+df[e+12]
    df['NetE'] = df['Etot']**2 - df['pxtot']**2 - df['pytot']**2 - df['pztot']**2


    #dvpi0p = dvpi0p.query("pmass<0.945 and pmass>0.931")
    #df = df.query('NetE<22.5 and NetE>19.5')
    #df = df.query('Mpi0<0.16 and Mpi0>0.11')

    df["Mpi0"] = df["Mpi0"]/.135

    bin_size = [100,100]

    var_name = 'Mpi0'
    xvals = df[var_name]

    
    #x_name = "Gamma-Gamma Invariant Mass (GeV)"
    x_name = "Reconstructed Pion Mass (Reduced)"
    output_dir = "./"
    #ranges = "none"
    #ranges = [12,28,100]
    ranges = [.25,1.75,100]
    print("PLOTTTING")
    make_histos.plot_1dhist(xvals,[x_name,],ranges=ranges,second_x=None,annotation=None,
                     saveplot=False,pics_dir=output_dir,plot_title="Reconstructed Pion Mass, NF Sampled",
                     density=False,proton_line=1,first_color="blue",xlabel_1="NF Data")
    
    sys.exit()
    x_data = df["Mpi0"]
    y_data = df["pmass"]
    var_names = ["E Px","E Py"]
    saveplots = False
    output_dir = "./"
    title = "Pi Mass vs EPx"
    filename = title
    units = ["GeV","Gev"]
    ranges = [[0,.3,100],[.9,.96,100]]

    #interactive(True)
    make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=False,
                                saveplot=saveplots,pics_dir=output_dir,plot_title=title.replace("/",""),
                                filename=filename.replace(" ",""),units=units)
    sys.exit()

    x_data = df2[1]
    y_data = df2[2]
    title = "Electron $P_X$ vs. $P_Y$, NF 4-Feat. Model, $M_e^2$ Cut"

    make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=False,
                            saveplot=saveplots,pics_dir=output_dir,plot_title=title.replace("/",""),
                            filename=filename.replace(" ",""),units=units)
    #interactive(False)
    #plt.show()

    sys.exit()

if gen_all_emd:

    num_vars = len(df_nflow_data.columns)
    num_iters = 10
    nflow_set = []
    test_set = []
    nflow_vals = np.empty(shape=(num_vars,num_iters))
    test_vals = np.empty(shape=(num_vars,num_iters))
    for i in range(num_iters):
        df_test_data_0 = df_test_data.sample(n=int(nflow_data_len/10))
        df_test_data_1 = df_test_data.sample(n=int(nflow_data_len/10))
        #df_nflow_data_0 = df_test_data.sample(n=int(nflow_data_len/10))
        df_nflow_data_0 = df_nflow_data.sample(n=int(nflow_data_len/10))
        for feature_num in range(num_vars):
            emd_nflow, _, _ = meter(df_test_data_0.to_numpy(),df_nflow_data_0.to_numpy(),feature_num)
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
    plt.title("EMD Ratio, NF Sampled vs. Physics Data")
    #ax.legend([emd_plot, ideal_line])
    plt.legend(loc='upper center')

    plt.xlim([-.5,15.5])
    plt.ylim([0,10])

    #locs, labels = plt.xticks()  # Get the current locations and labels.
    #xticks(np.arange(0, 1, step=0.2))  # Set label locations. 
    #plt.xticks(np.arange(16), ['Energy', 'X-Mom.', 'Y-Mom.', 'Z-Mom.','Energy', 'X-Mom.', 'Y-Mom.', 'Z-Mom.','Energy', 'X-Mom.', 'Y-Mom.', 'Z-Mom.','Energy', 'X-Mom.', 'Y-Mom.', 'Z-Mom.' ])  # Set text labels.
    plt.xticks(np.arange(16),np.arange(16))  # Set text labels.

    #plt.show()
    plotname = "nflow_emd_16_updated.png"


    plt.savefig(plotname)
    plt.close()

if gen_emd_comp:

    #df_nflow_data = df_nflow_data.sample(n=len(df_nflow_data_16))
    #df_test_data = df_test_data.sample(n=len(df_nflow_data_16))

    print(int(len(df_nflow_data_16.index)))

    num_vars = len(df_nflow_data.columns)
    num_iters = 20
    nflow_set = []
    test_set = []
    nflow_vals = np.empty(shape=(num_vars,num_iters))
    nflow_vals_16 = np.empty(shape=(num_vars,num_iters))
    test_vals = np.empty(shape=(num_vars,num_iters))
    for i in range(num_iters):
        df_test_data_0 = df_test_data.sample(n=int(len(df_nflow_data_16.index)))
        df_test_data_1 = df_test_data.sample(n=int(len(df_nflow_data_16.index)))
        df_nflow_data_0 = df_nflow_data.sample(n=int(len(df_nflow_data_16.index)))
        for feature_num in range(num_vars):
            emd_nflow, _, _ = meter(df_test_data_0.to_numpy(),df_nflow_data_0.to_numpy(),feature_num)
            emd_nflow16, _, _ = meter(df_test_data_0.to_numpy(),df_nflow_data_16.to_numpy(),feature_num)
            emd_test_data, _, _ = meter(df_test_data_0.to_numpy(),df_test_data_1.to_numpy(),feature_num)
            #print("EMD for feature {} is {:.4f} vs. {:.4f} from test data, a {:.3f} ratio".format(feature_num,emd_nflow,emd_test_data,emd_nflow/emd_test_data))
            nflow_vals[feature_num][i] = emd_nflow
            nflow_vals_16[feature_num][i] = emd_nflow16
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

    nflow_mean_16 = np.mean(nflow_vals_16,axis=1)
    nflow_std_16 = np.std(nflow_vals_16,axis=1)
    ratio_16 = np.mean(nflow_vals_16/test_vals,axis=1)
    ratio_std_16 = np.std(nflow_vals_16/test_vals,axis=1)



    x = np.arange(len(ratio_2))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "20"

    fig, ax = plt.subplots(figsize =(10, 7)) 
    
    ax.set_xlabel("Feature Number")  
    ax.set_ylabel("Earth Mover's Distance Ratio")
    #plt.tick_params(labelsize=24)

    emd_plot = plt.errorbar(x,ratio_2,yerr=ratio_std_2,fmt='gx',label="EMD Ratio - 4 Feat. Model",linewidth=2)
    emd_plot_16 = plt.errorbar(x,ratio_16,yerr=ratio_std_16,fmt='o',label="EMD Ratio - 16 Feat. Model",linewidth=2)
    ideal_line = plt.axhline(y = 1, color = 'r', linestyle = '-',label="Ideal Case",linewidth=2)
    plt.title("EMD Ratio, 4 and 16 Feat. Models")
    #ax.legend([emd_plot, ideal_line])
    plt.legend(loc='upper right')

    plt.xlim([-.5,3.5])
    plt.ylim([0,10])
    plt.show()
    plotname = "nflow_emd_4.png"
    plt.savefig(plotname)
    plt.close()



if gen_1d_histos:
    output_dir = "hists_1D/"

    # # import itertools
    # # parts = ["Proton","Photon 1","Photon 2"]
    # # feat = ["Energy (GeV)","X-Momentum (GeV)","Y-Momentum (GeV)","Z-Momentum (GeV)"]
    # # a = parts
    # # b = feat
    # # #names = map(''.join, itertools.chain(itertools.product(list1, list2), itertools.product(list2, list1)))
    # # names = [r for r in itertools.product(a, b)]#: print r[0] + r[1]

    
    # # #df = df_nflow_data
    # # # e = 0
    # # # df['emass2'] = df[e]**2-df[e+1]**2-df[e+2]**2-df[e+3]**2
    # # # epsilon = 2
    # # # df_nflow_data = df.query("emass2>(0-{}) and emass2<(0+{})".format(epsilon,epsilon))
    
    # # df_test_data.columns = ["e",1,2,3]
    # # df_nflow_data.columns = ["e",1,2,3]
    # # print(df_test_data)
    # # df_nflow_data = df_nflow_data.query("e<1.03")
    # # df_test_data = df_test_data.query("e < 1.03")
    # # print(df_test_data)

    # # if len(df_nflow_data) > len(df_test_data):
    # #     df_nflow_data = df_nflow_data.sample(n=len(df_test_data))
    # # else:
    # #     df_test_data = df_test_data.sample(n=len(df_nflow_data))

    # # print(df_nflow_data)
    # # print(df_test_data)

    # # df_test_data.columns = [0,1,2,3]
    # # df_nflow_data.columns = [0,1,2,3]

    # # #Uncomment for plotting all 1D histograms
    # # for feature_ind in range(4):
    # #         name = names[feature_ind]
    # #         x_name = "{} {}".format(name[0],name[1])
    # #         print("Creating 1 D Histogram for: {} ".format(name))
    # #         emd_nflow, _, _ = meter(df_test_data.to_numpy(),df_nflow_data.to_numpy(),feature_ind)
    # #         xvals_1 = df_test_data[feature_ind]
    # #         xvals_2 = df_nflow_data[feature_ind]
    # #         make_histos.plot_1dhist(xvals_1,[x_name,],ranges="none",second_x=xvals_2,
    # #                 saveplot=True,pics_dir=output_dir,plot_title="{}, NF 4-Feature Model".format(x_name),density=True,
    # #                 annotation=emd_nflow,first_color="red",xlabel_1="Microphysics Data",xlabel_2="NF Model Sample")
    # # sys.exit()

    # # # output_dir = "hists_2D_4F/"
    #saveplots = False
    saveplots = True

    #interactive(True)


    # #electron x mom, proton x mom
    # x,y = 6,11
    # var_names = ["Proton Y-Momentum","Photon 1 Z-Momentum"]
    # title = "Proton $P_X$ vs. Photon 1 $P_Z$"
    # ranges = [[-.75,.75,100],[1,9,100]]

    x,y = 0,3
    var_names = ["Proton Energy","Proton Z-Momentum"]
    title = "Proton E vs. Proton $P_Z$"
    ranges = [[1,1.5,100],[0,1.2,100]]

    ######### electron x mom, proton x mom
    # x,y = 1,5 
    # var_names = ["Electron X-Momentum","Proton X-Momentum"]
    # title = "Electon $P_X$ vs. Proton $P_X$"
    # ranges = [[-1.5,1.5,100],[-.75,.75,100]]

    # #electron x mom, electron y mom
    # x,y = 1,2
    # var_names = ["Electron X-Momentum","Electron Y-Momentum"]
    # title = "Electon $P_X$ vs. Electron $P_Y$"
    # #ranges = [[-3,3,100],[1,9,100]]
    # ranges = [[-1.5,1.5,200],[-1.5,1.5,200]]

    # #electron x mom, electron y mom
    # x,y = 1,2
    # var_names = ["Electron X-Momentum","Electron Z-Momentum"]
    # title = "Electon $P_X$ vs. Electron $P_Z$"
    # ranges = [[-1.5,1.5,100],[1,7,100]]


    # e = 0
    # df_nflow_data['emass2'] = df_nflow_data[e]**2-df_nflow_data[e+1]**2-df_nflow_data[e+2]**2-df_nflow_data[e+3]**2
    # epsilon = 0.5
    # df_nflow_data = df_nflow_data.query("emass2>(0-{}) and emass2<(0+{})".format(epsilon,epsilon))


    #electron mass squared, electron energy
    # x,y = 'emass2',1
    # var_names = ["Electron Mass Squared","Electron Z Mom."]
    # title = "Electon Mass Squared vs. Electron Z Mom."
    # ranges = [[-1.5,1.5,100],[-1.5,1.5,100]]


    title_phys = title+ ", Physics Data"
    title_phys_z = title+ ", Gen Events Data"
    title_nf = title + ", NF 4-Feature Cond Model"
    filename = title
    units = ["GeV","Gev"]
    x_data = df_test_data[x]
    y_data = df_test_data[y]
    x_data_z = df_test_data_z[x]
    y_data_z = df_test_data_z[y]
    x_data_nf = df_nflow_data[x]
    y_data_nf = df_nflow_data[y]


    interactive(True)

    make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=False,
                                saveplot=saveplots,pics_dir=output_dir,plot_title=title_phys.replace("/",""),
                                filename=filename,units=units)

    make_histos.plot_2dhist(x_data_z,y_data_z,var_names,ranges,colorbar=False,
                                saveplot=saveplots,pics_dir=output_dir,plot_title=title_phys_z.replace("/",""),
                                filename=filename,units=units)

    
    make_histos.plot_2dhist(x_data_nf,y_data_nf,var_names,ranges,colorbar=False,
                                saveplot=saveplots,pics_dir=output_dir,plot_title=title_nf.replace("/",""),
                                filename=filename,units=units)


    interactive(False)
    plt.show()