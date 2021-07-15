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

# # df_16 = pd.read_pickle("16_feature_pion.pkl")
# # df_4 = pd.read_pickle("4_feature_pion.pkl")


# # if len(df_16) > len(df_4):
# #     df_16 = df_16.head(len(df_4))
# # else:
# #     df_4 = df_4.head(len(df_16))




# # bin_size = [100,100]

# # var_name = 'Mpi0'
# # xvals = df_16[var_name]
# # xvals = df_4['recon_Mpi0']
# # xvals2 = df_4['nf_Mpi0']
# # ###############





# # #x_name = "Gamma-Gamma Invariant Mass (GeV)"
# # x_name = "Reconstructed Pion Mass (GeV)"
# # output_dir = "./"
# # #ranges = "none"
# # #ranges = [12,28,100]
# # ranges = [.02,.3,100]
# # print("PLOTTTING")
# # make_histos.plot_1dhist(xvals,[x_name,],ranges=ranges,second_x=xvals2,annotation="yes",
# #                     saveplot=False,pics_dir=output_dir,plot_title="Reconstructed Pion Mass, NF Sampled",
# #                     density=True,proton_line=0.135,first_color="green",xlabel_1="Physics Recon. Data",xlabel_2="4-Feature Model")



df_4 = pd.read_pickle("4_feature_pion.pkl")
ic(df_4)

#df = df_4.head(50000)
df = df_4





df.loc[:,"Pz_1_diff"] = np.sqrt(np.square(df["recon_Pz_1"]-df["nf_Pz_1"]))
df.loc[:,"Pz_2_diff"] = np.sqrt(np.square(df["recon_Pz_2"]-df["nf_Pz_2"]))

x_name ="diffs"
# make_histos.plot_1dhist(df['Pz_1_diff'],[x_name,],
#                     saveplot=False,plot_title="Reconstructed Pion Mass, NF Sampled",
#                     density=True,proton_line=0.135,first_color="green",xlabel_1="Physics Recon. Data",xlabel_2="4-Feature Model")

# make_histos.plot_1dhist(df['Pz_2_diff'],[x_name,],
#                     saveplot=False,plot_title="Reconstructed Pion Mass, NF Sampled",
#                     density=True,proton_line=0.135,first_color="green",xlabel_1="Physics Recon. Data",xlabel_2="4-Feature Model")


df = df.query("Pz_2_diff<0.05")
df = df.query("Pz_1_diff<0.05")
# df = df.query("nf_Mpi0<0.16")
# df = df.query("nf_Mpi0>0.12")
ic(df)
bin_size = [100,100]

# # names = ["Px_1","Py_1","Pz_1","Px_2","Py_2","Pz_2","Mpi0"]
# # for name in names:
# #     make_histos.plot_2dhist(df["recon_{}".format(name)],df["nf_{}".format(name)],["recon_{}".format(name),"nf_{}".format(name)],colorbar=True,
# #             saveplot=True,pics_dir="linear_pics/",plot_title="none",
# #             filename="ExamplePlot",units=["GeV","GeV"])
# #     #plt.scatter(df["recon_{}".format(name)],df["recon_{}".format(name)]-df["nf_{}".format(name)])
# #     # plt.scatter(df["recon_{}".format(name)],df["nf_{}".format(name)])
# #     # plt.title("Recon vs. NF, {}".format(name))
# #     # plt.xlabel("Recon")
# #     # plt.ylabel("NF")
# #     # plt.show()
# #     #plt.savefig("linear_pics/{}".format(name))

if len(df) > len(df_4):
    df = df.head(len(df_4))
else:
    df_4 = df_4.head(len(df))


var_name = 'Mpi0'
#xvals = df['recon_Mpi0']
xvals = df_4['nf_Mpi0']
xvals2 = df['nf_Mpi0']
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
                    density=True,proton_line=0.135,first_color="green",xlabel_1="No Cuts",xlabel_2="Pz Difference Cut")




# plt.scatter(df["gen_Px_1"],df["gen_Py_1"])
# plt.show()
# plt.scatter(df["nf_Px_1"],df["recon_Px_1"])
# plt.show()
# plt.scatter(df["nf_Py_1"],df["recon_Py_1"])
# plt.show()