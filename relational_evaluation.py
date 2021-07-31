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

#data_path = "gendata/4features/" #Just electorn features
#data_path = "gendata/16features/" #All 16 features
#data_path = "gendata/Cond/16features/maaf/"
#data_path = "gendata/Cond/16features/UMNN/"
#data_path = "gendata/Relational/photon1_1M/"
#data_path = "gendata/Relational/QT/photon_1/"
data_path = "gendata/Relational/QT/INV/photon_1/"





dfs = []
    
filenames = os.listdir(data_path)

for f in filenames:
    df0 = pd.read_pickle(data_path+f)
    dfs.append(df0)

df_photon1 = pd.concat(dfs)

#df_photon1.columns = ["gen_E_1","gen_Px_1","gen_Py_1","gen_Pz_1","recon_E_1","recon_Px_1","recon_Py_1","recon_Pz_1","nf_E_1","nf_Px_1","nf_Py_1","nf_Pz_1"]

#For forward:
#df_photon1.columns = ["gen_Px_1","gen_Py_1","gen_Pz_1","recon_Px_1","recon_Py_1","recon_Pz_1","nf_Px_1","nf_Py_1","nf_Pz_1"]
#For Reverse:
df_photon1.columns = ["recon_Px_1","recon_Py_1","recon_Pz_1","gen_Px_1","gen_Py_1","gen_Pz_1","nf_Px_1","nf_Py_1","nf_Pz_1"]

print("The Generated dataset has {} events".format(len(df_photon1)))



# plt.hist(df_photon1["gen_Py_1"],color = "red", density=True,bins=100)
# ##plt.savefig(outdir+"feature0"+"_noQT")
# #plt.close()
# plt.show()


# bin_size = [200,200]


# fig, ax = plt.subplots(figsize =(5, 3)) 
# plt.hist2d(df_photon1["gen_Py_1"], df_photon1["gen_Px_1"],bins =bin_size,norm=mpl.colors.LogNorm())# cmap = plt.cm.nipy_spectral) 
# plt.colorbar()
# plt.show()

# sys.exit()


#data_path = "gendata/Relational/photon2_1M/"
#data_path = "gendata/Relational/QT/photon_2/"
data_path = "gendata/Relational/QT/INV/photon_2/"

dfs = []
    
filenames = os.listdir(data_path)

for f in filenames:
    df0 = pd.read_pickle(data_path+f)
    dfs.append(df0)

df_photon2 = pd.concat(dfs)
#df_photon2.columns = ["gen_E_2","gen_Px_2","gen_Py_2","gen_Pz_2","recon_E_2","recon_Px_2","recon_Py_2","recon_Pz_2","nf_E_2","nf_Px_2","nf_Py_2","nf_Pz_2"]
# #For Forward
#df_photon2.columns = ["gen_Px_2","gen_Py_2","gen_Pz_2","recon_Px_2","recon_Py_2","recon_Pz_2","nf_Px_2","nf_Py_2","nf_Pz_2"]
# #For Inverse:
df_photon2.columns = ["recon_Px_2","recon_Py_2","recon_Pz_2","gen_Px_2","gen_Py_2","gen_Pz_2","nf_Px_2","nf_Py_2","nf_Pz_2"]



print("The Generated dataset has {} events".format(len(df_photon2)))


physics_cuts = False
gen_all_emd = False
gen_1d_histos = True
gen_emd_comp = False

if len(df_photon1) > len(df_photon2):
    df_photon1 = df_photon1.head(len(df_photon2))
else:
    df_photon2 = df_photon2.head(len(df_photon1))


df_ps = pd.concat([df_photon1,df_photon2],axis=1)

ic(df_ps)
ic(df_ps.columns)

def dot(vec1, vec2):
    # dot product of two 3d vectors
    return vec1[0]*vec2[0]+vec1[1]*vec2[1]+vec1[2]*vec2[2]
def mag(vec1):
    # L2 norm of vector
    return np.sqrt(dot(vec1, vec1))

gam1 = [df_ps['nf_Px_1'],df_ps['nf_Py_1'],df_ps['nf_Pz_1']]
gam2 = [df_ps['nf_Px_2'],df_ps['nf_Py_2'],df_ps['nf_Pz_2']]
pi0 = [df_ps['nf_Px_1']+df_ps['nf_Px_2'],df_ps['nf_Py_1']+df_ps['nf_Py_2'],df_ps['nf_Pz_1']+df_ps['nf_Pz_2']]
df_ps.loc[:,"nf_Mpi0"] = np.sqrt((mag(gam1)+mag(gam2))**2 - mag(pi0)**2)


gam1 = [df_ps['recon_Px_1'],df_ps['recon_Py_1'],df_ps['recon_Pz_1']]
gam2 = [df_ps['recon_Px_2'],df_ps['recon_Py_2'],df_ps['recon_Pz_2']]
pi0 = [df_ps['recon_Px_1']+df_ps['recon_Px_2'],df_ps['recon_Py_1']+df_ps['recon_Py_2'],df_ps['recon_Pz_1']+df_ps['recon_Pz_2']]
df_ps.loc[:,"recon_Mpi0"] = np.sqrt((mag(gam1)+mag(gam2))**2 - mag(pi0)**2)


gam1 = [df_ps['gen_Px_1'],df_ps['gen_Py_1'],df_ps['gen_Pz_1']]
gam2 = [df_ps['gen_Px_2'],df_ps['gen_Py_2'],df_ps['gen_Pz_2']]
pi0 = [df_ps['gen_Px_1']+df_ps['gen_Px_2'],df_ps['gen_Py_1']+df_ps['gen_Py_2'],df_ps['gen_Pz_1']+df_ps['gen_Pz_2']]
df_ps.loc[:,"gen_Mpi0"] = np.sqrt((mag(gam1)+mag(gam2))**2 - mag(pi0)**2)


##df_ps = df_ps.query("nf_Mpi0<.15")
#df_ps = df_ps.query("nf_Mpi0>0.12")
# df_ps = df_ps.query("nf_Pz_1-recon_Pz_1<0.01")
# df_ps = df_ps.query("nf_Pz_1-recon_Pz_1>-0.01")
# df_ps = df_ps.query("nf_Pz_2-recon_Pz_2<0.01")
# df_ps = df_ps.query("nf_Pz_2-recon_Pz_2>-0.01")


df = df_ps


names = ["Px_1","Py_1","Pz_1","Px_2","Py_2","Pz_2","Mpi0"]
for name in names:
    make_histos.plot_2dhist(df["gen_{}".format(name)],df["gen_{}".format(name)]-df["recon_{}".format(name)],["gen_{}".format(name),"gen - recon: {}".format(name)],colorbar=True,
            saveplot=True,pics_dir="julypics/",plot_title="none",
           filename="gen-recon_{}".format(name),units=["GeV","GeV"])
    make_histos.plot_2dhist(df["gen_{}".format(name)],df["gen_{}".format(name)]-df["nf_{}".format(name)],["gen_{}".format(name),"gen - nf: {}".format(name)],colorbar=True,
            saveplot=True,pics_dir="julypics/",plot_title="none",
           filename="gen-recon_{}",units=["GeV","GeV"])
   # plt.scatter(df["recon_{}".format(name)],df["recon_{}".format(name)]-df["nf_{}".format(name)])
   # plt.scatter(df["recon_{}".format(name)],df["nf_{}".format(name)])
    #plt.title("Recon vs. NF, {}".format(name))
    #plt.xlabel("Recon")
    #plt.ylabel("NF")
    #plt.show()



ic(df_ps.mean())

df_ps.to_pickle("4_feature_pion_QT_INV.pkl")

make_histos.plot_1dhist(df_ps['recon_Mpi0'],['nf pion mass',],ranges=[0.02,0.4,150],
                            second_x=df_ps['nf_Mpi0'],first_color='red')


 # #         make_histos.plot_1dhist(xvals_1,[x_name,],ranges="none",second_x=xvals_2,
    # #                 saveplot=True,pics_dir=output_dir,plot_title="{}, NF 4-Feature Model".format(x_name),density=True,
    # #                 annotation=emd_nflow,first_color="red",xlabel_1="Microphysics Data",xlabel_2="NF Model Sample")
    
    # dvpi0p = df_nflow_data
    # #dvpi0p = df_test_data

    # e=4
    # dvpi0p.loc[:,'pmass'] = np.sqrt(dvpi0p[e]**2-dvpi0p[e+1]**2-dvpi0p[e+2]**2-dvpi0p[e+3]**2)/0.938


   
    # dvpi0p.loc[:, "Gpx"] = dvpi0p.loc[:, 9]
    # dvpi0p.loc[:, "Gpy"] = dvpi0p.loc[:, 10]
    # dvpi0p.loc[:, "Gpz"] = dvpi0p.loc[:, 11]
    # dvpi0p.loc[:, "Gpx2"] = dvpi0p.loc[:, 13]
    # dvpi0p.loc[:, "Gpy2"] = dvpi0p.loc[:, 14]
    # dvpi0p.loc[:, "Gpz2"] = dvpi0p.loc[:, 15]
    # gam1 = [dvpi0p['Gpx'], dvpi0p['Gpy'], dvpi0p['Gpz']]
    # gam2 = [dvpi0p['Gpx2'], dvpi0p['Gpy2'], dvpi0p['Gpz2']]

    # pi0 = [dvpi0p['Gpx']+dvpi0p['Gpx2'], dvpi0p['Gpy']+dvpi0p['Gpy2'], dvpi0p['Gpz']+dvpi0p['Gpz2']]


    # dvpi0p.loc[:, "Mpi0"] = np.sqrt((mag(gam1)+mag(gam2))**2 - mag(pi0)**2)
    
    

    # df = dvpi0p
