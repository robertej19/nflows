import pandas as pd

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

from utils import make_histos
import sys
import torch
from torch import nn
from torch import optim

import pandas as pd

run_1 = False
run_2 = False
run_3 = False
run_4 = False
run_5 = True
if run_1:
    df0 = pd.read_pickle("GenData0.pkl")
    df1 = pd.read_pickle("GenData1.pkl")
    dfs = [df0,df1]

    df = pd.concat(dfs)
    #print(df)
    df.hist()

    zX = df.to_numpy()

    bin_size = [100,100]


    xvals = df[1]
    x_name = "px E"
    output_dir = "./"

    make_histos.plot_1dhist(xvals,[x_name,],ranges="none",second_x="none",
                    saveplot=False,pics_dir=output_dir,plot_title=x_name)
                       
    # pairs = [(0,1),(1,2),(0,4),(8,12),(5,6),(9,10)]

    # for a,b in pairs:
    #     fig, ax = plt.subplots(figsize =(10, 7)) 
    #     plt.hist2d(zX[:,a], zX[:,b],bins =bin_size,norm=mpl.colors.LogNorm())# cmap = plt.cm.nipy_spectral) 
    #     #plt.xlim([-2,2])
    #     #plt.ylim([-2,2])
    #     #plt.colorbar()
    #     plotname = "finalplotname2.jpeg"

    #     plt.show()


if run_2:
    dfd = pd.read_pickle("data/pi0.pkl")
    df_small = dfd.head(100000)
    df_small.to_pickle("data/pi0_100k.pkl")

if run_3:
    dfd = pd.read_pickle("data/pi0_100k.pkl")

    df0 = pd.read_pickle("GenData0.pkl")
    df1 = pd.read_pickle("GenData1.pkl")
    dfX = [df0,df1]

    dfXX = pd.concat(dfX)
    zX = dfXX.to_numpy()


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
                    saveplot=True,pics_dir=output_dir,plot_title=x_name)


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
    

if run_4:
    #df0 = pd.read_pickle("GenData0.pkl")
    #df1 = pd.read_pickle("GenData1.pkl")
    #dfX = [df0,df1]

    #dfXX0 = pd.concat(dfX)
    #zX = dfXX.to_numpy()

    dfXX = pd.read_pickle("data/pi0_100k.pkl")

    dfXX = dfXX.head(10)

    e = 0
    px = e+1
    py = e+2
    pz = e+3
    print(dfXX)
    dfXX['pmass'] = np.sqrt(dfXX[e]**2-dfXX[px]**2-dfXX[py]**2-dfXX[pz]**2)

    print(dfXX.pmass.values)
    output_dir = '.'
    ranges = "none"
    #ranges = [-.51,.51,100]
    make_histos.plot_1dhist(dfXX['pmass'],["pmass",],ranges=ranges,second_x="none",
                saveplot=False,pics_dir=output_dir,plot_title="Pmass")
                 

if run_5:
    print("in run 5")
    class dataXZ:
        """
        read the data stored in pickle format
        the converting routine is at https://github.com/6862-2021SP-team3/hipo2pickle
        """
        def __init__(self, standard = False):
            with open('data/pi0.pkl', 'rb') as f:
                xz = np.array(pickle.load(f), dtype=np.float64)
                x = cartesian_converter(xz)
                xwithoutPid = x
                self.xz = xz
                self.x = torch.from_numpy(np.array(x))
                self.xwithoutPid = torch.from_numpy(xwithoutPid)

            if standard:
                self.standardize()

        def standardize(self):
            self.xMu = self.xwithoutPid.mean(0)
            self.xStd = self.xwithoutPid.std(0)
            self.zMu = self.zwithoutPid.mean(0)
            self.zStd = self.zwithoutPid.std(0)
            self.xwithoutPid = (self.xwithoutPid - self.xMu) / self.xStd
            self.zwithoutPid = (self.zwithoutPid - self.zMu) / self.zStd

        def restore(self, data, type = "x"):
            mu = self.xMu
            std = self.xStd
            if type == "z":
                mu = self.zMu
                std = self.zStd
            return data * std + mu

        def sample(self, n):
                randint = np.random.randint( self.xz.shape[0], size =n)
                xz = self.xz[randint]
                x = self.x[randint]
                xwithoutPid = self.xwithoutPid[randint]
                return {"xz":xz, "x": x, "xwithoutPid": xwithoutPid}

    def cartesian_converter(xznp):
        #split into electron, proton, gammas
        print("in cc")
        e_vec = xznp[:,1:5]
        p_vec = xznp[:,5:9]
        g1_vec = xznp[:,9:13]
        g2_vec = xznp[:,13:17]

        mass_e = .000511
        mass_p = 0.938
        mass_g = 0

        particles = [e_vec,p_vec,g1_vec,g2_vec]
        masses = [mass_e,mass_p,mass_g,mass_g]

        parts_new = []
        #convert from spherical to cartesian
        for part_vec, mass in zip(particles,masses):
            mom = part_vec[:,0]
            thet = part_vec[:,1]*np.pi/180
            phi = part_vec[:,2]*np.pi/180

            pz = mom*np.cos(thet)
            px = mom*np.sin(thet)*np.cos(phi)
            py = mom*np.sin(thet)*np.sin(phi)
            p2 = pz*pz+px*px+py*py
            E = np.sqrt(mass**2+p2)
            
            x_new = np.array([E,px,py,pz])

            mnew = E*E-px*px-py*py-pz*pz
            print(mnew)
            parts_new.append(x_new)

        #reshape output into 1x16 arrays for each event
        e = parts_new[0]
        p = parts_new[1]
        g1 = parts_new[2]
        g2 = parts_new[3]
        out = np.concatenate((e.T,p.T,g1.T,g2.T), axis=1)

        return out

    # Define device to be used
    #dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev = "cpu"
    device = torch.device(dev)
    print(dev)

    #read the data, with the defined data class
    xz = dataXZ()
    sampleDict = xz.sample(2) #Get a subset of the datapoints
    x = sampleDict["xwithoutPid"]
    print(x)
    dfx = pd.DataFrame(x.detach().numpy())

    e = 4
    dfx['emass'] = np.sqrt(dfx[e]**2-dfx[e+1]**2-dfx[e+2]**2-dfx[e+3]**2)
    print(dfx)

