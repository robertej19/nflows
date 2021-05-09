
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

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.distributions.normal import DiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

import sys
sys.path.insert(0,'/mnt/c/Users/rober/Dropbox/Bobby/Linux/classes/GAML/GAMLX/nflows/nflows')

from nflows.transforms.autoregressive import MaskedUMNNAutoregressiveTransform

#Replace this with argparse!!!
email = True


#TODO:
#Time logging
#Quantile Distribution implementation
#Save generated data to pandas DF
#Do doodle poll
# Re open Iddo lecture notes

#Create data class
class dataXZ:
  """
  read the data stored in pickle format
  the converting routine is at https://github.com/6862-2021SP-team3/hipo2pickle
  """
  def __init__(self, standard = False):
    with open('data/pi0.pkl', 'rb') as f:
        xz = np.array(pickle.load(f), dtype=np.float32)
        #xz = xz[:, 1:]
        # z = xz[:, 16:]
        x = cartesian_converter(xz)
        # x = x[:, [0,4,8,12]]
        #x = x[:, [3,7,11,15]]
        #x = x[:, [0,1,2,3]]
        #x = xz[:, :16]
        #xwithoutPid = x[:, [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]]
        #xwithoutPid = x[:, [0,  4, 8, 12, ]]
        xwithoutPid = x
        #xwithoutPid = x[:, [0, 1, 4, 5, 8, 12, ]]
        # zwithoutPid = z[:, [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]]
        self.xz = xz
        self.x = torch.from_numpy(np.array(x))
        # self.z = torch.from_numpy(np.array(z))
        self.xwithoutPid = torch.from_numpy(xwithoutPid)
        # self.zwithoutPid = torch.from_numpy(zwithoutPid)

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
        # z = self.z[randint]
        xwithoutPid = self.xwithoutPid[randint]
        # zwithoutPid = self.zwithoutPid[randint]
        # return {"xz":xz, "x": x, "z": z, "xwithoutPid": xwithoutPid, "zwithoutPid": zwithoutPid}
        return {"xz":xz, "x": x, "xwithoutPid": xwithoutPid}

#returns an nx16 array, of energy, px, py, pz, for electron, proton, g1, g2
#You should just pass it the xz object from the dataXZ() class
def cartesian_converter(xznp):
  #split into electron, proton, gammas
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
    parts_new.append(x_new)

  #reshape output into 1x16 arrays for each event
  e = parts_new[0]
  p = parts_new[1]
  g1 = parts_new[2]
  g2 = parts_new[3]
  out = np.concatenate((e.T,p.T,g1.T,g2.T), axis=1)

  return out

# Define device to be used
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)
print(dev)

#read the data, with the defined data class
xz = dataXZ()
sampleDict = xz.sample(400000) #Get a subset of the datapoints
x = sampleDict["xwithoutPid"]
x = x.detach().numpy()

# #visualize the data
bin_size = [150,150]
fig, ax = plt.subplots(figsize =(10, 7)) 
plt.rcParams["font.size"] = "16"
ax.set_xlabel("Electron Momentum")  
ax.set_ylabel("Proton Momentum")
plt.title('Microphysics Simulated EP Distribution')

plt.hist2d(x[:,0], x[:,1],bins =bin_size,norm=mpl.colors.LogNorm())# cmap = plt.cm.nipy_spectral) 
#plt.xlim([1,6.5])
#plt.ylim([0.2,1.1])
#plt.colorbar()

#plt.show()
#sys.exit()
plt.savefig("slurm/figures/raw_distribution_01.pdf")

# fig, ax = plt.subplots(figsize =(10, 7)) 
# plt.rcParams["font.size"] = "16"
# ax.set_xlabel("Photon 1 Momentum")  
# ax.set_ylabel("Photon 2 Momentum")
# plt.title('Microphysics Simulated GG Distribution')
# plt.hist2d(x[:,2], x[:,3],bins =bin_size,norm=mpl.colors.LogNorm())# cmap = plt.cm.nipy_spectral) 
# plt.xlim([1,9])
# plt.ylim([0,5])
# plt.colorbar()
# plt.savefig("slurm/figures/raw_distribution_23.pdf")

#construct the model
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
optimizer = optim.Adam(flow.parameters())
print("number of params: ", sum(p.numel() for p in flow.parameters()))

def plot_histo_1D(real_vals, gen_vals, label_real="Physics Data", label_gen="NFlow Model", col2 = "blue",title="Physics vs NFlow Models", saveloc=None):
    fig, axes = plt.subplots(1, num_features, figsize=(4*5, 5))
    for INDEX, ax in zip((0, 1, 2,3 ), axes):
        _, bins, _ = ax.hist(real_vals[:, INDEX], bins=100, color = "red", label=label_real, density=True)
        ax.hist(gen_vals[:, INDEX], bins=bins, label=label_gen, color = col2,alpha=0.5, density=True)
        ax.legend(loc="lower left")
        ax.set_title("Feature {}".format(INDEX) )
    plt.tight_layout()
    if saveloc is not None: plt.savefig(saveloc)
    # plt.show()

def meter(dist1,dist2,feature):
  kld = entropy(dist1[:,feature],dist2[:,feature])
  emd = wasserstein_distance(dist1[:,feature],dist2[:,feature])
  jsd = distance.jensenshannon(dist1[:,feature],dist2[:,feature]) ** 2
  return [kld, emd, jsd]

num_iter = 4000

losses = []
f1_kd = []
f1_em = []
f1_js = []
f2_em = []
f3_em = []

start = datetime.now()
start_time = start.strftime("%H:%M:%S")
print("Start Time =", start_time)


for i in range(num_iter):
    # x, y = datasets.make_moons(12, noise=.1)
    # x = torch.tensor(x, dtype=torch.float32)
    # print(x)
    # print(y)
    
    sampleDict = xz.sample(1000)
    x = sampleDict["xwithoutPid"][:, 0:num_features].to(device)
    #y = sampleDict["xwithoutPid"][:, 1:2] 
    #print(x)
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x.float64()).mean()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())

    if ((i+1)%10) == 0:
      print("On step {} - loss {:.2f}, Current Running Time = {:.2f} seconds".format(i,loss.item(),i))#loss.item(),elapsedTime.total_seconds())) 
    
      now = datetime.now()
      elapsedTime = (now - start )
      print("Elapsed time is {}".format(elapsedTime))
      print("Rate is {} seconds per epoch".format(elapsedTime/i))
      print("Total estimated run time is {}".format(elapsedTime+elapsedTime/i*(num_iter+1-i)))

    if ((i+1)%100) == 0:
                    run_time = datetime.now()
                    elapsedTime = (run_time - start )
                    
                    bbb = 50000
                    torch.save(flow.state_dict(), "trainedmodel_{}_{:.2f}.pkl".format(i,loss.item()))
    #                 z1= flow.sample(200).cpu().detach().numpy()
    #                 print("samp 1")
    #                 z2= flow.sample(200).cpu().detach().numpy()
    #                 print("samp 2")

    #                 z3= flow.sample(200).cpu().detach().numpy()
    #                 print("samp 3")

    #                 z4= flow.sample(200).cpu().detach().numpy()
    #                 print("samp 4")

    #                 z = np.concatenate((z1,z2,z3,z4),axis=0)
    #                 # #print(z)
    #                 # #sys.exit()
    #                 # print('gen z done')

    #                 # sampleDict = xz.sample(5)
    #                 # x = sampleDict["x"][:, 0:num_features] 
    #                 # x = x.detach().numpy()

    #                 # #plot_histo_1D(x,z)

    #                 # #f1 = meter(x,z,0)
    #                 # #f2 = meter(x,z,1)
    #                 # #f3 = meter(x,z,2)
    #                 # #f4 = meter(x,z,3)

    #                 bin_size = [100,100]
    #                 fig, ax = plt.subplots(figsize =(10, 7)) 
    #                 plt.rcParams["font.size"] = "16"
    #                 ax.set_xlabel("Electron Momentum x")  
    #                 ax.set_ylabel("Electron Momentum y")
    #                 plt.title('NFlow Generated Distribution - Iteration {}'.format(i))

    #                 plt.hist2d(z[:,1], z[:,2],bins =bin_size,norm=mpl.colors.LogNorm())# cmap = plt.cm.nipy_spectral) 
    #                 #plt.xlim([-2,2])
    #                 #plt.ylim([-2,2])
    #                 #plt.colorbar()
    #                 titlenum = i
    #                 if titlenum < 10:
    #                   plotname = "pics/iter_00{}.jpeg".format(i)
    #                 elif titlenum < 100:
    #                   plotname = "pics/iter_0{}.jpeg".format(i)
    #                 else:
    #                   plotname = "pics/iter_{}.jpeg".format(i)
    #                 #plt.show()
    #                 plt.savefig(plotname)
    #                 plt.close()


                    # #if f1[1]*f2[1]*f3[1]*f4[1] < 1:
                    # #print("EM Distance   Values: F0: {:.5f}  F1: {:.5f}  F2: {:.5f} F3: {:.5f} ".format((f1[1]),(f2[1]),(f3[1]),(f4[1]),))
                    # #print("EM Distance   Values: F0: {:.5f}  F1: {:.5f}  F2: {:.5f} F3: {:.5f} ".format((f1[1]),(f2[1]),(f2[1]),(f2[1]),))
                    # #if f1[1]*f2[1] < .001:
                    #   #break

                    # #f1_kd.append(f1[0])
                    # #f1_em.append(f1[1])
                    # #f1_js.append(f1[2])
                    # #f2_em.append(f2[1])

tm_name = "trainedmodel_final.pkl"
torch.save(flow.state_dict(), tm_name)
print("trained model saved to {}".format(tm_name))

z0= flow.sample(2000).cpu().detach().numpy()
z1= flow.sample(2000).cpu().detach().numpy()
z2= flow.sample(2000).cpu().detach().numpy()
z3= flow.sample(2000).cpu().detach().numpy()
z4= flow.sample(2000).cpu().detach().numpy()
z5= flow.sample(2000).cpu().detach().numpy()
z6= flow.sample(2000).cpu().detach().numpy()
z7= flow.sample(2000).cpu().detach().numpy()
z8= flow.sample(2000).cpu().detach().numpy()
z9= flow.sample(2000).cpu().detach().numpy()

zX = np.concatenate((z0,z1,z2,z3,z4,z5,z6,z7,z8,z9),axis=0)
#print(z)
#sys.exit()
print('gen z final done')

#sampleDict = xz.sample(5)
#x = sampleDict["x"][:, 0:num_features] 
#x = x.detach().numpy()

#plot_histo_1D(x,z)

#f1 = meter(x,z,0)
#f2 = meter(x,z,1)
#f3 = meter(x,z,2)
#f4 = meter(x,z,3)

bin_size = [100,100]
fig, ax = plt.subplots(figsize =(10, 7)) 
plt.rcParams["font.size"] = "16"
ax.set_xlabel("Electron Momentum")  
ax.set_ylabel("Proton Momentum")
plt.title('NFlow Generated Distribution - Iteration')

plt.hist2d(zX[:,1], zX[:,2],bins =bin_size,norm=mpl.colors.LogNorm())# cmap = plt.cm.nipy_spectral) 
#plt.xlim([-2,2])
#plt.ylim([-2,2])
#plt.colorbar()
plotname = "finalplotname.jpeg"
# titlenum = i
# if titlenum < 10:
#   plotname = "pics/iter_00{}.jpeg".format(i)
# elif titlenum < 100:
#   plotname = "pics/iter_0{}.jpeg".format(i)
# else:
#   plotname = "pics/iter_{}.jpeg".format(i)
plt.show()
plt.savefig(plotname)
plt.close()




now = datetime.now()
end_time = now.strftime("%H:%M:%S")
print("End Time =", end_time)
elapsedTime = (now - start_now )
print("Total Run Time = {:.5f} seconds".format(elapsedTime.total_seconds()))
    # if (i + 1) % 50 == 0:
    #     xline = torch.linspace(-1.5, 2.5)
    #     yline = torch.linspace(-.75, 1.25)
    #     xgrid, ygrid = torch.meshgrid(xline, yline)
    #     xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    #     with torch.no_grad():
    #         zgrid = flow.log_prob(xyinput).exp().reshape(100, 100)

    #     plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
    #     plt.title('iteration {}'.format(i + 1))
    #     plt.show()

#f1_kd = []
#f1_em = []
#f1_js = []

fig, ax = plt.subplots(figsize =(10, 7)) 
#print(np.arange(len(losses)))
plt.rcParams["font.size"] = "16"

plt.plot(np.arange(len(f1_em)),f1_em, '-b',label="Feature 0")
plt.plot(np.arange(len(f1_em)),f2_em, '-g',label="Feature 1")
#plt.plot(np.arange(len(f1_em)),f3_em, '-r',label="Feature 2")
#plt.ylim([1000000000,0.0001])
ax.set_yscale('log')
plt.title('Wasserstein-1 Distance vs. Training Step')
ax.legend()
ax.set_xlabel("Training Step")  
ax.set_ylabel("Earth-Mover Distance")
plt.savefig("slurm/figures/EMD_training.pdf")


fig, ax = plt.subplots(figsize =(10, 7)) 
#print(np.arange(len(losses)))
plt.rcParams["font.size"] = "16"

#plt.scatter(np.arange(len(f1_em)),f3_em, c='b', s=20)
#plt.ylim([1000000000,0.0001])
ax.set_yscale('log')
plt.title('Loss vs. Training Step')
ax.set_xlabel("Training Step")  
ax.set_ylabel("Loss")

fig, ax = plt.subplots(figsize =(10, 7)) 
#print(np.arange(len(losses)))
plt.rcParams["font.size"] = "16"

plt.scatter(np.arange(len(f1_js)),f1_js, c='g', s=20)
#plt.ylim([1000000000,0.0001])
#ax.set_yscale('log')
plt.title('Jensen–Shannon Divergence vs. Training Step')
ax.set_xlabel("Training Step")  
ax.set_ylabel("Jensen–Shannon Divergence")
plt.savefig("slurm/figures/JSD_training.pdf")

fig, ax = plt.subplots(figsize =(10, 7)) 
#print(np.arange(len(losses)))
plt.rcParams["font.size"] = "16"

plt.scatter(np.arange(len(f1_kd)),f1_kd, c='g', s=20)
#plt.ylim([1000000000,0.0001])
#ax.set_yscale('log')
plt.title('Kullback–Leibler Divergence vs. Training Step')
ax.set_xlabel("Training Step")  
ax.set_ylabel("Kullback–Leibler Divergence")
plt.savefig("slurm/figures/KLD_training.pdf")

#Testing

aa = flow.sample(100000).cpu().detach().numpy()
# plt.scatter(aa[:,0], aa[:,1], c='r', s=5, alpha=0.5)
# plt.savefig("test_sample.pdf")

z = aa

bbb = 100000
z= flow.sample(bbb).cpu().detach().numpy()
sampleDict = xz.sample(bbb)
sampleDict2 = xz.sample(bbb)
y = sampleDict2["x"]
y = y.detach().numpy()
x = sampleDict["x"]
x = x.detach().numpy()

plot_histo_1D(x,z, saveloc="slurm/figures/testing_xz.pdf")
plot_histo_1D(x,y,label_real="Physics Sample 1", label_gen="Physics Sample 2",col2="green", saveloc="slurm/figures/testing_xy.pdf")

f1 = meter(x,z,0)
f2 = meter(x,z,1)
#f3 = meter(x,z,2)
#f4 = meter(x,z,3)
"""
print("Values for Physics Data vs. NFlow Model:")
print("KL Divergence Values: F0: {:.5f}  F1: {:.5f}  F2: {:.5f} ,F3: {:.5f}  ".format((f1[0]),(f2[0]),(f3[0]),(f4[0])))
print("EM Distance   Values: F0: {:.5f}  F1: {:.5f}  F2: {:.5f} ,F3: {:.5f} ".format((f1[1]),(f2[1]),(f3[1]),(f4[1])))
print("JS Divergence Values: F0: {:.5f}  F1: {:.5f}  F2: {:.5f}  ,F3: {:.5f} ".format((f1[2]),(f2[2]),(f3[2]),(f4[2])))
print('\n')

f1 = [i / j for i, j in zip(f1,meter(x,y,0))]
f2 = [i / j for i, j in zip(f2,meter(x,y,1))]
f3 = [i / j for i, j in zip(f3,meter(x,y,2))]
f4 = [i / j for i, j in zip(f4,meter(x,y,3))]

print("Ratio of KL, EM, and JS values from NFlow comparision and two physics model samples:")
print("KL Divergence Ratio: F0: {:.5f}  F1: {:.5f}  F2: {:.5f} ,F3: {:.5f}  ".format((f1[0]),(f2[0]),(f3[0]),(f4[0])))
print("EM Distance   Ratio: F0: {:.5f}  F1: {:.5f}  F2: {:.5f} ,F3: {:.5f} ".format((f1[1]),(f2[1]),(f3[1]),(f4[1])))
print("JS Divergence Ratio: F0: {:.5f}  F1: {:.5f}  F2: {:.5f}  ,F3: {:.5f} ".format((f1[2]),(f2[2]),(f3[2]),(f4[2])))
print('\n')

f1 = meter(x,y,0)
f2 = meter(x,y,1)
f3 = meter(x,y,2)
f4x = meter(x,y,3)

print("Values for two samples from physics data")
print("KL Divergence Values: F0: {:.5f}  F1: {:.5f}  F2: {:.5f} ,F3: {:.5f}  ".format((f1[0]),(f2[0]),(f3[0]),(f4x[0])))
print("EM Distance   Values: F0: {:.5f}  F1: {:.5f}  F2: {:.5f} ,F3: {:.5f} ".format((f1[1]),(f2[1]),(f3[1]),(f4x[1])))
print("JS Divergence Values: F0: {:.5f}  F1: {:.5f}  F2: {:.5f}  ,F3: {:.5f} ".format((f1[2]),(f2[2]),(f3[2]),(f4x[2])))
"""


if email:
  import os
  from pytools import circle_emailer

  now = datetime.now()
  script_end_time = now.strftime("%H:%M:%S")
  s_name = os.path.basename(__file__)
  subject = "Completion of {}".format(s_name)
  body = "Your script {} finished running at {}".format(s_name,script_end_time)
  circle_emailer.send_email(subject,body)
