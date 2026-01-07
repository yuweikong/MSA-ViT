from tkinter import Image
import numpy as np
import scipy.io as io
import PIL.Image as Image
import os
from scipy.interpolate import griddata


def norm(x):
    a=np.array((x - np.min(x)),dtype=np.float32)
    b=np.array((np.max(x) - np.min(x)),dtype=np.float32)
    return np.divide(a,b,np.zeros_like(a),where=b!=0)

os.chdir('D:/MSA-ViT')
dirs='data/source/evitau-gama-pr-sm.mat'

datas=io.loadmat(dirs)

mask_dirs='data/land_mask.mat'
mask_data = io.loadmat(mask_dirs)
mask = mask_data['land_mask'][21:,60:956]


evi = datas['input1']
tau = datas['input1_1']
gama=datas['input2']
pr=datas['input3']
sm=datas['output']
ndvi = io.loadmat('data/source/ndvi.mat')['ndvi']

data_misstake = []
sm_mask = sm<0

for i in range(gama.shape[-1]):
    gama1 = gama[:,:,i]
    gama1[np.isnan(gama1)] = 0
    if gama1.max() < 0.001:
        print('问题数据:',i)
        data_misstake.append(i)
        continue
    gama1[gama1<0]=0
    gama[:,:,i]=gama1
    # gama[:,:,i]=norm(gama1)

    pr1=pr[:,:,i]
    pr1[np.isnan(pr1)] = 0
    pr1[pr1<0]=0
    pr[:,:,i]=norm(pr1)

    tau1 = tau[:,:,i]
    tau1[np.isnan(tau1)] = 0
    tau1[tau1<0]=0
    # tau[:,:,i]=norm(tau1)
    tau[:,:,i]=tau1

    sm1 = sm[:,:,i]
    sm1[np.isnan(sm1)] = 0
    sm1[sm1<0]=0
    sm[:,:,i]=sm1

    evi1 = evi[:,:,i]
    evi1[np.isnan(evi1)] = 0
    evi1[evi1<-0.2]=-0.2
    # evi[:,:,i]=norm(evi1)
    evi[:,:,i]=evi1    

    ndvi1 = ndvi[:,:,i]
    ndvi1[np.isnan(ndvi1)] = 0
    ndvi1[ndvi1<-0.2]=-0.2
    # ndvi[:,:,i]=norm(ndvi1)
    ndvi[:,:,i]=ndvi1 

ndvi[sm_mask]=0
evi[sm_mask]=0
tau[sm_mask]=0
gama[sm_mask]=0
pr[sm_mask]=0


# 245->224 964->832   21:,100:932

savedir='data/train/input/'
savelabel='data/train/label/'
for i in range(24):
    if i in data_misstake:
        continue
    savename=savedir+str(i)+'.npy'
    img_cat = np.concatenate([gama[np.newaxis,:,:,i],pr[np.newaxis,:,:,i],
                              evi[np.newaxis,:,:,i]],axis=0)
    np.save(savename,img_cat)
    labelname=savelabel+str(i)+'.npy'
    np.save(labelname,sm[:,:,i])

savedir='data/test/input/'
savelabel='data/test/label/'
for i in range(24,46):
    if i in data_misstake:
        continue
    savename=savedir+str(i)+'.npy'
    img_cat = np.concatenate([gama[np.newaxis,:,:,i],pr[np.newaxis,:,:,i],
                              evi[np.newaxis,:,:,i]],axis=0)
    np.save(savename,img_cat)
    labelname=savelabel+str(i)+'.npy'
    np.save(labelname,sm[:,:,i])





