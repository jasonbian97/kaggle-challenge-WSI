from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as npc
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def HETransform(img,sigma=0.5):
    DeconvMatrix = np.array([[1.88, -0.07, -0.6],
                             [-1.02, 1.13, -0.48],
                             [-0.55, -0.13, 1.57]])
    factor = np.random.normal(1.,sigma)
    # decompose: he = D*rgb
    HEcomponent = np.dot(DeconvMatrix, img.reshape(-1, 3).T).T.reshape(img.shape[0], img.shape[1], 3)
    # enhance Esoin
    HEcomponent[:, :, 1] = HEcomponent[:, :, 1] * factor
    # recons
    recons = np.dot(np.linalg.inv(DeconvMatrix), HEcomponent.reshape(-1, 3).T).T.reshape(128, 128, 3)
    recons = np.maximum(recons, 0)
    recons = recons / recons.max() * 255
    recons = recons.astype(np.uint8)
    return recons


TRAIN = "/mnt/ssd2/AllDatasets/ProstateDataset/Level1_128_rich/train"
trainlist = os.listdir(TRAIN)
sample  = trainlist[2]
img = io.imread(os.path.join(TRAIN,sample))
shape = img.shape
recons = HETransform(img)
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].imshow(img)
ax[1].imshow(recons)
plt.show()

# print(trainlist[:5])
#

#
# DeconvMatrix = np.array([[1.88,-0.07,-0.6],
#                          [-1.02,1.13,-0.48],
#                          [-0.55,-0.13,1.57]])
# # decompose: he = D*rgb
# HEcomponent = np.dot(DeconvMatrix,img.reshape(-1,3).T).T.reshape(shape[0],shape[1],3)
#
# fig, ax = plt.subplots(nrows=2, ncols=3)
#
# ax.flat[0].imshow(HEcomponent[:,:,0],cmap='gray') # Hematoxylin
# ax.flat[0].set_title("Hematoxylin")
# ax.flat[1].imshow(HEcomponent[:,:,1],cmap='gray') # Eosin
# ax.flat[1].set_title("Eosin")
# ax.flat[2].imshow(HEcomponent[:,:,2],cmap='gray') # DAB
# ax.flat[2].set_title("DAB")
# ax.flat[3].imshow(img)
# ax.flat[3].set_title("original")
#
# # recons
# recons = np.dot(np.linalg.inv(DeconvMatrix),HEcomponent.reshape(-1,3).T).T.reshape(128,128,3).astype(np.uint8)
# ax.flat[4].imshow(recons)
# ax.flat[4].set_title("recons")
#
# # enhance
# factor = 0.7
# HEcomponent[:,:,1] = HEcomponent[:,:,1] * factor
# recons = np.dot(np.linalg.inv(DeconvMatrix),HEcomponent.reshape(-1,3).T).T.reshape(128,128,3)
# recons = np.maximum(recons,0)
# recons = recons/recons.max() * 255
# recons  = recons.astype(np.uint8)
#
# ax.flat[5].imshow(recons)
# ax.flat[5].set_title("recons_enhance")
# plt.show()
# print()


