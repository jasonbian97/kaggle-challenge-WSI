import pandas as pd
import skimage.io as io
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
import  matplotlib
matplotlib.use("TkAgg")
import numpy as np
from tqdm import tqdm
import skimage.morphology as morphology
import skimage.filters as filter
from matplotlib import pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean
import cv2
import os
import json


def segment_ROI(img,threshold = (235,235,235)):
    mask = np.logical_and(img[:,:,0]<threshold[0], img[:,:,1]<threshold[1],img[:,:,2]<threshold[2]).astype(np.uint8)
    # downsample --> dilation --> upsample
    mask = cv2.resize(mask, dsize=(img.shape[1]//4,img.shape[0]//4), interpolation=cv2.INTER_NEAREST)
    # morphology
    kernel_size = max(5,min(img.shape[0]//100,img.shape[1]//100))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel_size,kernel_size),np.uint8))
    mask = cv2.resize(mask, dsize=(img.shape[1],img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask

def IoU(pred,gt):
    Union = np.logical_or(pred.flatten(),gt.flatten())
    Intersection = np.logical_and(pred.flatten(),gt.flatten())
    return Intersection.sum()/Union.sum()

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def save_slices_to_disk(image,mask, indices,block_size,output_dir,img_fname):
    ""
    TRAIN = os.path.join(output_dir,"train")
    MASK = os.path.join(output_dir, "mask")
    if not os.path.exists(TRAIN):
        os.makedirs(TRAIN)
    if not os.path.exists(MASK):
        os.makedirs(MASK)

    pad_image = padding_2_multiple(image,block_size)
    pad_mask = padding_2_multiple(mask,block_size)

    num_col = image.shape[1] // block_size + 1
    for id,index in enumerate(list(indices)):
        start_row = index // num_col
        start_col = index % num_col
        patch = pad_image[start_row*block_size:(start_row + 1 )*block_size,
                start_col * block_size:(start_col + 1) * block_size, :]
        patch_mask = pad_mask[start_row*block_size:(start_row + 1 )*block_size,
                start_col * block_size:(start_col + 1) * block_size] * 40

        io.imsave(TRAIN + "/{}-{}.jpg".format(img_fname.split(".")[0], id+1), patch)
        io.imsave(MASK + "/{}-{}.jpg".format(img_fname.split(".")[0], id+1), patch_mask)

def padding_2_multiple(image,multiple):
    img_pad = None
    pad_row = (image.shape[0] // multiple + 1) * multiple
    pad_col = (image.shape[1] // multiple + 1) * multiple
    if len(image.shape) == 2:
        img_pad = np.zeros((pad_row, pad_col),dtype=np.uint8)
        img_pad[:image.shape[0], :image.shape[1]] = image
    elif len(image.shape) == 3:
        # if this is image, then padding with white background
        img_pad = np.ones((pad_row, pad_col,3),dtype=np.uint8)*255
        img_pad[:image.shape[0], :image.shape[1],:] = image
    else:
        exit()
    return img_pad

def compute_stats_for_patches(patches,data_provider):
    stats = {}
    patches = patches.reshape(patches.shape[0],-1)
    stats["bg_perc"] = np.sum(patches==0,axis=1)/patches.shape[1]
    if data_provider == "karolinska":
        stats["benign_tissue_perc"] = np.sum(patches==1,axis=1)/patches.shape[1]
        stats["cancerous_tissue_perc"] = np.sum(patches==2,axis=1)/patches.shape[1]
    elif data_provider == "radboud":
        stats["stroma_perc"] = np.sum(patches == 1, axis=1) / patches.shape[1]
        stats["benign_epi_perc"] = np.sum(patches == 2, axis=1) / patches.shape[1]
        stats["Gleason_3_perc"] = np.sum(patches == 3, axis=1) / patches.shape[1]
        stats["Gleason_4_perc"] = np.sum(patches == 4, axis=1) / patches.shape[1]
        stats["Gleason_5_perc"] = np.sum(patches == 5, axis=1) / patches.shape[1]
    else:
        print("Wrong key")
        exit()
    return stats

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
