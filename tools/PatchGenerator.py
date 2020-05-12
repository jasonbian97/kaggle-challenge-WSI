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
import warnings
warnings.filterwarnings("ignore")
from tools.utils import *


args = {"ORIGINAL_CSV": "/mnt/ssd2/AllDatasets/ProstateDataset/Level1/train.csv",
        "DATASET_NAME": "Level1Ds2_128_rich",
        "SRC_DATASET": "/mnt/ssd2/AllDatasets/ProstateDataset/Level1",
        "DST_DATASET": "/mnt/ssd2/AllDatasets/ProstateDataset",
        "downscale":2,
        "block_size":128,
        "thresh_valid":0.6}

df = pd.read_csv(args["ORIGINAL_CSV"]).set_index('image_id')

TRAIN = os.path.join(args["SRC_DATASET"], "Train")
MASK = os.path.join(args["SRC_DATASET"], "Mask")

# set output file path
PROCESSED_DATASET_DIR = os.path.join(args["DST_DATASET"], args["DATASET_NAME"])
OUTPUT_LABEL_DIR = os.path.join(PROCESSED_DATASET_DIR, "Label")
if not os.path.exists(OUTPUT_LABEL_DIR):
    os.makedirs(OUTPUT_LABEL_DIR)

# parse args
block_size = args["block_size"]
thresh_valid = args["thresh_valid"]
downscale = args["downscale"]

def generate_patches_training(img_fname):

    # read img and gt_mask
    sample_train = os.path.join(TRAIN, img_fname)
    sample_mask = os.path.join(MASK, img_fname)

    train = io.imread(sample_train)
    mask = io.imread(sample_mask)

    if args["downscale"] > 1:
        train = cv2.resize(train, (train.shape[1] // downscale, train.shape[0] // downscale))
        mask = cv2.resize(mask, (mask.shape[1] // downscale, mask.shape[0] // downscale))

    # segment ROI
    mask_pred = segment_ROI(train)
    # padding
    img_pad = padding_2_multiple(mask_pred, block_size)

    img_slice_stack = blockshaped(img_pad, block_size, block_size)

    perc_foreground = np.sum(img_slice_stack.reshape(img_slice_stack.shape[0], -1), axis=-1) / (block_size * block_size)
    valid_index = np.where(perc_foreground > thresh_valid)[0]

    # if not many valid patch found, decrease threshold by 2
    if len(valid_index) < 4:
        valid_index = np.where(perc_foreground > thresh_valid * 0.8)[0]
        if len(valid_index) < 4:
            valid_index = np.where(perc_foreground > thresh_valid * 0.6)[0]
            if len(valid_index) == 0:
                valid_index = np.where(perc_foreground > thresh_valid * 0.3)[0]
        print("lower thresh for ", img_fname, "which finally have patches ", len(valid_index))

    save_slices_to_disk(train, mask, valid_index, block_size, PROCESSED_DATASET_DIR, img_fname)

    # compute stats for patches from mask
    pad_mask = padding_2_multiple(mask,block_size)
    mask_slice_stack = blockshaped(pad_mask, block_size, block_size)
    valid_mask_stack = mask_slice_stack[valid_index,:,:]

    # compute stats and write info to dict
    img_dict = {}
    img_dict["image_id"] = img_fname.split(".")[0]
    df_row = df.loc[img_fname.split(".")[0]]
    img_dict["data_provider"] = df.loc[img_fname.split(".")[0]]["data_provider"]
    img_dict["isup_grade"] = df.loc[img_fname.split(".")[0]]["isup_grade"]
    img_dict["gleason_score"] = df.loc[img_fname.split(".")[0]]["gleason_score"]
    img_dict["patch_size"] = block_size
    img_dict["thresh_valid"] = thresh_valid
    img_dict["stats"] = {"shape": train.shape[:2],
                         "non_bg_perc": (mask != 0).sum() / (mask.shape[0] * mask.shape[1])
                         }
    img_dict["patches_num"] = len(valid_index)
    img_dict["patches_id"] = np.arange(len(valid_index)) + 1
    img_dict["patches_loc"] = valid_index
    # compute_stats_for_patches
    stats = compute_stats_for_patches(valid_mask_stack, img_dict["data_provider"])
    img_dict["patches_stat"] = stats


    with open( OUTPUT_LABEL_DIR+"/{}.json".format(img_fname.split(".")[0]), "w") as json_file:
        # magic happens here to make it pretty-printed
        json_file.write(json.dumps(img_dict, indent=4, sort_keys=False, cls=MyEncoder))
        json_file.close()

    # return img_dict
















