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
from tools.PatchGenerator import *
# df = pd.DataFrame( columns = ["a", "b"])
# to_append = [5, 6]
# a_series = pd.Series(to_append, index = df.columns)
# df = df.append(a_series, ignore_index=True)
# print(df.head())
"""vis"""

# fig, ax = plt.subplots(1, 3)
# ax[0].imshow(train)
# ax[1].imshow(mask != 0)
# ax[2].imshow(mask_pred)
# plt.show()
import multiprocessing
import time



TRAIN = os.path.join(args["SRC_DATASET"],"Train")

TRAIN_LIST = os.listdir(TRAIN)



# starttime = time.time()
# for fname in TRAIN_LIST[:20]:
#     generate_patches_training(fname, df, args)
# print('That took {} seconds'.format(time.time() - starttime))

starttime = time.time()
pool = multiprocessing.Pool()
pool.map(generate_patches_training, TRAIN_LIST)
pool.close()
print('That took {} seconds'.format(time.time() - starttime))

# downscale = 2
# block_size = 224
# thresh_valid = 0.5
# ["07a14fa5b8f74272e4cc0a439dbc8f7f.jpg"]


# with open(os.path.join(OUTPUT_DIR,'iou_test.txt'), 'w') as f:
#     f.write("genrated mask iou test \n")
#     for item in outlier:
#         f.write("%s\n" % item)
#     f.write("===File End=== \n")
#
# print("meanIOU:",np.mean(iou_l))
# print("mean shape:{}x{}".format(np.mean(H),np.mean(W)))




