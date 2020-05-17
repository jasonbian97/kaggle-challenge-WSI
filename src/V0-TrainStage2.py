import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import os
from torchvision import datasets, transforms
from torch.optim import Adam
from pytorch_lightning import Trainer
from argparse import ArgumentParser
import torchvision.models as models
import torch
import torchvision
from tqdm.auto import tqdm
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import numpy as np
from datetime import datetime
import pandas as pd
import random
from torchvision.datasets import ImageFolder
import re
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
from sklearn.metrics import roc_auc_score
from skimage.io import imread, imsave
import skimage
from PIL import ImageFile
from PIL import Image
import json
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateLogger
from sklearn.metrics import confusion_matrix,f1_score
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import KFold
from mish_activation import Mish
import Dataset as MyDataset
import utils as utils

class V0_Stage2_System(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # parser.add_argument('--batch_size', type=int, default=32)
        # parser.add_argument('--learning_rate', type=float, default= 1e-4)
        return parser

    def __init__(self, hparams, train_list, val_list):
        super().__init__()

        self.hparams = hparams

        if hparams.arch.split("-")[0] == "efficientnet":
            if hparams.load_pretrained:
                self.enc = EfficientNet.from_pretrained(hparams.arch)
                out_features = self.enc.extract_features(torch.rand((1, 3, 128, 128))).shape[1]

        self.maxpool_branch = nn.AdaptiveMaxPool2d(1)
        self.avgpool_branch = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(2 * out_features, 512),
                                  nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, hparams.num_class))
        # filter out Benign patches
        print("filtering out Benign patches...")
        self.train_list = self.filter_out_benign(train_list)
        self.val_list = self.filter_out_benign(val_list)
        print("Num_Train = ", len(self.train_list))
        print("Num_Val = ", len(self.val_list))

    def forward(self,x):
        if self.hparams.arch.split("-")[0] == "efficientnet":
            x = self.enc.extract_features(x)
        else:
            x = self.enc(x)
        x = torch.cat([self.avgpool_branch(x),self.maxpool_branch(x)],dim=1)
        x = self.head(x)
        return x

    def train_dataloader(self):
        # transforms
        train_transformer = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            utils.MyRotateTransform([0,90,180,270]),
            transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=MyDataset.ImageNet_mean,
                                 std=MyDataset.ImageNet_std)
        ])
        #TODO
        trainset = MyDataset.Level1_128_rich_V0Stage2(self.train_list, self.hparams.label_dir,
                                             transform=train_transformer, preload=self.hparams.preload_data)
        return DataLoader(trainset, batch_size=self.hparams.batch_size, drop_last=False, shuffle=True, num_workers=8)

    def val_dataloader(self):
        # transforms
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MyDataset.ImageNet_mean,
                                 std=MyDataset.ImageNet_std)
        ])
        # data
        #TODO:
        valset = MyDataset.Level1_128_rich_V0Stage2(self.val_list,self.hparams.label_dir,
                                           transform=train_transformer, preload = self.hparams.preload_data)
        return DataLoader(valset, batch_size=self.hparams.batch_size, drop_last=False, shuffle=False, num_workers=8)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        # scheduler = StepLR(optimizer, step_size=300)
        scheduler = CosineAnnealingLR(optimizer, self.trainer.max_epochs, self.hparams.cosine_scheduler_end_lr)
        return {"optimizer":optimizer,"lr_scheduler":scheduler}

    def training_step(self, batch, batch_idx):
        data,label = batch
        output = self(data)
        if self.hparams.loss_type == "L1":
            criterion = nn.L1Loss()
        elif self.hparams.loss_type == "L2":
            criterion = nn.MSELoss()
        elif self.hparams.loss_type == "SmoothL1":
            criterion = nn.SmoothL1Loss()
        else:
            raise ValueError("Wrong type of loss given")
        loss = criterion(output, label)
        # add logging
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if self.hparams.loss_type == "L1":
            criterion = nn.L1Loss()
        elif self.hparams.loss_type == "L2":
            criterion = nn.MSELoss()
        elif self.hparams.loss_type == "SmoothL1":
            criterion = nn.SmoothL1Loss()
        else:
            raise ValueError("Wrong type of loss given")
        loss = criterion(logits, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # pred_total = torch.cat([x['pred'] for x in outputs]).view(-1).cpu()
        # y_total = torch.cat([x['label'] for x in outputs]).view(-1).cpu()
        # logits_total = torch.cat([x['logits'] for x in outputs],dim=0).cpu()

        # self.log_distribution(logits_total,y_total)
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def on_epoch_start(self):
        print("\nCall hook function: on_epoch_start")
        if self.current_epoch == self.hparams.freeze_epochs:
            self.unfreeze_for_transfer()

    def on_epoch_end(self):
        print("\nCall hook function: on_epoch_end")
        self.log_histogram()

    def on_train_start(self):
        print("\nCall hook function: on_train_start")
        self.freeze_for_transfer()

    """=============================self-defined function============================="""

    def filter_out_benign(self, img_list):
        valid_list =[]
        for img_path in tqdm(img_list):
            # read label
            image_id = img_path.split("/")[-1].split("-")[0]
            patch_id = int(img_path.split("/")[-1].split(".")[0].split("-")[1])
            with open(os.path.join(self.hparams.label_dir, image_id + ".json"), "r") as f:
                dict = json.loads(f.read())

            if dict["data_provider"] == "karolinska":
                if dict["patches_stat"]["cancerous_tissue_perc"][patch_id - 1] > 0.1:
                    valid_list.append(img_path)
            else:

                if dict["patches_stat"]["Gleason_3_perc"][patch_id - 1] > 0.01 or \
                        dict["patches_stat"]["Gleason_4_perc"][patch_id - 1] > 0.01 or \
                        dict["patches_stat"]["Gleason_5_perc"][patch_id - 1] > 0.01:
                    valid_list.append(img_path)
        return valid_list


    def log_distribution(self, logits, label):
        fig, ax = plt.subplots()
        prob = F.softmax(logits, dim=1).numpy()
        bins = np.linspace(-0.1, 1.1, 100)
        cancerous_p = prob[:, 1]
        ax.hist([cancerous_p[label == 0], cancerous_p[label == 1]], bins=50,
                label=["Benign", "Cancerous"], range=(0, 1))
        ax.legend(loc='upper right')
        self.logger.experiment.add_figure("2 class prob distribution", fig, self.current_epoch)

    def log_histogram(self):
        print("\nlog hist of weights")

        enc_dict = self.enc.state_dict()
        for name, val in enc_dict.items():
            self.logger.experiment.add_histogram("encoder/" + name, val, self.current_epoch)

        head_dict = self.head.state_dict()
        for name, val in head_dict.items():
            self.logger.experiment.add_histogram("head/" + name, val, self.current_epoch)

    def freeze_for_transfer(self):
        print("Freeze encoder for {} epochs".format(self.hparams.freeze_epochs))
        for param in self.enc.parameters():
            param.requires_grad = False

    def unfreeze_for_transfer(self):
        print("UnFreeze encoder at {}-th epoch".format(self.hparams.freeze_epochs))
        for param in self.enc.parameters():
            param.requires_grad = True


def main(args):
    now = datetime.now()
    time_stamp = now.strftime("%m-%d-%y_%H-%M-%S")

    IMAGE_LIST = [os.path.join(args.img_dir,fname) for fname in  os.listdir(args.img_dir)]
    random.shuffle(IMAGE_LIST)
    # kf = KFold(n_splits=3,shuffle=True)
    # for train_index, test_index in kf.split(IMAGE_LIST):
    if args.overfit_test:
        # first overfitting on small dataset
        IMAGE_LIST = IMAGE_LIST[:10000]

    #split train val
    num_train = int(0.7 * len(IMAGE_LIST))
    train_list = IMAGE_LIST[:num_train]
    val_list = IMAGE_LIST[num_train:]
    print("train_list:", train_list[:5])
    if args.checkpoint_path:
        checkpoint_fn = os.path.join(args.checkpoint_path,time_stamp + "_{epoch:02d}-{val_loss:.2f}")
    else:
        checkpoint_fn = None
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_fn,
        monitor='val_loss',
        save_top_k=1,
        mode='min'
    )
    lr_logger = LearningRateLogger()

    model = V0_Stage2_System(hparams=args,train_list=train_list,val_list = val_list)

    trainer = Trainer(checkpoint_callback=checkpoint_callback,
                      callbacks=[lr_logger],
                      gpus=args.gpus,
                      max_epochs=args.max_epoch,
                      progress_bar_refresh_rate=50,
                      default_save_path = args.training_log_path)
    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    # parser = Trainer.add_argparse_args(parser)
    """training strategy"""
    parser.add_argument('--gpus', type=int, default=1, help='')
    parser.add_argument('--overfit_test', type=int, default=1, help='')
    parser.add_argument('--max_epoch', type=int, default=30, help='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default= 1e-3)
    parser.add_argument('--freeze_epochs', type=int, default=10)
    parser.add_argument('--img_dir', type=str, default="/mnt/ssd2/AllDatasets/ProstateDataset/Level1_128_rich/train", help='')
    parser.add_argument('--label_dir', type=str, default="/mnt/ssd2/AllDatasets/ProstateDataset/Level1_128_rich/Label", help='')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Default is None, the checkpoint will be saved under the training log folder')
    parser.add_argument('--training_log_path', type=str, default="./", help='')
    """model selection"""
    parser.add_argument('--NOTE', type=str, default="use ImageNet mean and var when load image", help='')
    parser.add_argument('--arch', type=str, default='efficientnet-b2', help='')
    parser.add_argument('--num_class', type=int, default=3, help='')
    parser.add_argument('--load_pretrained', type=int, default=1, help='')
    parser.add_argument('--pretrained_weights', type=str, default="", help='')
    # /mnt/ssd2/Projects/ProstateChallenge/output/PretrainedModelLB79/RNXT50_0.pth

    parser.add_argument('--loss_w1', type=float, default=3., help='CrossEntropy loss weight for Cancerous type')
    parser.add_argument('--preload_data', type=int, default=0, help='default is 0. Preload images into RAM')
    parser.add_argument('--cosine_scheduler_end_lr', type=float, default= 5e-6, help='CrossEntropy loss weight for Cancerous type')
    parser.add_argument('--loss_type', type=str, default="", help='')


    parser = V0_Stage2_System.add_model_specific_args(parser)

    args = parser.parse_args()

    main(args)
