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

class Stage1V0(pl.LightningModule):

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
                print("Using ", hparams.arch)
        elif hparams.load_pretrained:
            print("load pretrained weights: ",hparams.pretrained_weights)
            m = utils.OLD_Model_enc()
            state_dict = torch.load(hparams.pretrained_weights)
            m.load_state_dict(state_dict,strict=False) # set strict = False, this will load known weights to model
            self.enc = m.enc
            out_features = m.out_features
        else:
            m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', hparams.arch)
            self.enc = nn.Sequential(*list(m.children())[:-2])
            out_features = list(m.children())[-1].in_features

        self.maxpool_branch = nn.AdaptiveMaxPool2d(1)
        self.avgpool_branch = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(2 * out_features, 512),
                                  nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, hparams.num_class))

        self.train_list = train_list
        self.val_list = val_list

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
        # data
        trainset = MyDataset.Level1_128_rich(self.train_list,self.hparams.label_dir,
                                             transform=train_transformer,preload = self.hparams.preload_data)
        return DataLoader(trainset, batch_size=self.hparams.batch_size, drop_last=False, shuffle=True, num_workers=8)

    def val_dataloader(self):
        # transforms
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MyDataset.ImageNet_mean,
                                 std=MyDataset.ImageNet_std)
        ])
        # data
        valset = MyDataset.Level1_128_rich(self.val_list,self.hparams.label_dir,
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
        criterion = nn.CrossEntropyLoss(weight = torch.tensor([1.,self.hparams.loss_w1]).cuda()) # weighted loss
        loss = criterion(output, label)
        # add logging
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        criterion = nn.CrossEntropyLoss(weight = torch.tensor([1.,self.hparams.loss_w1]).cuda())
        loss = criterion(logits, y)
        # compute confucsion matrix on this batch
        pred = logits.argmax(dim=1).view_as(y)
        return {'val_loss': loss, "pred":pred, "label":y, "logits":logits}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        pred_total = torch.cat([x['pred'] for x in outputs]).view(-1).cpu()
        y_total = torch.cat([x['label'] for x in outputs]).view(-1).cpu()
        logits_total = torch.cat([x['logits'] for x in outputs],dim=0).cpu()
        F1_score = f1_score(y_total.cpu(),pred_total,average="micro")
        Confusion_matrix = confusion_matrix(y_total,pred_total)
        print("\nConfusion_matrix = \n",Confusion_matrix)
        print("F1=",F1_score)
        self.log_distribution(logits_total,y_total)
        tensorboard_logs = {'val_loss': avg_loss,"F1_score":F1_score}
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
    def log_distribution(self,logits,label):
        fig,ax = plt.subplots()
        prob = F.softmax(logits, dim=1).numpy()
        bins = np.linspace(-0.1,1.1,100)
        cancerous_p = prob[:, 1]
        ax.hist([cancerous_p[label==0], cancerous_p[label==1]],bins = 50,
                label = ["Benign","Cancerous"],range = (0,1))
        ax.legend(loc='upper right')
        self.logger.experiment.add_figure("2 class prob distribution",fig,self.current_epoch)

    def log_histogram(self):
        print("\nlog hist of weights")

        enc_dict = self.enc.state_dict()
        for name, val in enc_dict.items():
            self.logger.experiment.add_histogram("encoder/"+name,val,self.current_epoch)

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

    # learning rate warm-up
    # def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
    #     # warm up lr
    #
    #     if self.trainer.global_step < 500:
    #         lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
    #         for pg in optimizer.param_groups:
    #             pg['lr'] = lr_scale * self.hparams.learning_rate
    #
    #     # update params
    #     optimizer.step()
    #     optimizer.zero_grad()



def main(args):
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

    print("Num_Train = ", len(train_list))
    print("Num_Val = ", len(val_list))

    checkpoint_callback = ModelCheckpoint(
        filepath=None,
        monitor='val_loss',
        save_top_k=1,
        mode='min'
    )
    lr_logger = LearningRateLogger()

    model = Stage1V0(hparams=args,train_list=train_list,val_list = val_list)
    trainer = Trainer(checkpoint_callback=checkpoint_callback,
                      callbacks=[lr_logger],
                      gpus=args.gpus,
                      max_epochs=args.max_epoch,
                      progress_bar_refresh_rate=50)
    trainer.fit(model)



if __name__ == '__main__':
    parser = ArgumentParser()
    # parser = Trainer.add_argparse_args(parser)
    """training strategy"""
    parser.add_argument('--gpus', type=int, default=1, help='')
    parser.add_argument('--overfit_test', type=int, default=0, help='')
    parser.add_argument('--max_epoch', type=int, default=30, help='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default= 1e-3)
    parser.add_argument('--freeze_epochs', type=int, default=10)
    parser.add_argument('--img_dir', type=str, default="/mnt/ssd2/AllDatasets/ProstateDataset/Level1_128_rich/train", help='')
    parser.add_argument('--label_dir', type=str, default="/mnt/ssd2/AllDatasets/ProstateDataset/Level1_128_rich/Label", help='')

    """model selection"""
    parser.add_argument('--NOTE', type=str, default="use ImageNet mean and var when load image", help='')
    parser.add_argument('--arch', type=str, default='efficientnet-b2', help='')
    parser.add_argument('--num_class', type=int, default=2, help='')
    parser.add_argument('--load_pretrained', type=int, default=1, help='')
    parser.add_argument('--pretrained_weights', type=str, default="", help='')
    # /mnt/ssd2/Projects/ProstateChallenge/output/PretrainedModelLB79/RNXT50_0.pth

    parser.add_argument('--loss_w1', type=float, default=3., help='CrossEntropy loss weight for Cancerous type')
    parser.add_argument('--preload_data', type=int, default=0, help='default is 0. Preload images into RAM')
    parser.add_argument('--cosine_scheduler_end_lr', type=float, default= 5e-6, help='CrossEntropy loss weight for Cancerous type')


    parser = Stage1V0.add_model_specific_args(parser)

    args = parser.parse_args()

    main(args)





