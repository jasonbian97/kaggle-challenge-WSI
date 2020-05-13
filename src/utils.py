



import torchvision.transforms.functional as TF
import random

class MyRotateTransform(object):
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

# pretrained model class
def _resnext(url, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    # state_dict = load_state_dict_from_url(url, progress=progress)
    # model.load_state_dict(state_dict)
    return model

import torch
import torch.nn as nn

class OLD_Model_enc(nn.Module):
    def __init__(self, arch='resnext50_32x4d_ssl', n=6, pre=True):
        super().__init__()
        m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        self.enc = nn.Sequential(*list(m.children())[:-2])
        self.out_features = list(m.children())[-1].in_features
        # self.head = nn.Sequential(nn.AvgPool2d(4), nn.Flatten(), nn.Linear(2 * nc, 512),
        #                           nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.5), nn.Linear(512, n))

    def forward(self, x):
        pass
