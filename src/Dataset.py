
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import os
import json


# Level1_128_rich_mean = [0.80566824, 0.59932643, 0.73495563]
# Level1_128_rich_std = [0.02481834, 0.06068507, 0.02895263]

Level1_128_rich_mean = [0.45271412, 0.45271412, 0.45271412]
Level1_128_rich_std = [0.33165374, 0.33165374, 0.33165374]

ImageNet_mean = [0.485, 0.456, 0.406]
ImageNet_std = [0.229, 0.224, 0.225]

class Level1_128_rich(Dataset):
    def __init__(self, image_list, label_dir, transform=None, preload = False):
        self.image_list = image_list
        self.label_dir = label_dir
        self.classes = ['Benign', 'Cancerous']
        self.num_cls = len(self.classes)
        self.transform = transform
        self.preload = preload

        if preload:
            print("Preloading dataset into RAM ...")
            self.image_set = []
            self.label_set = []
            for img_path in self.image_list:
                image = Image.open(img_path).convert('RGB')
                label = self.get_label(img_path)
                self.image_set.append(image)
                self.label_set.append(label)
            print("Preloading finished!")

    def __len__(self):
        return len(self.image_list)

    def get_label(self,img_path):
        # read label
        image_id = img_path.split("/")[-1].split("-")[0]
        patch_id = int(img_path.split("/")[-1].split(".")[0].split("-")[1])
        with open(os.path.join(self.label_dir, image_id + ".json"), "r") as f:
            dict = json.loads(f.read())

        label = None
        if dict["data_provider"] == "karolinska":
            if dict["patches_stat"]["cancerous_tissue_perc"][patch_id - 1] > 0.01:
                label = self.classes.index("Cancerous")
            else:
                label = self.classes.index("Benign")

        elif dict["data_provider"] == "radboud":
            if dict["patches_stat"]["Gleason_3_perc"][patch_id - 1] > 0.01:
                label = self.classes.index("Cancerous")
            elif dict["patches_stat"]["Gleason_4_perc"][patch_id - 1] > 0.01:
                label = self.classes.index("Cancerous")
            elif dict["patches_stat"]["Gleason_5_perc"][patch_id - 1] > 0.01:
                label = self.classes.index("Cancerous")
            else:
                label = self.classes.index("Benign")
        else:
            raise ValueError("Wrong data_provider {}".format(dict["data_provider"]))

        return label

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.preload:
            image = self.image_set[idx]
            label = self.label_set[idx]
        else:
            img_path = self.image_list[idx]
            image = Image.open(img_path).convert('RGB')
            label = self.get_label(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label


class Level1_128_rich_V0Stage2(Dataset):
    def __init__(self, image_list, label_dir, transform=None, preload = False):
        self.image_list = image_list
        self.label_dir = label_dir
        self.classes = ['G3', 'G4', 'G5']
        self.num_cls = len(self.classes)
        self.transform = transform
        self.preload = preload
        self.look_up_Kar = {"3+3":[1.0,0,0],
                            "3+4":[0.7,0.3,0.0],
                            "4+3":[0.3,0.7,0.0],
                            "4+4":[0.0,1.0,0.0],
                            "4+5":[0.0,0.7,0.3],
                            "5+4":[0.0,0.3,0.7],
                            "5+5":[0.0,0.0,1.0],
                            "3+5":[0.7,0.0,0.3],
                            "5+3":[0.3,0.0,0.7],
                            "negative":[0.0,0.0,0.0],
                            "0+0":[0.0,0.0,0.0]}
    def __len__(self):
        return len(self.image_list)

    def get_label(self,img_path):
        # read label
        image_id = img_path.split("/")[-1].split("-")[0]
        patch_id = int(img_path.split("/")[-1].split(".")[0].split("-")[1])
        with open(os.path.join(self.label_dir, image_id + ".json"), "r") as f:
            dict = json.loads(f.read())

        if dict["data_provider"] == "karolinska":
            gleason_level =dict["gleason_score"]
            label = self.look_up_Kar[gleason_level]
            label = torch.tensor(label)
        else:
            label = torch.tensor([dict["patches_stat"]["Gleason_3_perc"][patch_id-1],
                                  dict["patches_stat"]["Gleason_4_perc"][patch_id-1],
                                  dict["patches_stat"]["Gleason_5_perc"][patch_id-1]])

        # normalize
        label = label / (label.sum()+ 1e-8 )

        return label


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_list[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.get_label(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label






















