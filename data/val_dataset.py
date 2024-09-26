import torch.utils.data as data
import numpy as np
import cv2
import os
import torch
import torchvision.transforms as transforms
from utils.utils import make_weights_for_balanced_classes, get_val_data, perform_val, buffer_val, setup_seed

def de_preprocess(tensor):

    return tensor * 0.5 + 0.5

hflip = transforms.Compose([
            de_preprocess,
            transforms.ToPILImage(),
            transforms.functional.hflip,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

ccrop = transforms.Compose([
            de_preprocess,
            transforms.ToPILImage(),
            transforms.Resize([128, 128]),  # smaller side resized
            transforms.CenterCrop([112, 112]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

transform = transforms.Compose([
            # transforms.Resize([128, 128]),
            # transforms.RandomCrop([112, 112]),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])


class ValDataset(data.Dataset):
    def __init__(self, root, data_name):
        self.data = torch.tensor(root[:, [2, 1, 0], :, :])
        self.ccrop_transform = ccrop
        self.hflip_transform = hflip
        print("data name:", data_name, "---", "dataset size:", len(self.data))

    def __getitem__(self, index):
        ccroped = self.ccrop_transform(self.data[index])
        hfliped = self.hflip_transform(ccroped)
        return ccroped, hfliped

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    lfw, cfp_ff, cfp_fp, agedb, calfw, cplfw, lfw_issame, cfp_ff_issame, cfp_fp_issame, agedb_issame, calfw_issame, cplfw_issame = get_val_data(
        '/home/ubuntu/public2/cjp/FR/data')
    # root = '/home/ubuntu/public2/cjp/FR/data/lfw_flip.pt'
    val_set = ValDataset(lfw, 'lfw_tta')
    # print(len(val_set[0]))
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=256, shuffle=False, num_workers=8, drop_last=False)
    for (i, j) in val_loader:
        print(len(i))
        print(len(j))
        print(len(val_loader.dataset))
        break
