import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np


def get_data(root,mode):

    data = json.load(open(os.path.join(root,f'{mode}.json')))
    obj = json.load(open(os.path.join(root,'objects.json')))
    if mode == 'train':

        img_name = list(data.keys())
        label = list(data.values())
        for sample in range(len(label)):
            for feature in range(len(label[sample])):
                label[sample][feature] = obj[label[sample][feature]]
            tmp = np.zeros(len(obj))
            tmp[label[sample]] = 1
            label[sample] = tmp
        return np.squeeze(img_name), np.squeeze(label)

    else:

        label = data
        for sample in range(len(label)):
            for feature in range(len(label[sample])):
                label[sample][feature] = obj[label[sample][feature]]
            tmp = np.zeros(len(obj))
            tmp[label[sample]] = 1
            label[sample] = tmp

        return None, label

class ICLEVRLoader(Dataset):
    def __init__(self, trans=None, mode='train'):
        self.mode = mode
        self.img_list, self.label_list = get_data('data_information',mode)
       
        if self.mode=='train':print("%s > Found %d images..." % (mode,len(self.img_list)))
        
        self.num_classes = 24
        self.trans=trans
        
                
    def __len__(self):
        """'return the size of dataset"""
        return len(self.label_list)

    def __getitem__(self, index):
        if self.mode=='train':
            img=Image.open(os.path.join('data',self.img_list[index])).convert('RGB')

            if self.trans:
                img=self.trans(img)
            cond=torch.Tensor(self.label_list[index])
            return img,cond

        elif self.mode=='test':
            cond=torch.Tensor(self.label_list[index])
            return cond