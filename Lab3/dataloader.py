import pandas as pd
from torch.utils import data
import numpy as np
import PIL.Image as Image
from torchvision import transforms


def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode,transform=None):
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        if transform == None:
            self.transform = transforms.ToTensor()
        else: self.transform = transform
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        
        path = self.root + self.img_name[index] + '.jpeg'
        img = Image.open(path , mode='r')

        if self.transform!=None:
            img = self.transform(img)

        label = self.label[index]

        return img, label
