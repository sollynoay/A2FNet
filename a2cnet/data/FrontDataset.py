from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageOps
from PIL import ImageFilter

torch.manual_seed(3)

class FrontDataset(Dataset):
    def __init__(self,label_dir,transform=None, is_train=True):
        self.transform = transform
        self.train = is_train
        with open(label_dir) as f:
            lines = f.readlines()
            self.length = len(lines)
            self.path2image=[]
            self.path2front=[]
            self.path2front_hd=[]
            self.path2ele=[]
            #self.path2mask=[]

            for data in lines:
                data = data.strip('\n')
                path2image,path2front,path2front_hd,path2ele = data.split(' ')
                self.path2image.append(path2image)
                self.path2front.append(path2front)
                self.path2front_hd.append(path2front_hd)
                self.path2ele.append(path2ele)

            #path2image, path2label = lines[0].split(' ')
            #print(length)
    def __len__(self):
        return self.length
    def __getitem__(self,idx):
        #path2image, path2label = self.lines[idx].split(' ')
        image = Image.open(self.path2image[idx]).convert('L')
        front = Image.open(self.path2front[idx])
        front_hd = np.loadtxt(self.path2front_hd[idx])
        ele =  Image.open(self.path2ele[idx])
        
        front_hd_resize = torch.tensor(front_hd,dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        #front_hd_resize = F.interpolate(front_hd_resize,(32,128), mode='bicubic')
        
        #mask = image.filter(ImageFilter.MinFilter(3))
        mask = image.point(lambda p: p > 25 and 255)
        #SnP_noise = SaltAndPepperNoise()
        #image = SnP_noise(image)
        
        hflip = np.random.random() < 0.5
        if hflip and self.train:
            image=ImageOps.mirror(image)
            front=ImageOps.mirror(front)
            ele=ImageOps.mirror(ele)
            mask=ImageOps.mirror(mask)
            front_hd_resize = torch.flip(front_hd_resize, [1, 3])
        
        front_hd_resize = front_hd_resize.squeeze(0)
        #front_hd_resize =1/front_hd_resize
        if self.transform is not None:
            image = self.transform(image)
            front = self.transform(front)
            ele = self.transform(ele)
            mask = self.transform(mask)

                        
           
        return image,front,front_hd_resize,ele,mask