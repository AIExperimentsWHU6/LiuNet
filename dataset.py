import pathlib
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import os

batch_size=1

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

transforms1=transforms.Compose(
    [transforms.Resize((256,256)),
     transforms.ToTensor(),
     transforms.Normalize(mean,std)])
transforms2=transforms.Compose(
    [transforms.Resize((256,256)),
     transforms.ToTensor()])

current_path=os.getcwd()
imgs_dir=os.path.join(current_path,"jsr")
masks_dir=os.path.join(current_path,"jsr_mask")

class MyData(Dataset):
    def __init__(self,img_dir,mask_dir,transform):
        super(MyData,self).__init__()
        self.imgs_dir=img_dir
        self.masks_dir=mask_dir
        self.transform=transform
        self.images=os.listdir(os.path.join(current_path,"jsr"))

    def __getitem__(self,idx):
        img_name=self.images[idx]
        img_path=os.path.join(self.imgs_dir,img_name)
        mask_path=os.path.join(self.masks_dir,img_name)

        img=Image.open(img_path).convert('RGB')
        mask=Image.open(mask_path).convert('1')
        img = transforms1(img)
        mask = transforms2(mask)


        return img,mask,img_name

    def __len__(self):
        return len(self.images)

train_dataset,test_dataset=torch.utils.data.random_split(MyData(imgs_dir,masks_dir,transforms),[0.7,0.3])
train_dataloader=DataLoader(train_dataset,batch_size=batch_size)
test_dataloader=DataLoader(test_dataset,batch_size=batch_size)



