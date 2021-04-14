import os
import numpy as np
import torch
import cv2
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from .constants import Constants

def data_transforms():
    train_transform = albumentations.Compose(
        [   
            albumentations.Resize(Constants.DIM[0],Constants.DIM[1],always_apply=True),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=120, p=0.8),
            albumentations.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
            albumentations.HueSaturationValue(p=0.5, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
            albumentations.ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
            albumentations.CoarseDropout(p=0.5),
            albumentations.Normalize()
        ]
    )

    valid_transform = albumentations.Compose(
        [
            albumentations.Resize(Constants.DIM[0],Constants.DIM[1],always_apply=True),
            albumentations.Normalize()
        
        ]
    )

    return train_transform, valid_transform

def read_dataset():
    # Defining DataSet
    data = pd.read_csv('./input/shopee-folds/folds.csv')
    data['image'] = data['image'].apply(lambda x: os.path.join('./input/shopee-product-matching/', 'train_images', x))
    encoder = LabelEncoder()
    data['label_group'] = encoder.fit_transform(data['label_group'])
    train = data[data['fold'] != 0].reset_index(drop=True)
    valid = data[data['fold'] == 0].reset_index(drop=True)

    return train, valid

def read_inference_dataset():
    df = pd.read_csv('./input/shopee-product-matching/train.csv')
    df['image'] = './input/shopee-product-matching/train_images/' + df['image']
    tmp = df.groupby(['label_group'])['posting_id'].unique().to_dict()
    df['target'] = df['label_group'].map(tmp)

    return df


class ShopeeDataset(Dataset):
    def __init__(self, df, mode, transform=None):
        
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.loc[index]
        img = cv2.imread(row.image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']
                
        img = img.astype(np.float32)
        img = img.transpose(2,0,1)
        
        if self.mode == 'test':
            return torch.tensor(img).float()
        else:
            return torch.tensor(img).float(), torch.tensor(row.label_group)