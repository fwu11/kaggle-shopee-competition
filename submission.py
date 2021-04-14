# %% [markdown]
# # About this Notebook
# 
# Hi all this is the inference notebook for the training notebook found [here](https://www.kaggle.com/tanulsingh077/pytorch-metric-learning-pipeline-only-images?scriptVersionId=57596864) and is a Pytorch Implementation of kernel given by @ragnar from [here](https://www.kaggle.com/ragnar123/unsupervised-baseline-arcface)
# 
# What we are using in inference :
# * Effnet-B3 Trained with arc-face and Cross Entropy loss for Images
# * TFiDF for texts
# 
# I am able to achieve 0.712 lb using the training and this inference notebook without any changes on the baseline. I will be adding training and inference code for transformer model on texts as well
# 
# This notebook runs without errors for all the efficientnet architectures
# 
# I am quick saving notebook for now as I don't have GPU left , I will commit and get an lb score on this on the weekend .

# %% [code] {"jupyter":{"outputs_hidden":false}}
import sys
sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Preliminaries
import math
import random
import os
import pandas as pd
import numpy as np

# Visuals and CV2
import cv2

# albumentations for augs
import albumentations

#torch
import torch
import timm
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader


import gc
import matplotlib.pyplot as plt
import cudf
import cuml
import cupy
from cuml.feature_extraction.text import TfidfVectorizer
from cuml import PCA
from cuml.neighbors import NearestNeighbors

# %% [code] {"jupyter":{"outputs_hidden":false}}
DIM = (512,512)

NUM_WORKERS = 4
BATCH_SIZE = 12
SEED = 2020

device = torch.device('cuda')

CLASSES = 11014

################################################  ADJUSTING FOR CV OR SUBMIT ##############################################

CHECK_SUB = True
GET_CV = False

test = pd.read_csv('../input/shopee-product-matching/test.csv')
if len(test)>3: GET_CV = False
else: print('this submission notebook will compute CV score, but commit notebook will not')


################################################# MODEL ####################################################################

model_name = 'efficientnet_b3' #efficientnet_b0-b7

################################################ MODEL PATH ###############################################################

IMG_MODEL_PATH = '../input/shopeearcface/model_efficientnet_b3_IMG_SIZE_512_arcface.bin'

################################################ Metric Loss and its params #######################################################
loss_module = 'arcface' #'cosface' #'adacos'
s = 30.0
m = 0.5 
ls_eps = 0.0
easy_margin = False

model_params = {
    'n_classes':11014,
    'model_name': model_name,
    'fc_dim':512,
    'dropout':0.5,
    'loss_module':loss_module,
    's':s,
    'margin':m,
    'ls_eps':ls_eps,
    'theta_zero':0.785,
    'pretrained':False
}

# %% [markdown]
# # Loading Data

# %% [code] {"jupyter":{"outputs_hidden":false}}
def read_dataset():
    df = pd.read_csv('../input/shopee-product-matching/test.csv')
    df['image'] = '../input/shopee-product-matching/test_images/' + df['image']
    df_cu = cudf.DataFrame(df)
    return df, df_cu

# %% [code] {"jupyter":{"outputs_hidden":false}}
def init_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set True to be faster
    
init_seeds(SEED)

# %% [code] {"jupyter":{"outputs_hidden":false}}
def getMetric(col):
    def f1_score(row):
        n = len( np.intersect1d(row.target,row[col]))
        if len(row[col])==0:
            p = 0
        else:
            p = n/len(row[col])
        if len(row.target) == 0:
            r = 0
        else:
            r = n/len(row.target)
        return p, r, 2*n/(len(row.target)+len(row[col]))
    return f1_score

# %% [code] {"jupyter":{"outputs_hidden":false}}
def combine_predictions(row):
    x = np.concatenate([row['image_predictions'], row['text_predictions']])
    return ' '.join( np.unique(x) )

# %% [code] {"jupyter":{"outputs_hidden":false}}
def get_neighbors(df, embeddings, image = True):
    '''
    https://www.kaggle.com/ragnar123/unsupervised-baseline-arcface?scriptVersionId=57121538
    '''
    if len(df) > 3:
        KNN = 50
    else : 
        KNN = 3

    model = NearestNeighbors(n_neighbors = KNN, metric='cosine')
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)
    
    # Iterate through different thresholds to maximize cv, run this in interactive mode, then replace else clause with a solid threshold
    predictions = []
    for k in range(embeddings.shape[0]):
        if image:
            idx = np.where(distances[k,] < 0.47)[0]
        else:
            idx = np.where(distances[k,] < 0.60)[0]
        ids = indices[k,idx]
        posting_ids = df['posting_id'].iloc[ids].values
        predictions.append(posting_ids)
        
    del model, distances, indices
    gc.collect()
    return df, predictions

# %% [markdown]
# # Using Images

# %% [code] {"jupyter":{"outputs_hidden":false}}
def get_test_transforms():

    return albumentations.Compose(
        [
            albumentations.Resize(DIM[0],DIM[1],always_apply=True),
            albumentations.Normalize()
        ]
    )

# %% [code] {"jupyter":{"outputs_hidden":false}}
class ShopeeDataset(Dataset):
    def __init__(self, df, transform=None):
        
        self.df = df.reset_index(drop=True)
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
        
        return torch.tensor(img)


# %% [code] {"jupyter":{"outputs_hidden":false}}
class ShopeeNet(nn.Module):
    def __init__(self,
                 n_classes,
                 model_name='resnet18',
                 fc_dim=512,
                 dropout=0.5,
                 loss_module='softmax',
                 s=30.0,
                 margin=0.50,
                 ls_eps=0.0,
                 theta_zero=0.785,
                 pretrained=True):
        """
        :param n_classes:
        :param model_name: name of model from pretrainedmodels
            e.g. resnet50, resnext101_32x4d, pnasnet5large
        :param loss_module: One of ('arcface', 'cosface', 'softmax')
        """
        super().__init__()

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        self.in_features = self.backbone.classifier.in_features
        # self.in_features = self.backbone.fc.in_features

        self.pooling = nn.AdaptiveAvgPool2d(1)
        
        self.bn1 = nn.BatchNorm2d(self.in_features)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        self.fc1 = nn.Linear(self.in_features * 16 * 16 , fc_dim)
        self.bn2 = nn.BatchNorm1d(fc_dim)

        self.loss_module = loss_module
        if loss_module == 'arcface':
            self.final = ArcMarginProduct(fc_dim, n_classes,
                                          s=s, m=margin, easy_margin=False, ls_eps=ls_eps)
        else:
            self.final = nn.Linear(fc_dim, n_classes)

    def forward(self, x, labels=None):
        features = self.backbone.forward_features(x)
        features = self.bn1(features)
        features = self.dropout(features)
        features = features.view(features.size(0), -1)
        features = self.fc1(features)
        features = self.bn2(features)
        features = F.normalize(features)
        if labels is not None:
            if self.loss_module in ('arcface', 'cosface', 'adacos'):
                features = self.final(features, labels)
            else:
                features = self.final(features)
        return features

# %% [code] {"jupyter":{"outputs_hidden":false}}
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

# %% [code] {"jupyter":{"outputs_hidden":false}}
def get_image_embeddings(df):
    embeds = []
    
    model = ShopeeNet(**model_params)
    model.load_state_dict(torch.load(IMG_MODEL_PATH),strict=False)
    model = model.to(device)
    model.eval()

    image_dataset = ShopeeDataset(df,transform=get_test_transforms())
    image_loader = DataLoader(
        image_dataset,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        drop_last=False,
        num_workers=NUM_WORKERS
    )
    
    
    with torch.no_grad():
        for img in image_loader: 
            img = img.to(device)
            feat = model(img)
            image_embeddings = feat.detach().cpu().numpy()
            embeds.append(image_embeddings)
    
    
    del model
    image_embeddings = np.concatenate(embeds)
    print(f'Our image embeddings shape is {image_embeddings.shape}')
    del embeds
    gc.collect()
    return image_embeddings

# %% [markdown]
# # Using Texts with TFiDF

# %% [code] {"jupyter":{"outputs_hidden":false}}
def get_text_predictions(df, max_features = 25_000):
    
    model = TfidfVectorizer(stop_words = 'english', binary = True, max_features = max_features)
    text_embeddings = model.fit_transform(df_cu['title']).toarray()
    preds = []
    CHUNK = 1024*4

    print('Finding similar titles...')
    CTS = len(df)//CHUNK
    if len(df)%CHUNK!=0: CTS += 1
    for j in range( CTS ):

        a = j*CHUNK
        b = (j+1)*CHUNK
        b = min(b,len(df))
        print('chunk',a,'to',b)

        # COSINE SIMILARITY DISTANCE
        cts = cupy.matmul( text_embeddings, text_embeddings[a:b].T).T

        for k in range(b-a):
            IDX = cupy.where(cts[k,]>0.75)[0]
            o = df.iloc[cupy.asnumpy(IDX)].posting_id.values
            preds.append(o)
    
    del model,text_embeddings
    gc.collect()
    return preds

# %% [markdown]
# # Calculating Predictions

# %% [code] {"jupyter":{"outputs_hidden":false}}
df,df_cu = read_dataset()
df.head()

# %% [code] {"jupyter":{"outputs_hidden":false}}
image_embeddings = get_image_embeddings(df)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Get neighbors for image_embeddings
df,image_predictions = get_neighbors(df, image_embeddings, image = True)

text_predictions = get_text_predictions(df, max_features = 25000)
# %% [code] {"jupyter":{"outputs_hidden":false}}
df.head()

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Get neighbors for text_embeddings
# df, text_predictions = get_neighbors(df, text_embeddings, KNN = 50, image = False)

# %% [code] {"jupyter":{"outputs_hidden":false}}
df.head()

# %% [markdown]
# # Preparing Submission

# %% [code] {"jupyter":{"outputs_hidden":false}}

df['image_predictions'] = image_predictions
df['text_predictions'] = text_predictions
df['matches'] = df.apply(combine_predictions, axis = 1)
df[['posting_id', 'matches']].to_csv('submission.csv', index = False)