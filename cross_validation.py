import random
import os
import pandas as pd
import numpy as np
import gc
import torch
import torch
from sklearn.neighbors import NearestNeighbors
from utils.constants import Constants
from models.model import ShopeeNet
from utils.datasets import *
from common import init_logger


device = torch.device('cuda')

################################################# MODEL ####################################################################

model_name = 'efficientnet_b3' #efficientnet_b0-b7

################################################ Metric Loss and its params #######################################################
loss_module = 'arcface' #'cosface' #'adacos'
s = 30.0
m = 0.5
ls_eps = 0.0
easy_margin = False

############################################## Model Params ###############################################################
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

################################################ MODEL PATH ###############################################################
IMG_MODEL_PATH = './model_efficientnet_b3_IMG_SIZE_512_arcface.bin'

# # Utils
def init_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set True to be faster

# 定义评价函数：准确率、召回率，F1
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

def get_neighbors(df, embeddings, KNN = 50, image = True):
    '''
    https://www.kaggle.com/ragnar123/unsupervised-baseline-arcface?scriptVersionId=57121538
    '''

    model = NearestNeighbors(n_neighbors = KNN, metric='cosine')
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)
    
    # Iterate through different thresholds to maximize cv, run this in interactive mode, then replace else clause with a solid threshold
    if image:
        thresholds = list(np.arange(0, 1, 0.01))
    else:
        thresholds = list(np.arange(0.1, 1, 0.1))
    scores = []
    for threshold in thresholds:
        predictions = []
        for k in range(embeddings.shape[0]):
            idx = np.where(distances[k,] < threshold)[0]
            ids = indices[k,idx]
            posting_ids = df['posting_id'].iloc[ids].values
            predictions.append(posting_ids)

        df['pred_matches'] = predictions
        df['cv_score'] = df.apply(getMetric('pred_matches'),axis=1)
        score = df['cv_score'].apply(lambda x:x[2]).mean()
        print(f'Our f1 score for threshold {threshold} is {score}')
        scores.append(score)
    thresholds_scores = pd.DataFrame({'thresholds': thresholds, 'scores': scores})
    max_score = thresholds_scores[thresholds_scores['scores'] == thresholds_scores['scores'].max()]
    best_threshold = max_score['thresholds'].values[0]
    best_score = max_score['scores'].values[0]
    print(f'Our best score is {best_score} and has a threshold {best_threshold}')
    
    # Use threshold
    predictions = []
    for k in range(embeddings.shape[0]):
        # Because we are predicting the test set that have 70K images and different label groups, confidence should be smaller
        if image:
            idx = np.where(distances[k,] < 0.47)[0]
        else:
            idx = np.where(distances[k,] < 0.60)[0]
        ids = indices[k,idx]
        posting_ids = df['posting_id'].iloc[ids].values
        predictions.append(posting_ids)
    
    return df, predictions


def get_image_embeddings(df):
    embeds = []
    
    model = ShopeeNet(**model_params)
    model.load_state_dict(torch.load(IMG_MODEL_PATH),strict=False)
    model = model.to(device)
    model.eval()
    _, valid_transform = data_transforms()

    image_dataset = ShopeeDataset(df,'test',transform=valid_transform)
    image_loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=Constants.INFERENCE_BATCH_SIZE,
        pin_memory=True,
        drop_last=False,
        num_workers=Constants.NUM_WORKERS
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


if __name__ == "__main__":
    init_logger(f'./inferenece_model_{model_name}_IMG_SIZE_{Constants.DIM[0]}_{loss_module}.log')
    init_seeds(Constants.SEED)
    # # Calculating Predictions
    df = read_inference_dataset()
    image_embeddings = get_image_embeddings(df)

    # Get neighbors for image_embeddings
    df,image_predictions = get_neighbors(df, image_embeddings, KNN = 50, image = True)


    df['oof_cnn'] = image_predictions

    df['cv_score'] = df.apply(getMetric('oof_cnn'),axis=1)
    print('P score for baseline =',df['cv_score'].apply(lambda x:x[0]).mean())
    print('R score for baseline =',df['cv_score'].apply(lambda x:x[1]).mean())
    print('F1 score for baseline =',df['cv_score'].apply(lambda x:x[2]).mean())

