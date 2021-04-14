import os
import logging
import torch
import torch
import torch.nn as nn
import random
import numpy as np
from utils.constants import Constants
from utils.average_meter import AverageMeter
from models.model import ShopeeNet
from utils.datasets import *
from modules.scheduler import fetch_scheduler
from modules.metric import accuracy
from common import init_logger

logger = logging.getLogger(__name__)

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
    'dropout':0.1,
    'loss_module':loss_module,
    's':s,
    'margin':m,
    'ls_eps':ls_eps,
    'theta_zero':0.785,
    'pretrained':True
}

def init_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set True to be faster

class Trainer:
    def __init__(self) -> None:
        self.device = torch.device("cuda")
        logger.info("Preparing data..")
        train, valid = read_dataset()

        train_transform, valid_transform = data_transforms()

        train_dataset = ShopeeDataset(
            train,
            'train',
            transform=train_transform,
        )
            
        valid_dataset = ShopeeDataset(
            valid,
            'train',
            transform= valid_transform,
        )
            
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size= Constants.TRAIN_BATCH_SIZE,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers= Constants.NUM_WORKERS
        )
        
        self.valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size= Constants.VALID_BATCH_SIZE,
            num_workers= Constants.NUM_WORKERS,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        # Defining Model for specific fold
        logger.info("Building model..")
        self.model = ShopeeNet(**model_params)
        self.model.to(self.device)

        #DEfining criterion
        self.criterion = nn.CrossEntropyLoss()
        
        # Defining Optimizer with weight decay to params other than bias and layer norms
        # param_optimizer = list(self.model.named_parameters())
        # no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        # optimizer_parameters = [
        #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        #         ]  
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Constants.LR)

        self.scheduler = fetch_scheduler(self.optimizer)

    def run(self):
        # THE ENGINE LOOP
        best_loss = 10000
        logger.info("Start training..")
        for epoch in range(Constants.EPOCHS):
            self.train_fn(epoch)
            valid_loss = self.eval_fn(epoch)
            
            if valid_loss.avg < best_loss:
                best_loss = valid_loss.avg
                torch.save(self.model.state_dict(),f'./model_{model_name}_IMG_SIZE_{Constants.DIM[0]}_{loss_module}.bin')
                logger.info('best model found for epoch {}'.format(epoch))
    
    def train_fn(self,epoch):
        self.model.train()
        loss_score = AverageMeter()
        acc = AverageMeter()
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            batch_size = inputs.size(0)
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs,targets)
            loss = self.criterion(outputs,targets)
            loss.backward()
            self.optimizer.step()

            prec1, = accuracy(outputs, targets, topk=(1,))
            loss_score.update(loss.item(), batch_size)
            acc.update(prec1.item(), batch_size)

        logger.info('epoch %03d trian loss: %e, accurcy: %f',epoch, loss_score.avg, acc.avg)

        if self.scheduler is not None:
            self.scheduler.step()
            
        return loss_score

    def eval_fn(self,epoch):
        self.model.eval()
        loss_score = AverageMeter()
        acc = AverageMeter()
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.valid_loader):
                batch_size = inputs.size(0)
                inputs, targets= inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs,targets)
                loss = self.criterion(outputs,targets)

                prec1, = accuracy(outputs, targets, topk=(1,))
                loss_score.update(loss.item(), batch_size)
                acc.update(prec1.item(), batch_size)

        logger.info('epoch %03d val loss: %e, accuracy: %f',epoch,loss_score.avg,acc.avg)
        return loss_score

if __name__ == "__main__":
    init_logger(f'./train_model_{model_name}_IMG_SIZE_{Constants.DIM[0]}_{loss_module}.log')
    init_seeds(Constants.SEED)
    trainer = Trainer()
    trainer.run()