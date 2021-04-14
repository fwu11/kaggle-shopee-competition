import timm
import torch.nn as nn
import torch.nn.functional as F
import logging
from modules.loss_function import AdaCos,ArcMarginProduct,AddMarginProduct

logger = logging.getLogger(__name__)

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
        logger.info('Building Model Backbone for {} model'.format(model_name))

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
        elif loss_module == 'cosface':
            self.final = AddMarginProduct(fc_dim, n_classes, s=s, m=margin)
        elif loss_module == 'adacos':
            self.final = AdaCos(fc_dim, n_classes, m=margin, theta_zero=theta_zero)
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