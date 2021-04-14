from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

SCHEDULER = 'CosineAnnealingLR'
factor=0.2 # ReduceLROnPlateau
patience=4 # ReduceLROnPlateau
eps=1e-6 # ReduceLROnPlateau
T_max=10 # CosineAnnealingLR
T_0=4 # CosineAnnealingWarmRestarts
min_lr=1e-6

def fetch_scheduler(optimizer):
    if SCHEDULER =='ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True, eps=eps)
    elif SCHEDULER =='CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=min_lr, last_epoch=-1)
    elif SCHEDULER =='CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=min_lr, last_epoch=-1)
    return scheduler