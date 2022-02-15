import os, pickle

from typing import Union
from argparse import Namespace

import numpy as np

import torch
from torch import nn
from torch.optim import (Adam, Adadelta, Adagrad, AdamW, Adamax,
                         ASGD, RMSprop, Rprop, SGD)

from sklearn.metrics import precision_recall_fscore_support as prf
from sklearn import metrics

def xavier_initialize(model:nn.Module)->None:

    modules = list()
    for m in model.modules():
        if isinstance(m, nn.Linear):
            modules.append(m)
        if isinstance(m, nn.Conv2d):
            modules.append(m)
        if isinstance(m, nn.ConvTranspose2d):
            modules.append(m)
    parameters = [
        p for
        m in modules for
        p in m.parameters() if
        p.dim() >= 2
    ]

    for p in parameters:
        nn.init.xavier_normal(p)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        
def save_model(model, optimizer, scheduler, opts, seed, verbose:bool=False, c:torch.Tensor=None, pretrain:bool=False):

    # save opts
    opts_filename = os.path.join(opts.model_dir, "opts_seed_{}.pth".format(seed))
    if verbose:
        print("Save %s" %opts_filename)
    with open(opts_filename, 'wb') as f:
        pickle.dump(opts, f)

    # serialize model, scheduler and optimizer to dict
    if not (scheduler is None):
        state_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
    else:
        state_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': None
        }
    
    state_dict['c'] = c

    if not pretrain:
        model_filename = os.path.join(opts.model_dir, "model_epoch_{}_seed_{}.pth".format(model.epoch, seed))
    else:
        model_filename = os.path.join(opts.model_dir, "ae_model_epoch_{}_seed_{}.pth".format(model.epoch, seed))
    print("Save {}".format(model_filename))
    torch.save(state_dict, model_filename)

def load_model(model, optimizer, scheduler, opts, epoch, seed, device, verbose:bool=False, c:bool=False, pretrain:bool=False):

    # load model
    if not pretrain:
        model_filename = os.path.join(opts.model_dir, "model_epoch_{}_seed_{}.pth".format(epoch, seed))
    else:
        model_filename = os.path.join(opts.model_dir, "ae_model_epoch_{}_seed_{}.pth".format(epoch, seed))
    if verbose:
        print("Load %s" %model_filename)
    state_dict = torch.load(model_filename, map_location=device)
    
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    if not (scheduler is None):
        scheduler.load_state_dict(state_dict['scheduler'])

    ### move optimizer state to GPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    model.epoch = epoch ## reset model epoch

    if c:
        c = state_dict['c'].to(device)
        return model, optimizer, scheduler, c
    else:
        return model, optimizer, scheduler

def get_optimizer(model_params, optimizer_name:str,
                 kwargs:Union[Namespace,dict],):
    '''
    Returns an optimizer from an optimizer name:
    @params:
        {torch.module.parameters} model_params: model parameters to optimize
        {str} optimizer_name: name of optimizer, 
                              must be in [Adadelta, Adagrad, Adam, AdamW, Adamax,
                                          ASGD, RMSprop, Rprop, SGD].
        {float} lr: initial learning rate.
        {argparse.Namespace} kwargs: arguments necessary for optimizer, depends on chosen optimizer.
    @returns:
        {torch.optim.optimizer}
    '''
    assert optimizer_name in ['Adadelta', 'Adagrad', 'Adam',
                              'AdamW', 'Adamax','ASGD', 
                              'RMSprop', 'Rprop', 'SGD'], 'Optimizer `{}` not implemented'.format(optimizer_name)

    if optimizer_name=='Adadelta':
        optimizer = Adadelta(model_params, lr=kwargs.lr, rho=kwargs.rho if kwargs else 0.9,
                             eps=kwargs.eps if kwargs else 1e-06)

    if optimizer_name=='Adagrad':
        optimizer = Adagrad(model_params, lr=kwargs.lr, lr_decay=kwargs.lr_decay if kwargs else 0,
                             weight_decay=kwargs.weight_decay if kwargs else 0,
                             eps=kwargs.eps if kwargs else 1e-10)

    if optimizer_name=='Adam':
        optimizer = Adam(model_params, lr=kwargs.lr, betas=(0.9, 0.999), eps=kwargs.eps if kwargs else 1e-08,
                         weight_decay=kwargs.weight_decay if kwargs else 0,
                         amsgrad=kwargs.amsgrad)

    if optimizer_name=='AdamW':
        optimizer = AdamW(model_params, lr=kwargs.lr, betas=(0.9, 0.999), eps=kwargs.eps if kwargs else 1e-08,
                         weight_decay=kwargs.weight_decay if kwargs else 0.01,
                         amsgrad=False)

    if optimizer_name=='Adamax':
        optimizer = Adamax(model_params, lr=kwargs.lr, betas=(0.9, 0.999), eps=kwargs.eps if kwargs else 1e-08,
                         weight_decay=kwargs.weight_decay if kwargs else 0,)

    if optimizer_name=='ASGD':
        optimizer = ASGD(model_params, lr=kwargs.lr, lambd=kwargs.lambd if kwargs else 1e-4,
                         alpha=kwargs.alpha if kwargs else 0.75,
                         t0=kwargs.t0 if kwargs.t0 else 1e6,
                         weight_decay=kwargs.weight_decay if kwargs else 0.01,)

    if optimizer_name=='RMSprop':
        optimizer = RMSprop(model_params, lr=kwargs.lr, alpha=kwargs.alpha if kwargs else 0.99, eps=kwargs.eps if kwargs else 1e-08,
                            weight_decay=kwargs.weight_decay if kwargs else 0,
                            momentum=kwargs.momentum if kwargs else 0,
                            centered=kwargs.centered if kwargs else False)

    if optimizer_name=='Rprop':
        optimizer = Rprop(model_params, lr=kwargs.lr, etas=tuple(kwargs.etas) if kwargs else (0.5, 1.2),
                          step_sizes=tuple(kwargs.step_sizes) if kwargs else (1e-06, 50))

    if optimizer_name=='SGD':
        optimizer = SGD(model_params, lr=kwargs.lr, momentum=kwargs.momentum if kwargs else 0, 
                        dampening=kwargs.dampening if kwargs else 0, 
                        weight_decay=kwargs.weight_decay if kwargs else 0,
                        nesterov=kwargs.nesterov if kwargs else False)

    return optimizer

def get_auroc(label: np.ndarray, score: np.ndarray) -> float:
    fprs, tprs, _ = metrics.roc_curve(label, score, pos_label=1)
    return metrics.auc(fprs, tprs)

def get_aupr(label: np.ndarray, score: np.ndarray) -> float:
    precisions, recalls, thresh = metrics.precision_recall_curve(label, score, pos_label=1)
    F1s = 2 * (precisions * recalls) / (precisions + recalls)

    return metrics.auc(recalls, precisions), F1s, precisions, recalls, thresh

def f_score(scores:np.array, labels:np.array, type:str='max', ratio:float=None)->tuple:
    '''
    @args:
        {numpy.array} scores: score value amounting to which extent a points is likely to be 1 or 0
        {dict} labels: dictionnary containing the true labels 
    '''
    assert type in ['ratio', 'max']

    y_true = np.array(labels, dtype=int)

    if type=='ratio':
        thresh = np.percentile(scores, ratio)
        y_pred = (scores >= thresh).astype(int)
        precision, recall, fscore, support = prf(y_true, y_pred, pos_label=1, average='binary')
        best_f1 = 2 * (precision * recall) / (precision + recall)

    ap_score, F1s, precisions, recalls, thresh = get_aupr(y_true, scores)

    if type=='max':
        index_best_F1 = np.nanargmax(F1s)
        best_f1 = F1s[index_best_F1]
        precision = precisions[index_best_F1]
        recall = recalls[index_best_F1]
        thresh = thresh[index_best_F1]

    auc_score = get_auroc(y_true, scores)

    return best_f1, precision, recall, ap_score, auc_score, thresh