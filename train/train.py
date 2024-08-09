import logging
import numpy as np
import torch
from tqdm import tqdm
from utils.my_utils import save_model, load_model, EarlyStopMonitor, Metric
import time
from model.NODEPT import ODEPT
import math
from utils.data_processing import Data
from typing import Tuple, Dict, Type
from utils.my_utils import compute_loss
from torch.distributions.normal import Normal


def select_label(labels, types):
    train_idx = (labels != -1) & (types == 1)
    val_idx = (labels != -1) & (types == 2)
    test_idx = (labels != -1) & (types == 3)
    return {'train': train_idx, 'val': val_idx, 'test': test_idx}


def move_to_device(device, *args):
    results = []
    for arg in args:
        if type(arg) is torch.Tensor:
            results.append(arg.to(dtype=torch.float, device=device))
        else:
            results.append(torch.tensor(arg, device=device, dtype=torch.float))
    return results


def train_model(num: int, dataset: Data, decoder_data, model, logger: logging.Logger,
                early_stopper: EarlyStopMonitor,
                device: torch.device, param: Dict, metric: Metric, result: Dict, single_metric: Metric):
    train, val, test = dataset, dataset, dataset
    model = model.to(device)
    logger.info('Start training citation')
    optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'])
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))

    for epoch in range(param['epoch']):
        model.reset_state()
        model.train()
        model.external_memory.reset_memory()
        logger.info(f'Epoch {epoch}:')
        epoch_start = time.time()
        train_loss = []
        train_kldiv_z0 = []
        i = 0

        for x, label in tqdm(train.loader(param['bs']), total=math.ceil(train.length / param['bs']),
                             desc='training'):
            src, dst, trans_cas, trans_time, pub_time, types = x
            index_dict = select_label(label, types)
            i = i + 1
            target_idx = index_dict['train'] | index_dict['val'] | index_dict['test']
            trans_time, pub_time, label = move_to_device(device, trans_time, pub_time, label)
            pred, first_point = model.forward(src, dst, trans_cas, trans_time, pub_time, target_idx)
            train_idx = index_dict['train']
            if sum(train_idx) > 0:
                target, target_time = trans_cas[train_idx], trans_time[train_idx]
                target_label = torch.tensor(np.array([decoder_data[key] for key in target]))
                target_label = target_label + 1
                target_label = torch.log2(target_label)
                target_label = target_label.to(device)
                target_pred = pred[train_idx]
                target_first_point = first_point[train_idx]
                optimizer.zero_grad()
                loss = compute_loss(target_pred, target_label, first_point=target_first_point, z0_prior=z0_prior,
                                    observe_std=param['observe_std'],der_coef=param['lambda1'])
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

            model.update_state()
            model.detach_state()

        epoch_end = time.time()
        logger.info(f"Epoch{epoch}: time_cost:{epoch_end - epoch_start} train_loss:{np.mean(train_loss)}")
        for dtype in ['train', 'val', 'test']:
            metric.info(dtype)
    logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
    logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
    load_model(model, param['model_path'], num)
    logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
    logger.info(f'multi_predict_point:{param["predict_timestamps"]}')
    logger.info(f'Runs:{num}\n {metric.history}')
    metric.save()
    model_save_path=param['model_path']

    save_model(model, model_save_path, num)
