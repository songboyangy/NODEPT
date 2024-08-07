import logging
import numpy as np
import torch
from tqdm import tqdm
from utils.my_utils import save_model, load_model, EarlyStopMonitor, Metric
import time
from model.ODEPT import ODEPT
import math
from utils.data_processing import Data
from typing import Tuple, Dict, Type

from utils.my_utils import compute_loss
from torch.distributions.normal import Normal

from utils.my_utils import sample_multi_point


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



def eval_model(model, eval: Data, decoder_data, device: torch.device, param: Dict, metric: Metric,
               single_metric: Metric,
               move_final: bool = False):
    model.eval()
    model.reset_state()
    metric.fresh()
    single_metric.fresh()
    epoch_metric = {}
    single_timestamp_metric = {}
    loss = {'train': [], 'val': [], 'test': []}
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
    i = 0
    with torch.no_grad():
        for x, label in tqdm(eval.loader(param['bs']), total=math.ceil(eval.length / param['bs']), desc='eval_or_test'):
            i = i + 1
            src, dst, trans_cas, trans_time, pub_time, types = x
            index_dict = select_label(label, types)
            target_idx = index_dict['train'] | index_dict['val'] | index_dict['test']
            trans_time, pub_time, label = move_to_device(device, trans_time, pub_time, label)
            pred, first_point = model.forward(src, dst, trans_cas, trans_time, pub_time, target_idx)
            # first_point = extra_info['first_point']
            for dtype in ['train', 'val', 'test']:
                idx = index_dict[dtype]
                if sum(idx) > 0:
                    m_target = trans_cas[idx]
                    m_label = torch.tensor(np.array([decoder_data[key] for key in m_target])).to(device)
                    m_label = m_label + 1
                    m_label = torch.log2(m_label)
                    m_pred = pred[idx]
                    m_first_point = first_point[idx]
                    loss_tem = compute_loss(pred=m_pred, label=m_label, first_point=m_first_point, z0_prior=z0_prior,
                                            observe_std=param['observe_std'],der_coef=param['lambda1'])
                    loss[dtype].append(loss_tem.item())
                    metric.update(target=m_target, pred=m_pred.cpu().numpy(), label=m_label.cpu().numpy(), dtype=dtype)
                    single_pred, single_label = sample_multi_point(param['predict_timestamps'],
                                                                    param['observe_time'], pred=m_pred.cpu().numpy(),
                                                                    label=m_label.cpu().numpy())
                    single_metric.update(target=m_target, pred=single_pred, label=single_label, dtype=dtype)
            model.update_state()
        for dtype in ['train', 'val', 'test']:
            epoch_metric[dtype] = metric.calculate_metric(dtype, move_history=True, move_final=move_final,
                                                          loss=np.mean(loss[dtype]))
            single_timestamp_metric[dtype] = single_metric.calculate_metric(dtype, move_history=True,
                                                                            move_final=move_final,
                                                                            loss=np.mean(loss[dtype]))
        return epoch_metric, single_timestamp_metric


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
        epoch_metric, single_timestamp_metric = eval_model(model, val, decoder_data, device, param, metric,
                                                           single_metric, move_final=False)
        logger.info(f"Epoch{epoch}: time_cost:{epoch_end - epoch_start} train_loss:{np.mean(train_loss)}")
        for dtype in ['train', 'val', 'test']:
            metric.info(dtype)

        if early_stopper.early_stop_check(epoch_metric['val']['msle']):
            break
        else:
            ...
    logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
    logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
    load_model(model, param['model_path'], num)
    logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
    final_metric, single_point_result = eval_model(model, test, decoder_data, device, param, metric, single_metric,
                                                       move_final=True)
    logger.info(f'multi_predict_point:{param["predict_timestamps"]}')
    logger.info(f'Runs:{num}\n {metric.history}')
    metric.save()
    model_save_path=param['model_path']
    lambda1=param['lambda1']
    model_save_path=f'{model_save_path}_lambda1_{lambda1}'
    save_model(model, model_save_path, num)
    logger.info(f"Predict time point {param['predict_timestamps']}")
    for dtype in ['train', 'val', 'test']:
        logger.info(
            f"{dtype} result: msle:{single_point_result[dtype]['msle']} male:{single_point_result[dtype]['male']} "
            f"mape:{single_point_result[dtype]['mape']}")


    result['mlse'].append(final_metric['test']['msle'])
    result['mape'].append(final_metric['test']['mape'])
