import json
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import pickle as pk
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import time
import ast


def save_model(model: nn.Module, save_path, run):
    torch.save(model.state_dict(), f'{save_path}.pth')


def load_model(model: nn.Module, load_path, run):
    model_dict = torch.load(f"{load_path}.pth")
    model.load_state_dict(model_dict)


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-10, save_path=None, logger=None,
                 model: nn.Module = None,
                 run=0):
        self.max_round = max_round
        self.num_round = 0
        self.run = run

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance
        self.save_path = save_path
        self.logger = logger
        self.model = model

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
            save_model(self.model, self.save_path, self.run)  # 在这个地方保存了模型
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
            save_model(self.model, self.save_path, self.run)
        else:
            self.num_round += 1
        self.epoch_count += 1
        if self.num_round <= self.max_round:
            return False
        return True


def set_config(args):
    param = vars(args)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    # param['prefix'] = f'{args.prefix}_{args.dataset}_CTCP'
    param['prefix'] = f'{args.prefix}_{args.dataset}_ODEPT_{timestamp}'
    param['model_path'] = f"saved_models/{param['prefix']}"
    param['result_path'] = f"results/{param['prefix']}"
    param['log_path'] = f"log/{param['prefix']}.log"
    data_config = json.load(open('config/config.json', 'r'))[param['dataset']]
    param.update(data_config)
    return param


def compute_derivative(values):
    derivatives = (values[:, 1:] - values[:, :-1])
    return derivatives


def compute_derivative_loss(pred, label):
    dy_pred_dt = compute_derivative(pred)
    dy_true_dt = compute_derivative(label)
    derivative_loss = torch.mean(torch.abs(dy_pred_dt - dy_true_dt))
    return derivative_loss


def compute_loss(pred, label, first_point, z0_prior, observe_std=0.1, kl_coef=1, der_coef=50):
    rec_likelihood = compute_log_likelihood(pred, label, observe_std)
    derivative_loss = compute_derivative_loss(pred, label)
    fp_mu, fp_std = first_point[:, :, 0], first_point[:, :, 1]
    fp_std = fp_std.abs()
    fp_distr = Normal(fp_mu, fp_std)
    kldiv_z0 = kl_divergence(fp_distr, z0_prior)
    kldiv_z0 = torch.mean(kldiv_z0)
    loss = - torch.logsumexp(rec_likelihood - kl_coef * kldiv_z0 - der_coef * derivative_loss, 0)
    loss = torch.mean(loss)
    return loss


def compute_log_likelihood(pred, label, observe_std=1.0):
    log_p = ((label - pred) ** 2) / (2 * observe_std * observe_std)
    neg_log_p = -1 * log_p
    neg_log_p = torch.mean(neg_log_p)
    return neg_log_p


def msle(pred, label, flag=1):
    if flag == 1:
        return np.around(np.mean(mean_squared_error(label, pred, multioutput='raw_values')), 4)
    else:
        return np.around(mean_squared_error(label, pred, multioutput='raw_values'), 4)


def pcc(pred, label):
    pred_mean, label_mean = np.mean(pred, axis=0), np.mean(label, axis=0)
    pre_std, label_std = np.std(pred, axis=0), np.std(label, axis=0)
    return np.around(np.mean((pred - pred_mean) * (label - label_mean) / (pre_std * label_std), axis=0), 4)


# def male(pred, label):
#     return np.around(mean_absolute_error(label, pred, multioutput='raw_values'), 4)[0]
def male(pred, label, flag=1):
    mae_per_sample = mean_absolute_error(label, pred, multioutput='raw_values')
    if flag == 1:
        return np.around(np.mean(mae_per_sample), 4)
    else:
        return np.around(mae_per_sample, 4)


def mape(pred, label, flag=1):
    label = 2 ** label
    pred = 2 ** pred
    result = np.mean(np.abs(np.log2(pred + 1) - np.log2(label + 1)) / np.log2(label + 2), axis=0)
    if flag == 1:
        result = np.mean(result)
        return np.around(result, 4)
    else:
        return np.around(result, 4)


def sample_multi_point(predict_points, observe_time, pred, label):
    # for predict_point in predict_points
    #     index = predict_point - observe_time-1
    predict_points = np.array(predict_points)
    predict_points = predict_points - observe_time - 1
    multi_pred, multi_label = pred[:, predict_points], label[:, predict_points]
    return multi_pred, multi_label


class Metric:
    def __init__(self, path, logger, fig_path, flag=1):
        self.template = {'target': [], 'pred': [], 'label': [], 'msle': 0, 'male': 0, 'pcc': 0, 'mape': 0, 'loss': 0}
        self.mul_template = {'target': [], 'pred': [], 'label': [], 'msle': [], 'male': [], 'pcc': [], 'mape': [],
                             'loss': []}  # 多个时间点的时间戳
        self.final = {'train': deepcopy(self.template), 'val': deepcopy(self.template), 'test': deepcopy(self.template)}
        self.history = {'train': {'msle': [], 'male': [], 'mape': [], 'pcc': [], 'loss': []},
                        'val': {'msle': [], 'male': [], 'mape': [], 'pcc': [], 'loss': []},
                        'test': {'msle': [], 'male': [], 'mape': [], 'pcc': [], 'loss': []},
                        }
        self.temp = None
        self.path = path
        self.fig_path = fig_path
        self.logger = logger
        self.flag = flag

    def fresh(self):

        self.temp = {'train': deepcopy(self.template), 'val': deepcopy(self.template),
                     'test': deepcopy(self.template)}
        # else:
        #     self.temp = {'train': deepcopy(self.mul_template), 'val': deepcopy(self.mul_template),
        #                  'test': deepcopy(self.mul_template)}

    def update(self, target, pred, label, dtype):
        self.temp[dtype]['target'].append(target)
        self.temp[dtype]['pred'].append(pred)
        self.temp[dtype]['label'].append(label)

    def calculate_metric(self, dtype, move_history=True, move_final=False, loss=0):
        targets, preds, labels = self.temp[dtype]['target'], self.temp[dtype]['pred'], self.temp[dtype]['label']

        targets, preds, labels = np.concatenate(targets, axis=0), \
            np.concatenate(preds, axis=0), \
            np.concatenate(labels, axis=0)
        self.temp[dtype]['target'] = targets
        self.temp[dtype]['pred'] = preds
        self.temp[dtype]['label'] = labels
        self.temp[dtype]['msle'] = msle(preds, labels, self.flag)
        self.temp[dtype]['male'] = male(preds, labels, self.flag)
        self.temp[dtype]['mape'] = mape(preds, labels, self.flag)
        # self.temp[dtype]['pcc'] = pcc(preds, labels)
        self.temp[dtype]['loss'] = loss

        if move_history:
            for metric in ['msle', 'male', 'mape', 'loss']:
                self.history[dtype][metric].append(self.temp[dtype][metric])
        if move_final:
            self.move_final(dtype)
        return deepcopy(self.temp[dtype])

    def move_final(self, dtype):
        self.final[dtype] = self.temp[dtype]

    def save(self):
        pk.dump(self.final, open(self.path, 'wb'))

    def info(self, dtype):
        s = []
        for metric in ['loss', 'msle', 'male', 'mape', 'pcc']:
            s.append(f'{metric}:{self.temp[dtype][metric]:.4f}')
        self.logger.info(f'{dtype}: ' + '\t'.join(s))


if __name__ == '__main__':
    pred = np.array([2.5, 3.5, 4.5])
    label = np.array([2, 3, 4])
    print(male(pred, label))
