import logging
import sys
import random
import argparse
import torch
import numpy as np
from model.CTCP import CTCP
from utils.data_processing import get_data
from train.train import train_model
from utils.my_utils import EarlyStopMonitor, set_config, Metric
from collections import defaultdict
from train.train import test_model
import ast
parser = argparse.ArgumentParser('hyper parameters of CTCP')
parser.add_argument('--dataset', type=str, help='dataset name ',
                    default='twitter', choices=['aps', 'twitter', 'weibo'])
parser.add_argument('--bs', type=int, default=50, help='batch size')
parser.add_argument('--prefix', type=str, default='test', help='prefix to name a trial')
parser.add_argument('--epoch', type=int, default=150, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--run', type=int, default=1, help='number of runs')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=64, help='dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=16, help='dimensions of the time embedding')
parser.add_argument('--patience', type=int, default=15, help='patience for the early stopping strategy')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout probability')
parser.add_argument('--predictor', type=str, default="linear", choices=["linear", "merge"], help="type of predictor")
parser.add_argument('--embedding_module', type=str, default="aggregate", choices=["identity", "aggregate"],
                    help="type of embedding module")
parser.add_argument('--single', action='store_true',
                    help='whether to use different state updaters and message functions for users and cascades')  # 是否使用不同的update函数
parser.add_argument('--use_static', action='store_true',  help='whether use static embedding for users')
parser.add_argument('--use_dynamic', action='store_true',
                    help='whether use dynamic embedding for users and cascades')
parser.add_argument('--use_structural', action='store_true',
                    help='whether to adopt structural learning in the cascade embedding module')
parser.add_argument('--use_temporal', action='store_true',
                    help='whether to adopt temporal learning in the cascade embedding module')
parser.add_argument('--lambda', type=float, default=0.5,
                    help='the weight to balance the static result and dynamic result')
parser.add_argument('--solver', type=str, default="euler", help='dopri5,rk4,euler')
parser.add_argument('--observe_std', type=float, default=0.1,
                    help='the observe_std of data when compute loss')
parser.add_argument('--memory_size', type=int, default=32,
                    help='external memory size')
parser.add_argument('--predict_timestamps', type=str, default='',
                    help='time_point_timestamp_to_predict')
parser.add_argument('--test', action='store_true', default=False,
                    help='is_test_model')
parser.add_argument('--self_evolution', action='store_true', default=False,
                    help='is_only_self_evolution')
parser.add_argument('--test_model_path', type=str, default='',
                    help='test_model_path')
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)
param = set_config(args)  # 设置超参数，这是utils中的一个函数
param['predict_timestamps']=ast.literal_eval(args.predict_timestamps)
test_model_path=f"saved_models/{param['test_model_path']}"
##print(param['test'])
#print(param['predict_timestamps'])
# 禁用 matplotlib 的字体管理器的日志记录，避免不必要的警告
logging.getLogger('matplotlib.font_manager').disabled = True
# 设置日志记录的级别为 INFO 级别
logging.basicConfig(level=logging.INFO)
# 设置日志记录的级别为 INFO 级别，并获取默认的 logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # 设置 logger 的级别为 DEBUG 级别
# 创建一个文件处理器，将日志记录到指定的文件中，以覆盖写入模式 ('w') 打开，设置级别为 DEBUG 级别
fh = logging.FileHandler(f"{param['log_path']}", mode='w')
fh.setLevel(logging.DEBUG)
# 创建一个流处理器，将日志输出到控制台，设置级别为 WARN 级别
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
# 定义日志记录的格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 设置文件处理器和流处理器的格式
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# 将文件处理器和流处理器添加到 logger 中
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(f'observe_time:{param["observe_time"]}  restruct_time:{param["restruct_time"]}')
encoder_data, decoder_data = get_data(dataset=param['dataset'], observe_time=param['observe_time'],
                                      predict_time=param['predict_time'], restruct_time=param["restruct_time"],
                                      train_time=param['train_time'], val_time=param['val_time'],
                                      test_time=param['test_time'], time_unit=param['time_unit'],
                                      log=logger, param=param)
logger.info(param)
result = defaultdict(lambda: 0)
torch.set_num_threads(5)
time_steps_to_predict = torch.tensor(np.arange(param['observe_time'], param["restruct_time"]))

for num in range(param['run']):
    logger.info(f'begin runs:{num}')
    my_seed = num
    random.seed(my_seed)  # 设置了seed
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)
    device_string = 'cuda:{}'.format(param['gpu']) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)
    model = CTCP(args=param, device=device, node_dim=param['node_dim'], embedding_module_type=param['embedding_module'],
                 state_updater_type='gru', predictor=param['predictor'], time_enc_dim=param['time_dim'],
                 single=param['single'], ntypes={'user', 'cas'}, dropout=param['dropout'],
                 n_nodes=param['node_num'], max_time=param['max_time'], use_static=param['use_static'],
                 merge_prob=param['lambda'], max_global_time=param['max_global_time'], use_dynamic=param['use_dynamic'],
                 use_temporal=param['use_temporal'], use_structural=param['use_structural'],
                 time_steps_to_predict=time_steps_to_predict)
    metric = Metric(path=f"{param['result_path']}_{num}.pkl", logger=logger, fig_path=f"fig/{param['prefix']}")
    single_metric=Metric(path=f"{param['result_path']}_{num}_single.pkl", logger=logger, fig_path=f"fig/{param['prefix']}",flag=0)
    early_stopper = EarlyStopMonitor(max_round=param['patience'], higher_better=False, tolerance=1e-3,
                                     save_path=param['model_path'],
                                     logger=logger, model=model, run=num)
    if param['test']:
        test_model(encoder_data, decoder_data, model, logger, device, param, metric, single_metric,
                   model_path=test_model_path)
    else:
        train_model(num, encoder_data, decoder_data, model.to(device), logger, early_stopper, device, param, metric,
                    result, single_metric)


logger.info(
    f"Final: msle:{result['msle']:.4f} male:{result['male']:.4f} "
    f"mape:{result['mape']:.4f} pcc:{result['pcc']:.4f}")

