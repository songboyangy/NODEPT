import logging
import time
import pickle as pk
import numpy as np
import pandas as pd
import sys

# 处理数据，完成对数据的预处理和划分，根据type判断是训练集、验证机、测试集
# observe_time观测时间，希望考虑的时间,到底是什么
class Data:
    def __init__(self, data, is_split=False):
        self.srcs = data['src'].values
        self.dsts = data['dst'].values
        self.times = data['abs_time'].values
        self.trans_cascades = data['cas'].values  # 级联id
        self.pub_times = data['pub_time'].values
        self.labels = data['label'].values
        self.length = len(self.srcs)
        self.is_split = is_split
        if is_split:
            self.types = data['type'].values

    def loader(self, batch):
        for i in range(0, len(self.srcs), batch):
            right = min(i + batch, self.length)  # 计算右端项
            if self.is_split:
                yield (self.srcs[i:right], self.dsts[i:right], self.trans_cascades[i:right],
                       self.times[i:right], self.pub_times[i:right], self.types[i:right]), self.labels[i:right]
            else:
                yield (self.srcs[i:right], self.dsts[i:right], self.trans_cascades[i:right],
                       self.times[i:right], self.pub_times[i:right]), self.labels[i:right]


def get_label(x: pd.DataFrame, observe_time, label):  # get信息流行度的增量
    id = np.searchsorted(x['time'], observe_time, side='left')  # 找到级联传播中在观察时间内的时间戳。这个搜索返回一个索引，这是两个时间戳吗
    casid = x['cas'].values[0]  # 获取级联的ID
    if casid in label and id >= 10:  # 返回一个索引，如果这个id大于10，那么代表在observe_time之前有10个用户参与了这个级联，观测时间之前有10个用户参与了，才算在内
        length = min(id, 100) - 1
        #x['label'].iloc[length] = label[casid] - id  # 这个是什么意思  总的流行度减去对应observe_time的位置的流行度id
        x.iloc[length, x.columns.get_loc('label')] = label[casid] - id
        # ，也就是流行度的增量。仅仅改变了最终时间的label，观测时间的label
        return [x.iloc[:length + 1, :]]  # 观测时间之前的那些数据，进行了保留，返回了观测时间之前的那些数据
    else:
        return []


# 转化数据，将user与cascade转化为id，定义一个字典，然后对data进行转换
def data_transformation(dataset, data, cas_popularity, time_unit, min_time, param):
    if dataset == 'aps':
        data['pub_time'] = (pd.to_datetime(data['pub_time']) - pd.to_datetime(min_time)).apply(lambda x: x.days)
    else:
        data['pub_time'] -= min_time  # 进行了归0化，对pub_time进行了处理
    data['abs_time'] = (data['pub_time'] + data['time']) / time_unit  # 处于time_unit代表什么，绝对时间，time是对应转发在发布多长时间之后
    data['pub_time'] /= time_unit  # 这个也除了，可能是时间戳的单位，转化为了，转化为了每一天，本来是只有秒的
    data['time'] /= time_unit  # 都转化为了天
    data.sort_values(by=['abs_time', 'id'], inplace=True, ignore_index=True)  # 已经根据时间和id进行了排序
    users = list(set(data['src']) | set(data['dst']))
    ids = list(range(len(users)))
    user2id, id2user = dict(zip(users, ids)), dict(zip(ids, users))
    cases = list(set(data['cas']))
    ids = list(range(len(cases)))
    cas2id, id2cas = dict(zip(cases, ids)), dict(zip(ids, cases))
    data['src'] = data['src'].apply(lambda x: user2id[x])
    data['dst'] = data['dst'].apply(lambda x: user2id[x])
    data['cas'] = data['cas'].apply(lambda x: cas2id[x])
    # cas_popularity['cas']=cas_popularity.apply(lambda x: cas2id[x])
    cas_popularity1 = {cas2id[cas]: array for cas, array in cas_popularity.items()}
    param['node_num'] = {'user': max(max(data['src']), max(data['dst'])) + 1, 'cas': max(data['cas']) + 1}
    param['max_global_time'] = max(data['abs_time'])
    pk.dump({'user2id': user2id, 'id2user': id2user, 'cas2id': cas2id, 'id2cas': id2cas},
            open(f'data/{dataset}_idmap.pkl', 'wb'))
    return cas_popularity1


# 划分数据集，将数据集划分为训练集、验证集和测试集，通过设置不同的标签来表征属于哪一个数据集
def get_split_data(dataset, observe_time, predict_time, restruct_time, time_unit, all_data, min_time, metadata, log,
                   param):
    def data_split(legal_cascades, train_portion=0.7, val_portion=0.15):  # 合法的级联进行划分以及id的映射
        """
        set cas type, 1 for train cas, 2 for val cas, 3 for test cas , and 0 for other cas that will be dropped
        """
        m_metadata = metadata[metadata['casid'].isin(set(legal_cascades))]
        all_idx, type_map = {}, {}
        if dataset == 'twitter':
            dt = pd.to_datetime(m_metadata['pub_time'], unit='s', utc=True).dt.tz_convert('Asia/Shanghai')
            idx = dt.apply(lambda x: not (x.month == 4 and x.day > 10)).values  # 排除4月份且日期大于10号的数据
        elif dataset == 'weibo':
            dt = pd.to_datetime(m_metadata['pub_time'], unit='s', utc=True).dt.tz_convert('Asia/Shanghai')
            idx = dt.apply(lambda x: 18 > x.hour >= 8).values
        elif dataset == 'aps':
            idx = pd.to_datetime(m_metadata['pub_time']).apply(lambda x: x.year <= 1997).values
        else:
            idx = np.array([True] * len(m_metadata))
        cas = m_metadata[idx]['casid'].values
        rng = np.random.default_rng(42)
        rng.shuffle(cas)
        train_pos, val_pos = int(train_portion * len(cas)), int((train_portion + val_portion) * len(cas))
        train_cas, val_cas, test_cas = np.split(cas, [train_pos, val_pos])
        all_idx['train'] = train_cas
        type_map.update(dict(zip(train_cas, [1] * len(train_cas))))
        all_idx['val'] = val_cas
        type_map.update(dict(zip(val_cas, [2] * len(val_cas))))
        all_idx['test'] = test_cas
        type_map.update(dict(zip(test_cas, [3] * len(test_cas))))
        reset_cas = set(metadata['casid']) - set(train_cas) - set(val_cas) - set(test_cas)
        type_map.update(dict(zip(list(reset_cas), [0] * len(reset_cas))))
        return all_idx, type_map

    # 下面的代码统计出了predict_time时各个级联的流行度，根据id进行count，小于预测时间的流行度
    all_label = all_data[all_data['time'] < predict_time * time_unit].groupby(by='cas', as_index=False)['id'].count()
    all_label = dict(zip(all_label['cas'], all_label['id']))  # 这个id是什么，可以看一看,这个确实是流行度，
    m_data = []
    all_data_condition = all_data.copy(deep=True)
    for cas, df in all_data.groupby(by='cas'):  # 对级联数据进行聚合
        m_data.extend(get_label(df, observe_time * time_unit, all_label))  # 怪不得呢，在这里乘了time_unit，利用这个方法做标记
        # ，以天为时间单位，其实在这一步以及删除掉了那些数据
    all_data = pd.concat(m_data, axis=0)  # 这样就得到了观测时间之前的所有交互数据
    num_timestamps = restruct_time - observe_time
    all_idx, type_map = data_split(all_data[all_data['label'] != -1]['cas'].values)  # 有转发行为的那些级联进行划分，有转发数据
    all_data['type'] = all_data['cas'].apply(lambda x: type_map[x])  # 将级联id映射为相应的type
    all_data = all_data[all_data['type'] != 0]
    # all_data.to_csv(f'data/{dataset}_casper.csv')
    # # sys.exit()
    cas_id = all_data['cas'].unique()
    all_data_condition = all_data_condition[all_data_condition['cas'].isin(cas_id)]
    cas_popularity_dict = {}
    for cas, df in all_data_condition.groupby(by='cas'):
        popularity_array = np.zeros(num_timestamps)
        for ts in range(num_timestamps):
            popularity_array[ts] = df[df['time'] <= (observe_time + ts) * time_unit]['id'].count()-df[df['time'] <= observe_time  * time_unit]['id'].count()
        cas_popularity_dict[cas] = popularity_array
    """all_idx is used for baselines to select the cascade id, so it don't need to be remapped"""
    cas_popularity = data_transformation(dataset, all_data, cas_popularity_dict, time_unit, min_time,
                                         param)  # 到这里才开始转化时间了
    all_data.to_csv(f'data/{dataset}_split.csv', index=False)  # data中的label是增量，我这里面是直接的流行度
    pk.dump(all_idx, open(f'data/{dataset}_idx.pkl', 'wb'))
    log.info(
        f"Total Trans num is {len(all_data)}, Train cas num is {len(all_idx['train'])}, "
        f"Val cas num is {len(all_idx['val'])}, Test cas num is {len(all_idx['test'])}")
    return Data(all_data, is_split=True), cas_popularity


# 数据加载与预处理，返回一个Data对象，根据相应比例划分数据集，但是整个time有什么用，没看出来，难道再做一个验证吗
def get_data(dataset, observe_time, predict_time, restruct_time, train_time, val_time, test_time, time_unit,
             log: logging.Logger, param):
    a = time.time()
    """
    data stores all diffusion behaviors, in the form of (id,src,dst,cas,time). The `id` refers to the
    id of the interaction; `src`,`dst`,`cas`,`time` means that user `dst` forwards the message `cas` from `dst`
    after `time` time has elapsed since the publication of cascade `cas`. 
    -----------------
    metadata stores the metadata of cascades, including the publication time, publication user, etc.
    """
    data: pd.DataFrame = pd.read_csv(f'data/{dataset}.csv')
    metadata = pd.read_csv(f'data/{dataset}_metadata.csv')
    min_time = min(metadata['pub_time'])
    data = pd.merge(data, metadata, left_on='cas', right_on='casid')
    data = data[['id', 'src', 'dst', 'cas', 'time', 'pub_time']]
    param['max_time'] = {'user': 1, 'cas': param['observe_time']}  # user的设为1，级联设置为观测时间
    data['label'] = -1
    data.sort_values(by='id', inplace=True, ignore_index=True)  # 通过id进行了排序
    log.info(
        f"Min time is {min_time}, Train time is {train_time}, Val time is {val_time}, Test time is {test_time}, Time unit is {time_unit}")
    return_data = get_split_data(dataset, observe_time, predict_time, restruct_time, time_unit, data, min_time,
                                 metadata, log, param)
    b = time.time()
    log.info(f"Time cost for loading data is {b - a}s")
    return return_data
