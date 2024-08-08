import logging
import pickle as pk
import time

import numpy as np
import pandas as pd


class Data:
    def __init__(self, data, is_split=False):
        self.srcs = data['src'].values
        self.dsts = data['dst'].values
        self.times = data['abs_time'].values
        self.trans_cascades = data['cas'].values
        self.pub_times = data['pub_time'].values
        self.labels = data['label'].values
        self.length = len(self.srcs)
        self.is_split = is_split
        if is_split:
            self.types = data['type'].values

    def loader(self, batch):
        for i in range(0, len(self.srcs), batch):
            right = min(i + batch, self.length)
            if self.is_split:
                yield (self.srcs[i:right], self.dsts[i:right], self.trans_cascades[i:right],
                       self.times[i:right], self.pub_times[i:right], self.types[i:right]), self.labels[i:right]
            else:
                yield (self.srcs[i:right], self.dsts[i:right], self.trans_cascades[i:right],
                       self.times[i:right], self.pub_times[i:right]), self.labels[i:right]


def get_label(x: pd.DataFrame, observe_time, label):
    id = np.searchsorted(x['time'], observe_time, side='left')
    casid = x['cas'].values[0]
    if casid in label and id >= 10:
        length = min(id, 100) - 1

        x.iloc[length, x.columns.get_loc('label')] = label[casid] - id

        return [x.iloc[:length + 1, :]]
    else:
        return []




def get_split_data(dataset, observe_time, predict_time, restruct_time, time_unit, all_data, min_time, metadata, log,
                   param):
    def data_split(legal_cascades, train_portion=0.7, val_portion=0.15):
        """
        set cas type, 1 for train cas, 2 for val cas, 3 for test cas , and 0 for other cas that will be dropped
        """
        m_metadata = metadata[metadata['casid'].isin(set(legal_cascades))]
        all_idx, type_map = {}, {}
        if dataset == 'twitter':
            dt = pd.to_datetime(m_metadata['pub_time'], unit='s', utc=True).dt.tz_convert('Asia/Shanghai')
            idx = dt.apply(lambda x: not (x.month == 4 and x.day > 10)).values
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


    all_label = all_data[all_data['time'] < predict_time * time_unit].groupby(by='cas', as_index=False)['id'].count()
    all_label = dict(zip(all_label['cas'], all_label['id']))
    m_data = []
    all_data_condition = all_data.copy(deep=True)
    for cas, df in all_data.groupby(by='cas'):
        m_data.extend(get_label(df, observe_time * time_unit, all_label))
    all_data = pd.concat(m_data, axis=0)
    num_timestamps = restruct_time - observe_time
    all_idx, type_map = data_split(all_data[all_data['label'] != -1]['cas'].values)
    all_data['type'] = all_data['cas'].apply(lambda x: type_map[x])
    all_data = all_data[all_data['type'] != 0]
    cas_id = all_data['cas'].unique()
    all_data_condition = all_data_condition[all_data_condition['cas'].isin(cas_id)]
    cas_popularity_dict = {}
    for cas, df in all_data_condition.groupby(by='cas'):
        popularity_array = np.zeros(num_timestamps)
        for ts in range(num_timestamps):
            popularity_array[ts] = df[df['time'] <= (observe_time + ts) * time_unit]['id'].count()-df[df['time'] <= observe_time  * time_unit]['id'].count()
        cas_popularity_dict[cas] = popularity_array
    cas_popularity = data_transformation(dataset, all_data, cas_popularity_dict, time_unit, min_time,
                                         param)
    pk.dump(all_idx, open(f'data/{dataset}_idx.pkl', 'wb'))
    log.info(
        f"Total Trans num is {len(all_data)}, Train cas num is {len(all_idx['train'])}, "
        f"Val cas num is {len(all_idx['val'])}, Test cas num is {len(all_idx['test'])}")
    return Data(all_data, is_split=True), cas_popularity



def get_data(dataset, observe_time, predict_time, restruct_time, train_time, val_time, test_time, time_unit,
             log: logging.Logger, param):
    a = time.time()
    data: pd.DataFrame = pd.read_csv(f'data/{dataset}.csv')
    metadata = pd.read_csv(f'data/{dataset}_metadata.csv')
    min_time = min(metadata['pub_time'])
    data = pd.merge(data, metadata, left_on='cas', right_on='casid')
    data = data[['id', 'src', 'dst', 'cas', 'time', 'pub_time']]
    param['max_time'] = {'user': 1, 'cas': param['observe_time']}
    data['label'] = -1
    data.sort_values(by='id', inplace=True, ignore_index=True)
    return_data = get_split_data(dataset, observe_time, predict_time, restruct_time, time_unit, data, min_time,
                                 metadata, log, param)
    b = time.time()
    log.info(f"Time cost for loading data is {b - a}s")
    return return_data
