import numpy as np
import torch
import torch.nn as nn
from typing import Dict
from model.encoder.state.dynamic_state import DynamicState
from model.encoder.state.state_updater import get_state_updater
from model.encoder.embedding_module import get_embedding_module
from model.encoder.message.message_generator import get_message_generator
from model.decoder.prediction import get_predictor
from model.time_encoder import get_time_encoder
from utils.hgraph import HGraph
from model.decoder.cas_ode import CasODE
from model.encoder.encoder_z0 import EncodeZ0
from model.decoder.memory import ExternalMemory


class ODEPT(nn.Module):
    def __init__(self, args, device: torch.device, time_steps_to_predict, node_dim: int = 100,
                 embedding_module_type: str = "seq",
                 state_updater_type: str = "gru", predictor: str = 'linear', time_enc_dim: int = 8,
                 single: bool = False, ntypes: set = None, dropout: float = 0.1, n_nodes: Dict = None,
                 max_time: float = None, use_static: bool = False, merge_prob: float = 0.5,
                 max_global_time: float = 0, use_dynamic: bool = False, use_temporal: bool = False,
                 use_structural: bool = False):
        super(ODEPT, self).__init__()
        if max_time is None:
            max_time = {'user': 1, 'cas': 1}
        self.ntypes = ntypes
        self.device = device
        self.cas_num = n_nodes['cas']
        self.user_num = n_nodes['user']
        self.time_steps_to_predict = time_steps_to_predict.float().to(self.device)
        self.single = single
        self.hgraph = HGraph(num_user=n_nodes['user'], num_cas=n_nodes['cas'])
        self.time_encoder = get_time_encoder('difference', dimension=time_enc_dim, single=self.single)
        self.use_dynamic = use_dynamic
        self.node_dim = node_dim
        self.args = args
        self.dynamic_state = nn.ModuleDict({
            'user': DynamicState(n_nodes['user'], state_dimension=node_dim,
                                 input_dimension=node_dim, message_dimension=node_dim,
                                 device=device, single=False),
            'cas': DynamicState(n_nodes['cas'], state_dimension=node_dim,
                                input_dimension=node_dim, message_dimension=node_dim,
                                device=device, single=True)})
        self.init_state()
        self.message_generator = get_message_generator(generator_type='concat', state=self.dynamic_state,
                                                       time_encoder=self.time_encoder,
                                                       time_dim=time_enc_dim,
                                                       message_dim=node_dim, node_feature_dim=node_dim,
                                                       device=self.device, message_aggregator_type='mean',
                                                       single=single, max_time=max_time)

        self.state_updater = get_state_updater(module_type=state_updater_type,
                                               state=self.dynamic_state,
                                               message_dimension=node_dim,
                                               state_dimension=node_dim,
                                               device=self.device, single_updater=single, ntypes=ntypes)
        self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                     dynamic_state=self.dynamic_state, embedding_dimension=node_dim,
                                                     device=self.device, dropout=dropout, hgraph=self.hgraph,
                                                     input_dimension=node_dim, max_time=max_time['cas'],
                                                     use_static=use_static, user_num=n_nodes['user'],
                                                     max_global_time=max_global_time, use_dynamic=use_dynamic,
                                                     use_temporal=use_temporal, use_structural=use_structural)
        self.predictor = get_predictor(emb_dim=node_dim, predictor_type=predictor, merge_prob=merge_prob)

        self.encoder_z0 = EncodeZ0(emb_dim=node_dim)
        self.external_memory = ExternalMemory(cascade_dim=node_dim, memory_size=args['memory_size'], device=device)
        self.cas_ode = CasODE(ode_hidden_dim=node_dim, args=args, device=device, dropout=dropout,
                              external_memory=self.external_memory)

    def update_state(self):
        if self.use_dynamic:
            for ntype in self.ntypes:
                self.dynamic_state[ntype].store_cache()

    def forward(self, source_nodes: np.ndarray, destination_nodes: np.ndarray, trans_cascades: np.ndarray,
                edge_times: torch.Tensor, pub_times: torch.Tensor, target_idx: np.ndarray):
        """
        given a batch of interactions, update the corresponding nodes' dynamic states and give the popularity of the
        cascades that have reached the observation time.
        :param source_nodes: the sending users' id of the interactions, ndarray of shape (batch)
        :param destination_nodes: the receiving users' id of the interactions, ndarray of shape (batch)
        :param trans_cascades: the cascade id of the interactions,ndarray of shape (batch)，级联的id
        :param edge_times: the happening timestamps of the interactions, tensor of shape (batch)
        :param pub_times: the publication timestamps of the cascades in the interactions, tensor of shape (batch)
        :param target_idx: a mask tensor to indicating which cascade has reached the observation time,
               tensor of shape (batch)表面是否达到要预测的时间
        :return pred: the popularity of cascades that have reached the observation time, tensor of shape (batch)
        """
        if self.use_dynamic:
            nodes, messages, times = self.message_generator.get_message(source_nodes, destination_nodes,
                                                                        trans_cascades, edge_times, pub_times, 'all')
            self.state_updater.update_state(nodes, messages, times)
        self.hgraph.insert(trans_cascades, source_nodes, destination_nodes, edge_times,
                           pub_times)  # 把这一个batch的数据插入到图中，形成图
        target_cascades = trans_cascades[target_idx]  # 这个真的可以做到吗，表面是否到达预测时间
        pred = torch.zeros(len(trans_cascades), len(self.time_steps_to_predict)).to(self.device)
        first_point = torch.zeros(len(trans_cascades), self.node_dim, 2).to(self.device)
        if len(target_cascades) > 0:  # 存在到达了观测时间的级联，
            emb = self.embedding_module.compute_embedding(target_cascades)  # 这就对了，针对target来做计算embedding
            first_point_nor = self.encoder_z0(emb)
            pred[target_idx], first_point[target_idx] = self.cas_ode.get_reconstruction(first_point_nor=first_point_nor,
                                                                                        time_steps_to_predict=self.time_steps_to_predict)  # 这个embedding是对什么的，这个不太对吧
            if self.args['self_evolution']:
                pass
            else:
                self.external_memory.update_memory(emb)
        return pred, first_point

    def init_state(self):
        for ntype in self.ntypes:
            self.dynamic_state[ntype].__init_state__()
        self.hgraph.init()

    def reset_state(self):
        for ntype in self.ntypes:
            self.dynamic_state[ntype].reset_state()
        self.hgraph.init()

    def detach_state(self):
        for ntype in self.ntypes:
            self.dynamic_state[ntype].detach_state()
