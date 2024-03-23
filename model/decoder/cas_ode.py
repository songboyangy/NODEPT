# from lib.base_models import VAE_Baseline
import torch
import numpy as np
import utils.utils as utils
import torch.nn.functional as F
import torch.nn as nn
from model.decoder.gnn import CasSelf
from model.decoder.diffeq_solver import DiffeqSolver
from model.decoder.diffeq_solver import CasODEFunc


class CasODE(nn.Module):
    def __init__(self, ode_hidden_dim, args, device,external_memory, output_dim=1,dropout=0.2):
        super(CasODE, self).__init__()
        self.args = args
        self.ode_fun = CasODEFunc(ode_hidden_dim, ode_hidden_dim, device=device,dropout=dropout,external_memory=external_memory)
        self.diffeq_solver = DiffeqSolver(self.ode_fun, method=args['solver'])
        self.decoder = Decoder(latent_dim=ode_hidden_dim, output_dim=output_dim)
        self.ode_hidden_dim = ode_hidden_dim

    # 获得重构，预测后一段时间戳的值，
    def get_reconstruction(self, first_point_nor, time_steps_to_predict):
        # Encoder:
        first_point_mu, first_point_std = first_point_nor  # 初始点均值方差
        assert (not torch.isnan(first_point_std).any())
        assert (not torch.isnan(first_point_mu).any())

        first_point_enc = utils.sample_standard_gaussian(first_point_mu, first_point_std)  # [K*N,D]，获取初始值，这个就是点的特征，没错

        first_point_std = first_point_std.abs()

        assert (torch.sum(first_point_std < 0) == 0.)
        assert (not torch.isnan(time_steps_to_predict).any())

        assert (not torch.isnan(first_point_enc).any())

        # ODE:Shape of sol_y #[ K*N + K*N*N, time_length, d], concat of node and edge.，求解出方程，这个应该包含多个时间步
        # K_N is the index for node.返回的是多个时间戳的解，应该是对应的多个时间戳的数值
        sol_y = self.diffeq_solver(first_point_enc, time_steps_to_predict)

        assert (not torch.isnan(sol_y).any())

        # Decoder:
        pred = self.decoder(sol_y)
        pred = pred.squeeze(dim=2)

        # all_extra_info = {
        #     "first_point": (first_point_mu, first_point_std, first_point_enc),  # 在这个地方将first_point_mu传递出去
        #     "latent_traj": sol_y.detach()
        # }
        first_point = torch.stack((first_point_mu, first_point_std), dim=2)

        return pred, first_point


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, decoder_network=None):
        super(Decoder, self).__init__()
        # decode data from latent space where we are solving an ODE back to the data space
        if decoder_network == None:
            decoder = nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 2),
                nn.ReLU(),
                nn.Linear(latent_dim // 2, output_dim),
            )
            utils.init_network_weights(decoder)
        else:
            decoder = decoder_network

        self.decoder = decoder

    def forward(self, data):
        return self.decoder(data)
