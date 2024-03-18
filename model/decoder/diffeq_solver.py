import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
from model.decoder.gnn import CasSelf


class DiffeqSolver(nn.Module):
    def __init__(self, ode_func, method="euler",
                 odeint_rtol=1e-3, odeint_atol=1e-4, device=torch.device("cpu")):
        super(DiffeqSolver, self).__init__()

        self.ode_method = method
        self.device = device
        self.ode_func = ode_func

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_point, time_steps_to_predict):
        '''

        :param first_point:  [K*N,D]
        :param edge_initials: [K*N*N,D]
        :param time_steps_to_predict: [t]
        :return:
        '''

        pred_y = odeint(self.ode_func, first_point, time_steps_to_predict,
                        rtol=self.odeint_rtol, atol=self.odeint_atol,
                        method=self.ode_method)
        pred_y=pred_y.permute(1, 0, 2)

        return pred_y


class CasODEFunc(nn.Module):
    def __init__(self, input_dim, output_dim, device, dropout=0.2):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(CasODEFunc, self).__init__()

        self.device = device
        self.func_net = CasSelf(input_dim, output_dim, dropout=dropout).to(device)
        self.dropout = nn.Dropout(dropout)

    # 为什么返回的是梯度
    def forward(self, t_local, z, backwards=False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        z:  [H,E] concat by axis0. H is [K*N,D], E is[K*N*N,D], z is [K*N + K*N*N, D]
        """

        assert (not torch.isnan(z).any())
        grad_dy = self.func_net(z)

        return grad_dy
