
import torch
import numpy as np
import utils.utils as utils
import torch.nn.functional as F
import torch.nn as nn
from model.decoder.ode_fun import CasSelf
from model.decoder.diffeq_solver import DiffeqSolver
from model.decoder.diffeq_solver import CasODEFunc


class CasODE(nn.Module):
    def __init__(self, ode_hidden_dim, args, device,external_memory, output_dim=1,dropout=0.2):
        super(CasODE, self).__init__()
        self.args = args
        self.ode_fun = CasODEFunc(ode_hidden_dim, ode_hidden_dim, device=device,dropout=dropout,external_memory=external_memory,params=args)
        self.diffeq_solver = DiffeqSolver(self.ode_fun, method=args['solver'])
        self.decoder = Decoder(latent_dim=ode_hidden_dim, output_dim=output_dim)
        self.ode_hidden_dim = ode_hidden_dim


    def get_reconstruction(self, first_point_nor, time_steps_to_predict):
        # Encoder:
        first_point_mu, first_point_std = first_point_nor
        assert (not torch.isnan(first_point_std).any())
        assert (not torch.isnan(first_point_mu).any())

        first_point_enc = utils.sample_standard_gaussian(first_point_mu, first_point_std)

        first_point_std = first_point_std.abs()

        assert (torch.sum(first_point_std < 0) == 0.)
        assert (not torch.isnan(time_steps_to_predict).any())

        assert (not torch.isnan(first_point_enc).any())


        sol_y = self.diffeq_solver(first_point_enc, time_steps_to_predict)

        assert (not torch.isnan(sol_y).any())

        # Decoder:
        pred = self.decoder(sol_y)
        pred = pred.squeeze(dim=2)


        first_point = torch.stack((first_point_mu, first_point_std), dim=2)

        return pred, first_point


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, decoder_network=None):
        super(Decoder, self).__init__()

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
