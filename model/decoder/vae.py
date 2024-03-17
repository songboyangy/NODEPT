from utils.likelihood_eval import *
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import torch.nn as nn
import torch
import utils.utils as utils
import numpy as np


class VAE_Baseline(nn.Module):
    def __init__(self,
                 z0_prior, device,
                 obsrv_std=0.01,
                 ):

        super(VAE_Baseline, self).__init__()

        self.device = device

        self.obsrv_std = torch.Tensor([obsrv_std]).to(device)

        self.z0_prior = z0_prior

    # 计算重构损失，类似于平方误差
    def get_gaussian_likelihood(self, truth, pred_y, temporal_weights=None, mask=None):
        # pred_y shape [K*N, n_tp, n_dim]
        # truth shape  [K*N, n_tp, n_dim]

        # Compute likelihood of the data under the predictions

        log_density_data = masked_gaussian_log_density(pred_y, truth,
                                                       obsrv_std=self.obsrv_std, mask=mask,
                                                       temporal_weights=temporal_weights)  # 【num_traj = K*N] [250,3]
        log_density = torch.mean(log_density_data)

        # shape: [n_traj_samples]
        return log_density

    # 获取loss
    def get_loss(self, truth, pred_y, truth_gt=None, mask=None, method='MSE', istest=False):
        # pred_y shape [n_traj, n_tp, n_dim]，一共n个时间戳
        # truth shape  [n_traj, n_tp, n_dim]

        # Transfer from inc to cum   从增量转化为累计，为什么要这么设计，本身预测的是增量吗

        truth = utils.inc_to_cum(truth)
        pred_y = utils.inc_to_cum(pred_y)
        num_times = truth.shape[1]
        time_index = [num_times - 1]  # last timestamp，为什么这么计算

        if istest:
            truth = truth[:, time_index, :]
            pred_y = pred_y[:, time_index, :]  # [N,1,D]
            if truth_gt != None:
                truth_gt = truth_gt[:, time_index, :]

        # Compute likelihood of the data under the predictions，pred_y是预测，truth是真实值，mask的作用
        log_density_data = compute_loss(pred_y, truth, truth_gt, mask=mask, method=method)
        # shape: [1]
        return torch.mean(log_density_data)

    #
    def print_out_pred(self, pred_node, pred_edge):

        pred_node = pred_node  # [N,T,D]
        pred_node = utils.inc_to_cum(pred_node)

        num_times = pred_node.shape[1]
        time_index = [num_times - 1]  # last timestamp

        pred_node = pred_node[:, time_index, :]

        pred_node = torch.squeeze(pred_node)

        pred_node = pred_node.cpu().detach().tolist()  # [N,1,D]
        return utils.print_MAPE(pred_node)

    # 预测值的累计
    def print_out_pred_sum(self, pred_node):
        pred_node = pred_node  # [N,T,D]
        pred_node = utils.inc_to_cum(pred_node)

        num_times = pred_node.shape[1]
        time_index = [num_times - 1]  # last timestamp

        pred_node = pred_node[:, time_index, :]

        pred_node = torch.squeeze(pred_node)

        pred_node = pred_node.cpu().detach().numpy()  # [N,1,D]

        print(np.sum(pred_node))

    # 计算所有的loss，包括所有的评价指标，具体怎么用，还要看接下来
    def compute_all_losses(self, batch_dict_encoder, batch_dict_decoder, batch_dict_graph, num_atoms, edge_lamda,
                           kl_coef=1., istest=False):
        '''

		:param batch_dict_encoder:
		:param batch_dict_decoder: dict: 1. time 2. data: (K*N, T2, D)
		:param batch_dict_graph: #[K,T2,N,N], ground_truth graph with log normalization
		:param num_atoms:
		:param kl_coef:
		:return:
		'''
        # 下面的方法获得预测值
        pred_node, pred_edge, info, temporal_weights = self.get_reconstruction(batch_dict_encoder, batch_dict_decoder,
                                                                               num_atoms=num_atoms)
        # pred_node [ K*N , time_length, d]
        # pred_edge [ K*N*N, time_length, d]

        if istest:
            mask_index = batch_dict_decoder["masks"]  # test传入mask
            pred_node = pred_node[:, mask_index, :]  # 取出mask_index的预测值
            pred_edge = pred_edge[:, mask_index, :]

        # Reshape batch_dict_graph
        k = batch_dict_graph.shape[0]
        T2 = batch_dict_graph.shape[1]
        truth_graph = torch.reshape(batch_dict_graph, (k, T2, -1))  # [K,T,N*N]
        truth_graph = torch.unsqueeze(truth_graph.permute(0, 2, 1), dim=3)  # [K,N*N,T,1]
        truth_graph = torch.reshape(truth_graph, (-1, T2, 1))  # [K*N*N,T,1]

        # print("get_reconstruction done -- computing likelihood")

        # KL divergence only contains node-level (only z_node are sampled, z_edge are computed from z_node)
        fp_mu, fp_std, fp_enc = info["first_point"]  # [K*N,D]
        fp_std = fp_std.abs()

        fp_distr = Normal(fp_mu, fp_std)

        assert (torch.sum(fp_std < 0) == 0.)
        kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)  # [K*N,D_ode_latent]，计算z0_prior与正太分布的kl散度，在base
        # model这里用到了，果然是先验与后验的kl散度

        if torch.isnan(kldiv_z0).any():
            print(fp_mu)
            print(fp_std)
            raise Exception("kldiv_z0 is Nan!")

        if torch.isinf(kldiv_z0).any():
            locations = torch.where(kldiv_z0 == float("inf"), torch.Tensor([1]).to(fp_mu.device),
                                    torch.Tensor([0]).to(fp_mu.device))
            locations = locations.to("cpu").detach().numpy()
            mu_locations = fp_mu.to("cpu").detach().numpy() * locations
            std_locations = fp_std.to("cpu").detach().numpy() * locations
            _, mu_values = utils.convert_sparse(mu_locations)
            _, std_values = utils.convert_sparse(std_locations)
            print(mu_values)
            print(std_values)

        # Mean over number of latent dimensions
        # kldiv_z0 shape: [n_traj, n_latent_dims] if prior is a mixture of gaussians (KL is estimated)
        # kldiv_z0 shape: [1, n_traj, n_latent_dims] if prior is a standard gaussian (KL is computed exactly)
        # shape after: [1]
        kldiv_z0 = torch.mean(kldiv_z0)  # Contains infinity.

        # Compute likelihood of all the points
        rec_likelihood_node = self.get_gaussian_likelihood(
            batch_dict_decoder["data"], pred_node, temporal_weights,
            mask=None)  # negative value，在计算的过程中mask都是none

        rec_likelihood_edge = self.get_gaussian_likelihood(
            truth_graph, pred_edge, temporal_weights,
            mask=None)  # negative value

        rec_likelihood = (1 - edge_lamda) * rec_likelihood_node + edge_lamda * rec_likelihood_edge

        mape_node = self.get_loss(
            batch_dict_decoder["data"], pred_node, truth_gt=batch_dict_decoder["data_gt"],
            mask=None, method='MAPE', istest=istest)  # [1]

        mse_node = self.get_loss(
            batch_dict_decoder["data"], pred_node,
            mask=None, method='MSE', istest=istest)  # [1]

        # loss

        loss = - torch.logsumexp(rec_likelihood - kl_coef * kldiv_z0, 0)
        if torch.isnan(loss):
            loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0, 0)

        results = {}
        results["loss"] = torch.mean(loss)
        results["likelihood"] = torch.mean(rec_likelihood).data.item()
        results["MAPE"] = torch.mean(mape_node).data.item()
        results["MSE"] = torch.mean(mse_node).data.item()
        results["kl_first_p"] = kldiv_z0.detach().data.item()
        results["std_first_p"] = torch.mean(fp_std).detach().data.item()

        # if istest:
        # 	print("Predicted Inc Deaths are:")
        # 	print(self.print_out_pred(pred_node,pred_edge))
        # 	print(self.print_out_pred_sum(pred_node))

        return results
