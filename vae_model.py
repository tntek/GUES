import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# from ...data.datasets import BaseDataset
# from .vae_config import VAEConfig

def InfoNCE(mu, z):
    mu = mu.unsqueeze(0)
    z = z.unsqueeze(1)
    score = -((z-mu)**2).sum(-1)/20.  # 80 for ImageNet
    lower_bound = -score.logsumexp(dim=1).mean()
    return lower_bound

class VAE(torch.nn.Module):
    """Vanilla Variational Autoencoder model.

    Args:
        model_config (VAEConfig): The Variational Autoencoder configuration setting the main
        parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        encoder,
        decoder,
        reconstruction_loss
    ):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss = reconstruction_loss

    def forward(self, x,x2,cal_loss=True,auto=False):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        encoder_output = self.encoder(x)

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        log_var = log_var+1e-5
        std = torch.exp(0.5 * log_var + 1e-5)
        z, eps = self._sample_gauss(mu, std)
        recon_x = self.decoder(z)["reconstruction"]
        # recon_x = self.decoder(mu)["reconstruction"]
        # random_noise = torch.rand((3, 224, 224)).cuda()
        recon_x_2 = 0.2*recon_x + 0.8*x
        # x2 = 0.8*x2 + 0.2*recon_x.detach()
        # x2 = x2 - random_noise.detach()
        if(cal_loss==True):
            # loss, recon_loss, kld= self.loss_function(recon_x, x2, mu, log_var, z,x)
            loss, recon_loss, kld= self.loss_function(recon_x_2, x2, mu, log_var, z,x)
            return loss, recon_loss, kld,recon_x_2,recon_x
        if(auto ==True):
            recon_loss = 0.5 * F.mse_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1).mean(dim=0)
            return recon_loss,recon_x_2
        else:
            return recon_x_2
        


    def loss_function(self, recon_x, x, mu, log_var, z,x_ori):
        if self.reconstruction_loss == "mse":
            recon_loss = 0.5 * F.mse_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)
            # l1_loss = torch.nn.L1Loss(reduction='mean')
            # recon_loss = l1_loss(recon_x,x)

            # recon_loss = recon_loss + 0*torch.norm(x)
            # recon_loss = InfoNCE(recon_x,x)
            # recon_loss_2 = InfoNCE(recon_x,x_ori)
        elif self.reconstruction_loss == "bce":
            recon_loss = F.binary_cross_entropy(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        return (recon_loss + KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)
        # return (recon_loss + KLD).mean(dim=0), recon_loss, KLD.mean(dim=0),recon_loss_2

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def get_nll(self, data, n_samples=1, batch_size=100):
        """
        Function computed the estimate negative log-likelihood of the model. It uses importance
        sampling method with the approximate posterior distribution. This may take a while.

        Args:
            data (torch.Tensor): The input data from which the log-likelihood should be estimated.
                Data must be of shape [Batch x n_channels x ...]
            n_samples (int): The number of importance samples to use for estimation
            batch_size (int): The batchsize to use to avoid memory issues
        """

        if n_samples <= batch_size:
            n_full_batch = 1
        else:
            n_full_batch = n_samples // batch_size
            n_samples = batch_size

        log_p = []

        for i in range(len(data)):
            x = data[i].unsqueeze(0)

            log_p_x = []

            for j in range(n_full_batch):
                x_rep = torch.cat(batch_size * [x])

                encoder_output = self.encoder(x_rep)
                mu, log_var = encoder_output.embedding, encoder_output.log_covariance

                std = torch.exp(0.5 * log_var)
                z, _ = self._sample_gauss(mu, std)

                log_q_z_given_x = -0.5 * (
                    log_var + (z - mu) ** 2 / torch.exp(log_var)
                ).sum(dim=-1)
                log_p_z = -0.5 * (z**2).sum(dim=-1)

                recon_x = self.decoder(z)["reconstruction"]

                if self.model_config.reconstruction_loss == "mse":
                    log_p_x_given_z = -0.5 * F.mse_loss(
                        recon_x.reshape(x_rep.shape[0], -1),
                        x_rep.reshape(x_rep.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1) - torch.tensor(
                        [np.prod(self.input_dim) / 2 * np.log(np.pi * 2)]
                    ).to(
                        data.device
                    )  # decoding distribution is assumed unit variance  N(mu, I)

                elif self.model_config.reconstruction_loss == "bce":
                    log_p_x_given_z = -F.binary_cross_entropy(
                        recon_x.reshape(x_rep.shape[0], -1),
                        x_rep.reshape(x_rep.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1)

                log_p_x.append(
                    log_p_x_given_z + log_p_z - log_q_z_given_x
                )  # log(2*pi) simplifies

            log_p_x = torch.cat(log_p_x)

            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())
        return np.mean(log_p)

class VAE_new(torch.nn.Module):
    """Vanilla Variational Autoencoder model.

    Args:
        model_config (VAEConfig): The Variational Autoencoder configuration setting the main
        parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        encoder,
        decoder,
        reconstruction_loss
    ):
        super(VAE_new, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss = reconstruction_loss

    def forward(self,x,x2,cal_loss=True,auto=False):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        encoder_output = self.encoder(x)
        mu = encoder_output.latent_dist.mean
        log_var = encoder_output.latent_dist.logvar
        # mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        # log_var = log_var
        std = encoder_output.latent_dist.std
        z, eps = self._sample_gauss(mu, std)#[64,4,28,28]
        recon_x = self.decoder(z)["reconstruction"]
        # recon_x = self.decoder(mu)["reconstruction"]
        recon_x_2 = 0.1*recon_x + 0.9*x
        if(cal_loss==True):
            loss, recon_loss, kld ,recon_loss_2= self.loss_function(recon_x, x2, mu, log_var, z,x)
            return loss, recon_loss, kld,recon_x_2,recon_x,recon_loss_2
        if(auto ==True):
            recon_loss = 0.5 * F.mse_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1).mean(dim=0)
            return recon_loss,recon_x_2
        else:
            return recon_x_2
        


    def loss_function(self, recon_x, x, mu, log_var, z,x_ori):
        if self.reconstruction_loss == "mse":
            recon_loss = 0.5 * F.mse_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)
            # recon_loss = InfoNCE(recon_x,x)
            recon_loss_2 = InfoNCE(recon_x,x_ori)
        elif self.reconstruction_loss == "bce":
            recon_loss = F.binary_cross_entropy(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        return (recon_loss + KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0),recon_loss_2
        # return (recon_loss + KLD).mean(dim=0), recon_loss, KLD.mean(dim=0),recon_loss_2

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def get_nll(self, data, n_samples=1, batch_size=100):
        """
        Function computed the estimate negative log-likelihood of the model. It uses importance
        sampling method with the approximate posterior distribution. This may take a while.

        Args:
            data (torch.Tensor): The input data from which the log-likelihood should be estimated.
                Data must be of shape [Batch x n_channels x ...]
            n_samples (int): The number of importance samples to use for estimation
            batch_size (int): The batchsize to use to avoid memory issues
        """

        if n_samples <= batch_size:
            n_full_batch = 1
        else:
            n_full_batch = n_samples // batch_size
            n_samples = batch_size

        log_p = []

        for i in range(len(data)):
            x = data[i].unsqueeze(0)

            log_p_x = []

            for j in range(n_full_batch):
                x_rep = torch.cat(batch_size * [x])

                encoder_output = self.encoder(x_rep)
                mu, log_var = encoder_output.embedding, encoder_output.log_covariance

                std = torch.exp(0.5 * log_var)
                z, _ = self._sample_gauss(mu, std)

                log_q_z_given_x = -0.5 * (
                    log_var + (z - mu) ** 2 / torch.exp(log_var)
                ).sum(dim=-1)
                log_p_z = -0.5 * (z**2).sum(dim=-1)

                recon_x = self.decoder(z)["reconstruction"]

                if self.model_config.reconstruction_loss == "mse":
                    log_p_x_given_z = -0.5 * F.mse_loss(
                        recon_x.reshape(x_rep.shape[0], -1),
                        x_rep.reshape(x_rep.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1) - torch.tensor(
                        [np.prod(self.input_dim) / 2 * np.log(np.pi * 2)]
                    ).to(
                        data.device
                    )  # decoding distribution is assumed unit variance  N(mu, I)

                elif self.model_config.reconstruction_loss == "bce":
                    log_p_x_given_z = -F.binary_cross_entropy(
                        recon_x.reshape(x_rep.shape[0], -1),
                        x_rep.reshape(x_rep.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1)

                log_p_x.append(
                    log_p_x_given_z + log_p_z - log_q_z_given_x
                )  # log(2*pi) simplifies

            log_p_x = torch.cat(log_p_x)

            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())
        return np.mean(log_p)
