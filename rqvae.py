import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VQ(nn.Module):
    def __init__(self, hidden_dim=256, k=512):
        super(VQ, self).__init__()
        self.codebook = nn.Embedding(k, hidden_dim)
        # self.codebook = nn.Embedding(k, hidden_dim, dtype=torch.bfloat16)
        self.codebook.weight.data.uniform_(-1/k, 1/k)

    def forward(self, z):
        self.most_recent_input = z
        # Calculate distances from the codebook
        distances = (z.unsqueeze(-2) - self.codebook.weight)**2.0  # (B, k, 256)
        distances = distances.sum(-1)  # (B, k)

        # Assign each z to the closest code in the codebook
        _, ind = distances.min(-1)  # (B)

        z_q = self.codebook(ind)

        vq_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z, z_q.detach())

        z_q_stright_through = z + (z_q - z).detach()
        return z_q_stright_through, ind, vq_loss + commitment_loss

    def decode(self, idx):
        return self.codebook(idx)


class RQ(nn.Module):
    def __init__(self, hidden_dim=256, k=512, n_codebook=3):
        super(RQ, self).__init__()
        self.n_codebook = n_codebook
        codebooks = [VQ(hidden_dim, k) for i in range(n_codebook)]
        self.codebooks = nn.ModuleList(codebooks)

    def forward(self, z):
        residual = z

        quant_list = []
        ind_list = []
        vq_losses = []

        for i, codebook in enumerate(self.codebooks):
            z_q, ind, vq_loss = codebook(residual)
            #print('z_q: ', z_q.shape)      # (B, hidden_dim)
            #print('vq_loss: ', vq_loss.shape)

            vq_losses.append(vq_loss)
            residual = residual - z_q

            ind_list.append(ind)
            quant_list.append(z_q)
        # import pdb; pdb. set_trace()
        return torch.stack(quant_list, dim=1).sum(dim=1), torch.stack(ind_list, dim=1), torch.stack(vq_losses).sum()
    
    def decode(self, idx):
        return torch.stack([codebook.decode(idx[:,i]) for i, codebook in enumerate(self.codebooks)], dim=1).sum(dim=1)


class RQVAE(nn.Module):
    def __init__(self, in_features=128, encoder_dims = [128], hidden_dim=128, k=256, n_codebook=3):
        super(RQVAE, self).__init__()

        # Encoder
        layers = []
        all_dims = [in_features] + encoder_dims + [hidden_dim]
        for a,b in zip(all_dims[:-1], all_dims[1:]):
            layers.append(nn.Linear(a, b))
            layers.append(nn.ReLU())
        self.enc = nn.Sequential(*layers[:-1])  # do not include the last Relu

        # Codebook
        # self.vq = VQ(hidden_dim=hidden_dim, k=k)
        self.vq = RQ(hidden_dim=hidden_dim, k=k, n_codebook=n_codebook)

        # Decoder
        layers = []
        all_dims = list(reversed(all_dims))
        for a,b in zip(all_dims[:-1], all_dims[1:]):
            layers.append(nn.Linear(a, b))
            layers.append(nn.ReLU())
        # layers[-1] = nn.Sigmoid()
        self.dec = nn.Sequential(*layers[:-1])  # do not include the last Relu

    def forward(self, x):
        z_e = self.enc(x)
        z_q, ind, vq_loss = self.vq(z_e)
        x_recon = self.dec(z_q)
        return x_recon, ind, vq_loss
        # return x_recon, z_e, z_q, vq_loss
