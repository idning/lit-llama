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

    def reset_unused_codebooks(self, mask: torch.Tensor,) -> None:  # replace codebook for mask == Ture, with random tensors from self.most_recent_input
        """
        replace_mask: shape (batch_size, codebook_cardinality)
        """
        with torch.no_grad():
            # assert rank == 0
            random_ids = torch.randint(
                0,
                self.most_recent_input.size(0),
                (self.codebook.size(0),),
                device=self.most_recent_input.device,
            )
            random_tensors = torch.index_select(  # (codebook_cardinality, emb_dim)
                self.most_recent_input, 0, random_ids
            )
            self.codebook.data = torch.where(
                mask[:, None], random_tensors, self.codebook.data
            )

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

        for i in range(self.n_codebook):
            z_q, ind, vq_loss = self.codebooks[i](residual)
            #print('z_q: ', z_q.shape)      # (B, hidden_dim)
            #print('vq_loss: ', vq_loss.shape)

            vq_losses.append(vq_loss)
            residual = residual - z_q

            ind_list.append(ind)
            quant_list.append(z_q)
        # import pdb; pdb. set_trace()
        return torch.stack(quant_list, dim=1).sum(dim=1), torch.stack(ind_list, dim=1), torch.stack(vq_losses).sum()

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

def train(epoch):
    model.train()
    train_loss = 0
    idxs = []
    for batch_idx, data in enumerate(train_loader):
        data = data[0]
        x = data.view(data.size(0), -1)
        optimizer.zero_grad()
        x_recon, idx, vq_loss = model(x)
        cosine = F.cosine_similarity(x, x_recon).mean()  # TODO
        dot_sim = (x * x_recon).sum(dim=1).mean()

        mse_loss = F.mse_loss(x_recon, x)
        loss =  mse_loss + .1 * vq_loss

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        idxs.append(idx)

    if epoch % 1 == 0:
        # for i in range(model.n_codebook):
        #     bin_count = torch.cat(idxs)[:, i].view(-1).bincount(minlength=256)

        #     codebook_util = 1 - torch.sum(bin_count == 0).item() / 256
        #     print(f'{i=} {codebook_util=}')
        #     # print(bin_count)
        #     # model.vq[i].reset_unused_codebooks(mask=idx[:, i] == 0)
        print(f"Epoch: {epoch} \t MSE: {mse_loss}, Loss: {train_loss / len(train_loader.dataset):.6f}, cosine: {cosine} dot: {dot_sim}")
        a = x[0][:6]
        b = x_recon[0][:6]

        # import pdb; pdb. set_trace() 

        print(a)
        print(b)
        print(f'{(a*a).sum()=} {((a-b) * (a-b)).sum()=}')



import torch
from torch.utils.data import TensorDataset, DataLoader
# x = torch.load('ks1.pt').view(-1, 128)[:10*1024*1024]

# x = torch.load('ks1.pt').view(-1, 128)[:1024*10]
# x = torch.load('layer0_all_head_ks.pt').view(-1, 128)[:1024*1000]
# x = torch.load('ks_26G.pt').view(-1, 128)[:1024*1000]

# [s, h, d]
# x = torch.load('shd_ks.pt')[:,0,:].view(-1, 128)[:1024*10]
# x = torch.load('shd_ks.pt').view(-1, 128)[:1024*10]

x = torch.load('slhd_ks.pt').view(-1, 128*32*32)[:1024*10]

print(x.shape)
x = x.to(device).to(torch.float32)
idx = torch.randperm(x.size(0))
x = x[idx]


mean = torch.mean(x)
variance = torch.var(x)
print(f'{mean=} {variance=}, {x.max()=}, {x.min()=}')
# x = (x - mean) / torch.sqrt(variance)

# l2_norm = torch.norm(t, p=2)
# print(f'{l2_norm=}')
# t = t/l2_norm

# t = F.normalize(t, dim=-1, p=2.0)

# x = torch.randn(1024*10, 128).to(device)
train_loader = DataLoader(TensorDataset(x), batch_size=32)


# model = RQ(hidden_dim=128, k=256, n_codebook=4).to(device)
model = RQ(hidden_dim=128*32*32, k=256, n_codebook=8).to(device)
# model = RQVAE(hidden_dim=128, k=4096, n_codebook=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 1000
for epoch in range(1, num_epochs + 1):
    train(epoch)

ckpt = {
    "model_state_dict": model.state_dict(),
}

torch.save(obj=ckpt,f='rqvae.pt')
