import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import os
from rqvae import RQ, RQVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_on_tensor(x: torch.Tensor):
    print(x.shape)
    x = x.to(device).to(torch.float32)
    idx = torch.randperm(x.size(0))
    x = x[idx]

    train_loader = DataLoader(TensorDataset(x), batch_size=1024)

    model = RQ(hidden_dim=128*32, k=256, n_codebook=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 200

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

        if epoch % 10 == 0:
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

    for epoch in range(1, num_epochs + 1):
        train(epoch)

    return model


import torch
from torch.utils.data import TensorDataset, DataLoader

runid = 'rqvae_200_epoch'
os.makedirs(f'{runid}', exist_ok=True)

models = {
    'k': [],
    'v': [],
}
x = torch.load('slhd_ks.pt').to(device)
for i in range(32): 
    print(f'=====================k:{i}')
    m = train_on_tensor(x[:,0,:,:].view(-1, 128*32)[:1024*10])
    torch.jit.save(torch.jit.script(m), f'{runid}/k{i:02}.jit.pt')

x = torch.load('slhd_vs.pt').to(device)
for i in range(32): 
    print(f'=====================v:{i}')
    m = train_on_tensor(x[:,0,:,:].view(-1, 128*32)[:1024*10])
    torch.jit.save(torch.jit.script(m), f'{runid}/v{i:02}.jit.pt')

# ckpt = {
#     "k": [m.state_dict() for m in models['k']],
#     "v": [m.state_dict() for m in models['v']],
# }

# torch.save(obj=ckpt,f='rqvae_combined.pt')
