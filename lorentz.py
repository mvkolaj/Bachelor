import os
import sys
import torch
import random
import numpy as np
from torch import nn
from torch import optim
from tqdm import trange, tqdm
from collections import Counter
from datetime import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import datasets
import pickle

import matplotlib

matplotlib.use("Agg")  # this needs to come before other matplotlib imports
import matplotlib.pyplot as plt

plt.style.use("ggplot")


def arcosh(x):
    return torch.log(x + torch.sqrt(x ** 2 - 1))


def lorentz_scalar_product(x, y):
    # BD, BD -> B
    m = x * y
    result = m[:, 1:].sum(dim=1) - m[:, 0]
    return result


def tangent_norm(x):
    # BD -> B
    return torch.sqrt(lorentz_scalar_product(x, x))


def exp_map(x, v):
    # BD, BD -> BD
    tn = tangent_norm(v).unsqueeze(dim=1)
    tn_expand = tn.repeat(1, x.size()[-1])
    result = torch.cosh(tn) * x + torch.sinh(tn) * (v / tn)
    result = torch.where(tn_expand > 0, result, x)  # only update if tangent norm is > 0
    return result


def set_dim0(x):
    x = torch.renorm(x, p=2, dim=0, maxnorm=1e2)  # otherwise leaves will explode
    # NOTE: the paper does not mention the square part of the equation but if
    # you try to derive it you get a square term in the equation
    dim0 = torch.sqrt(1 + (x[:, 1:] ** 2).sum(dim=1))
    x[:, 0] = dim0
    return x


# ========================= models


class RSGD(optim.Optimizer):
    def __init__(self, params, learning_rate=None):
        learning_rate = learning_rate if learning_rate is not None else 0.01
        defaults = {"learning_rate": learning_rate}
        super().__init__(params, defaults=defaults)

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                B, D = p.size()
                gl = torch.eye(D, device=p.device, dtype=p.dtype)
                gl[0, 0] = -1
                h = p.grad.data @ gl          
                proj = (
                    h
                    - (
                        lorentz_scalar_product(p, h) / lorentz_scalar_product(p, p)
                    ).unsqueeze(1)
                    * p
                )
                update = exp_map(p, -group["learning_rate"] * proj)
                is_nan_inf = torch.isnan(update) | torch.isinf(update)
                update = torch.where(is_nan_inf, p, update)
                update[0, :] = p[0, :]
                update = set_dim0(update)
                p.data.copy_(update)


class Lorentz(nn.Module):
    """
    This will embed `n_items` in a `dim` dimensional lorentz space.
    """

    def __init__(self, n_items, dim, init_range=0.001):
        super().__init__()
        self.n_items = n_items
        self.dim = dim
        self.table = nn.Embedding(n_items + 1, dim, padding_idx=0)
        nn.init.uniform_(self.table.weight, -init_range, init_range)
        # equation 6
        with torch.no_grad():
            self.table.weight[0] = 5  # padding idx push it to corner
            set_dim0(self.table.weight)

    def forward(self, I, Ks):
        """
        Using the pairwise similarity matrix, generate the following inputs and
        provide to this function.

        Inputs:
            - I     :   - long tensor
                        - size (B,)
                        - This denotes the `i` used in all equations.
            - Ks    :   - long tensor
                        - size (B, N)
                        - This denotes at max `N` documents which come from the
                          nearest neighbor sample.
                        - The `j` document must be the first of the N indices.
                          This is used to calculate the losses
        Return:
            - size (B,)
            - Ranking loss calculated using
              document to the given `i` document.

        """
        n_ks = Ks.size()[1]
        ui = torch.stack([self.table(I)] * n_ks, dim=1)
        uks = self.table(Ks)
        # ---------- reshape for calculation
        B, N, D = ui.size()
        ui = ui.reshape(B * N, D)
        uks = uks.reshape(B * N, D)
        dists = -lorentz_scalar_product(ui, uks)
        dists = torch.where(dists <= 1, torch.ones_like(dists) + 1e-6, dists)
        # sometimes 2 embedding can come very close in R^D.
        # when calculating the lorenrz inner product,
        # -1 can become -0.99(no idea!), then arcosh will become nan
        dists = -arcosh(dists)
        # print(dists)
        # ---------- turn back to per-sample shape
        dists = dists.reshape(B, N)
        loss = -(dists[:, 0] - torch.log(torch.exp(dists).sum(dim=1) + 1e-6))
        return loss

    def lorentz_to_poincare(self):
        table = self.table.weight.data.cpu().numpy()
        return table[:, 1:] / (
            table[:, :1] + 1
        )  # diffeomorphism transform to poincare ball

    def get_lorentz_table(self):
        return self.table.weight.data.cpu().numpy()

    def _test_table(self):
        x = self.table.weight.data
        check = lorentz_scalar_product(x, x) + 1.0
        return check.cpu().numpy().sum()


class Graph(Dataset):
    def __init__(self, pairwise_matrix, batch_size, sample_size=10):
        self.pairwise_matrix = pairwise_matrix
        self.n_items = pairwise_matrix.shape[0]
        self.sample_size = int(sample_size)
        self.arange = np.arange(0, self.n_items)
        self.cnter = 0
        self.batch_size = int(batch_size)

    def __len__(self):
        return self.n_items

    def __getitem__(self, i):
        self.cnter = (self.cnter + 1) % self.batch_size

        # central node: 1..n, because 0 is padding
        I = torch.tensor(i + 1, dtype=torch.long)

        row = self.pairwise_matrix[i]

        # prefer non-self positive neighbors
        pos_candidates = np.where((row > 0) & (self.arange != i))[0]

        if len(pos_candidates) == 0:
            # fallback: allow self-loop only if absolutely necessary
            if row[i] > 0:
                j = i
                min_w = float(row[i])
            else:
                raise Exception(f"Node {i} has no neighbors")
        else:
            j = int(np.random.choice(pos_candidates))
            min_w = float(row[j])

        neg_candidates = np.where((row < min_w) & (self.arange != i))[0]
        np.random.shuffle(neg_candidates)
        negs = neg_candidates[: max(0, self.sample_size - 1)]

        # use -1 as padding placeholder, then shift everything by +1
        Ks_np = np.full((self.sample_size,), -1, dtype=np.int64)
        Ks_np[0] = j
        Ks_np[1 : 1 + len(negs)] = negs

        # -1 -> 0 padding, 0..n-1 -> 1..n
        Ks = torch.tensor(Ks_np + 1, dtype=torch.long)

        return I, Ks


def recon(table, pair_mat):
    "Reconstruction accuracy"
    count = 0
    table = torch.tensor(table[1:])
    n = pair_mat.shape[0]
    for i in range(1, n):  # 0 padding, 1 root, we leave those two
        x = table[i].repeat(len(table)).reshape([len(table), len(table[i])])  # N, D
        mask = torch.tensor([0.0] * len(table))
        mask[i] = 1
        mask = mask * -10000.0
        dists = lorentz_scalar_product(x, table) + mask
        dists = (
            dists.cpu().numpy()
        )  # arccosh is monotonically increasing, so no need of that here
        # and no -dist also, as acosh in m i, -acosh(-l(x,y)) is nothing but l(x,y)
        # print(dists)
        predicted_parent = np.argmax(dists)
        actual_parent = np.argmax(pair_mat[:, i])
        # print(predicted_parent, actual_parent, i, end="\n\n")
        count += actual_parent == predicted_parent
    count = count / (pair_mat.shape[0] - 1) * 100
    return count