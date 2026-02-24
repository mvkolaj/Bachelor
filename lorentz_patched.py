"""
lorentz.py

Train Lorentz-model hyperbolic embeddings from a (weighted) adjacency / similarity matrix.

Typical usage:
    python lorentz.py path/to/graph.npy
or
    python lorentz.py binary_tree

You can also plot a checkpoint in 2D Poincaré:
    python lorentz.py path/to/graph.npy -plot -ckpt ckpt/some.ckpt -plot_graph

This file is based on the Lorentz embeddings implementation inspired by:
Nickel & Kiela (2018) "Learning Continuous Hierarchies in the Lorentz Model..."
and the public codebase (theSage21/lorentz-embeddings and forks). citeturn1view0turn0academia5
"""
from __future__ import annotations

import os
import sys
import random
from collections import Counter
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tensorboardX import SummaryWriter
except Exception:
    SummaryWriter = None  # tensorboard logging optional

import datasets as dataset_lib


# ------------------------- Geometry helpers

def arcosh(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x + torch.sqrt(x ** 2 - 1))


def lorentz_scalar_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Lorentzian inner product <x,y>_L for batch vectors (B,D)."""
    m = x * y
    return m[:, 1:].sum(dim=1) - m[:, 0]


def tangent_norm(v: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(lorentz_scalar_product(v, v), min=0.0))


def exp_map(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Exponential map on the hyperboloid."""
    tn = tangent_norm(v).unsqueeze(dim=1)  # (B,1)
    tn_expand = tn.repeat(1, x.size(-1))
    # avoid division-by-zero
    out = torch.cosh(tn) * x + torch.sinh(tn) * torch.where(tn_expand > 0, v / tn, v)
    out = torch.where(tn_expand > 0, out, x)
    return out


def set_dim0(x: torch.Tensor) -> torch.Tensor:
    """Project points to the hyperboloid: -x0^2 + ||x1:||^2 = -1 with x0>0."""
    x = torch.renorm(x, p=2, dim=0, maxnorm=1e2)
    dim0 = torch.sqrt(1 + (x[:, 1:] ** 2).sum(dim=1))
    x[:, 0] = dim0
    return x


# ------------------------- Riemannian SGD

class RSGD(optim.Optimizer):
    def __init__(self, params, learning_rate: float = 0.01):
        defaults = {"learning_rate": float(learning_rate)}
        super().__init__(params, defaults=defaults)

    @property
    def learning_rate(self) -> float:
        return float(self.param_groups[0]["learning_rate"])

    @learning_rate.setter
    def learning_rate(self, lr: float) -> None:
        for g in self.param_groups:
            g["learning_rate"] = float(lr)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = float(group["learning_rate"])
            for p in group["params"]:
                if p.grad is None:
                    continue

                B, D = p.size()
                gl = torch.eye(D, device=p.device, dtype=p.dtype)
                gl[0, 0] = -1.0

                grad_norm = torch.norm(p.grad.data)
                grad_norm = torch.where(grad_norm > 1, grad_norm, torch.tensor(1.0, device=p.device))

                h = (p.grad.data / grad_norm) @ gl
                proj = h - (lorentz_scalar_product(p, h) / lorentz_scalar_product(p, p)).unsqueeze(1) * p

                update = exp_map(p, -lr * proj)
                is_nan_inf = torch.isnan(update) | torch.isinf(update)
                update = torch.where(is_nan_inf, p, update)

                # keep padding vector fixed
                update[0, :] = p[0, :]
                update = set_dim0(update)
                p.data.copy_(update)


# ------------------------- Model

class Lorentz(nn.Module):
    """Embed n_items in a dim-dimensional Lorentz space (dim = poincare_dim + 1)."""

    def __init__(self, n_items: int, dim: int, init_range: float = 1e-3):
        super().__init__()
        self.n_items = int(n_items)
        self.dim = int(dim)
        self.table = nn.Embedding(self.n_items + 1, self.dim, padding_idx=0)
        nn.init.uniform_(self.table.weight, -init_range, init_range)
        with torch.no_grad():
            self.table.weight[0] = 5.0
            set_dim0(self.table.weight)

    def forward(self, I: torch.Tensor, Ks: torch.Tensor) -> torch.Tensor:
        """
        Ranking loss as in the reference implementation:
          - I : (B,) indices of anchors
          - Ks: (B,N) indices, where Ks[:,0] is a positive and others are negatives/padding
        """
        n_ks = Ks.size(1)
        ui = torch.stack([self.table(I)] * n_ks, dim=1)
        uks = self.table(Ks)

        B, N, D = ui.size()
        ui = ui.reshape(B * N, D)
        uks = uks.reshape(B * N, D)

        d = -lorentz_scalar_product(ui, uks)
        d = torch.where(d <= 1, torch.ones_like(d) + 1e-6, d)
        d = -arcosh(d)

        d = d.reshape(B, N)
        loss = -(d[:, 0] - torch.log(torch.exp(d).sum(dim=1) + 1e-6))
        return loss

    def lorentz_to_poincare(self) -> np.ndarray:
        table = self.table.weight.data.detach().cpu().numpy()
        return table[:, 1:] / (table[:, :1] + 1)

    def get_lorentz_table(self) -> np.ndarray:
        return self.table.weight.data.detach().cpu().numpy()

    def _test_table(self) -> float:
        x = self.table.weight.data
        check = lorentz_scalar_product(x, x) + 1.0
        return float(check.detach().cpu().numpy().sum())

    @torch.no_grad()
    def init_from_embeddings(self, emb: np.ndarray, *, assume: str = "euclidean") -> None:
        """
        Initialize the embedding table from an existing array.

        emb shapes supported:
          - (n_items, poincare_dim) or (n_items+1, poincare_dim) -> placed in spatial dims
          - (n_items, poincare_dim+1) or (n_items+1, poincare_dim+1) -> treated as Lorentz vectors

        assume:
          - "euclidean": spatial coords copied, then projected to hyperboloid
          - "lorentz": copied and projected (dim0 recomputed)
        """
        emb = np.asarray(emb)
        if emb.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape={emb.shape}")

        n_expected = self.n_items
        d = self.dim

        # strip possible padding row
        if emb.shape[0] == n_expected + 1:
            emb_use = emb[1:]
        elif emb.shape[0] == n_expected:
            emb_use = emb
        else:
            raise ValueError(f"emb rows must be {n_expected} or {n_expected+1}, got {emb.shape[0]}")

        w = self.table.weight.data
        w[1:, :] = 0.0

        if emb_use.shape[1] == d:
            w[1:, :] = torch.tensor(emb_use, device=w.device, dtype=w.dtype)
        elif emb_use.shape[1] == d - 1:
            w[1:, 1:] = torch.tensor(emb_use, device=w.device, dtype=w.dtype)
        else:
            raise ValueError(f"emb dim must be {d} or {d-1}, got {emb_use.shape[1]}")

        # (re)project
        set_dim0(w)


# ------------------------- Dataset sampling

class Graph(Dataset):
    """
    Produces training triples (I, Ks) from a (weighted) adjacency matrix.

    We treat any positive entry as an edge (parent/child). For each node i:
      - pick one positive neighbor j (prefer parent if any; else child)
      - sample negatives among nodes where edge weight < weight(i,j) (for binary graphs this is 0s)
    """

    def __init__(self, pairwise_matrix: sp.csr_matrix, sample_size: int = 5, batch_size: int = 32):
        if not sp.issparse(pairwise_matrix):
            pairwise_matrix = sp.csr_matrix(pairwise_matrix)
        self.csr = pairwise_matrix.tocsr()
        self.csc = pairwise_matrix.tocsc()
        self.n_items = self.csr.shape[0]
        self.sample_size = int(sample_size)
        self.batch_size = int(batch_size)
        self._arange = np.arange(self.n_items, dtype=np.int64)
        self._cnter = 0

    def __len__(self) -> int:
        return self.n_items

    def _pick_positive(self, i: int, arange: np.ndarray) -> Tuple[int, float, bool]:
        """Return (j, min_weight, used_child_edge)."""
        parents = self.csc[:, i].indices
        children = self.csr[i].indices

        if len(parents) > 0:
            # choose a random parent
            j = int(random.choice(parents))
            w = float(self.csc[j, i])
            return j, w, False
        if len(children) > 0:
            j = int(random.choice(children))
            w = float(self.csr[i, j])
            return j, w, True
        raise ValueError(f"Node {i} has no parents and no children (disconnected)")

    def _sample_negatives(self, i: int, min_w: float, use_child: bool) -> np.ndarray:
        # Fast negative sampling: draw candidates uniformly until we have enough that satisfy weight < min_w.
        needed = max(self.sample_size - 1, 0)
        if needed == 0:
            return np.array([], dtype=np.int64)

        negs = []
        # cap attempts to avoid infinite loops on dense graphs
        max_attempts = needed * 50 + 100

        for _ in range(max_attempts):
            cand = int(random.randrange(self.n_items))
            if cand == i:
                continue
            if use_child:
                w = float(self.csr[i, cand])
            else:
                w = float(self.csc[cand, i])
            if w < min_w:
                negs.append(cand)
                if len(negs) >= needed:
                    break

        if len(negs) < needed:
            # fallback: just sample random distinct nodes (may include some false negatives if dense)
            remaining = needed - len(negs)
            pool = self._arange[self._arange != i]
            extra = np.random.choice(pool, size=remaining, replace=(remaining > len(pool)))
            negs.extend([int(x) for x in extra])

        return np.array(negs, dtype=np.int64)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cnter = (self._cnter + 1) % self.batch_size
        # (i+1) because 0 is padding index
        I = torch.tensor(i + 1, dtype=torch.long)

        # small shuffling trick like the original code
        arange = np.random.permutation(self._arange) if self._cnter == 0 else self._arange

        j, min_w, used_child = self._pick_positive(i, arange)
        negs = self._sample_negatives(i, min_w, used_child)

        # Ks must be length sample_size, with first being the positive j
        ks = np.concatenate([[j + 1], negs + 1, np.zeros(self.sample_size, dtype=np.int64)])[: self.sample_size]
        return I, torch.tensor(ks, dtype=torch.long)


def recon(lorentz_table: np.ndarray, pair_mat: sp.csr_matrix) -> float:
    """Reconstruction accuracy: does each node pick the correct parent as nearest neighbor?"""
    if sp.issparse(pair_mat):
        pair_dense = pair_mat.toarray()
    else:
        pair_dense = np.asarray(pair_mat)
    table = torch.tensor(lorentz_table[1:], dtype=torch.float32)  # skip padding row
    n = pair_dense.shape[0]

    correct = 0
    for i in range(1, n):
        x = table[i].repeat(len(table)).reshape(len(table), -1)
        mask = torch.zeros(len(table))
        mask[i] = -10000.0
        dists = lorentz_scalar_product(x, table) + mask
        predicted_parent = int(torch.argmax(dists).item())
        actual_parent = int(np.argmax(pair_dense[:, i]))
        correct += (actual_parent == predicted_parent)
    return correct / max(n - 1, 1) * 100.0


# ------------------------- CLI

def _collect_edges(pairwise: sp.csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
    """Return undirected edges (u,v) as arrays in 1-indexed node ids."""
    csr = pairwise.tocsr()
    rows, cols = csr.nonzero()
    # undirected unique edges for plotting
    edges = {tuple(sorted((int(r) + 1, int(c) + 1))) for r, c in zip(rows, cols)}
    e = np.array(sorted(edges), dtype=np.int64)
    if e.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    return e[:, 0], e[:, 1]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Builtin name or path to adjacency/similarity matrix")
    parser.add_argument("-sample_size", default=5, type=int, help="How many samples in Ks (1 pos + negatives)")
    parser.add_argument("-batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("-burn_c", default=10, type=int, help="Divide lr by this during burn-in")
    parser.add_argument("-burn_epochs", default=1, type=int, help="Number of burn-in epochs")
    parser.add_argument("-epochs", default=10, type=int, help="Total epochs")

    parser.add_argument("-poincare_dim", default=2, type=int, help="Poincaré dim (Lorentz dim is +1)")
    parser.add_argument("-n_items", default=None, type=int, help="Optional truncate number of nodes (top-left)")
    parser.add_argument("-learning_rate", default=0.1, type=float, help="RSGD learning rate")
    parser.add_argument("-shuffle", default=True, type=bool, help="Shuffle within DataLoader")

    parser.add_argument("-device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("-log", action="store_true", help="Log to TensorBoard (tensorboardX)")
    parser.add_argument("-logdir", default="runs", type=str)
    parser.add_argument("-log_step", default=1, type=int)

    parser.add_argument("-save_step", default=100, type=int, help="Checkpoint frequency (epochs)")
    parser.add_argument("-savedir", default="ckpt", type=str)

    parser.add_argument("-plot", action="store_true", help="Plot embeddings instead of training")
    parser.add_argument("-plot_graph", action="store_true", help="Plot edges on top of embeddings")
    parser.add_argument("-overwrite_plots", action="store_true", help="Overwrite existing plot files")
    parser.add_argument("-plot_size", default=7, type=int, help="Figure size")
    parser.add_argument("-ckpt", default=None, type=str, help="Checkpoint path or directory (for plotting)")

    parser.add_argument("-loader_workers", default=0, type=int, help="DataLoader workers")

    # NEW: initialize from an existing embedding array
    parser.add_argument("-init_embeddings", default=None, type=str, help="Path to .npy with initial embeddings")

    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.savedir, exist_ok=True)

    pairwise = dataset_lib.get_dataset(args.dataset)

    # optional truncation
    if args.n_items is not None:
        n = int(args.n_items)
        pairwise = pairwise[:n, :n]
    args.n_items = pairwise.shape[0]

    net = Lorentz(args.n_items, args.poincare_dim + 1).to(args.device)

    if args.init_embeddings:
        emb = np.load(args.init_embeddings)
        net.init_from_embeddings(emb)

    # ---------------- plot mode
    if args.plot:
        if args.poincare_dim != 2:
            print("Plotting is only supported for -poincare_dim 2.")
            sys.exit(1)
        if args.ckpt is None:
            print("Please provide -ckpt when using -plot.")
            sys.exit(1)

        paths = []
        if os.path.isdir(args.ckpt):
            for f in os.listdir(args.ckpt):
                if f.endswith(".ckpt") or f.endswith(".pt"):
                    paths.append(os.path.join(args.ckpt, f))
        else:
            paths = [args.ckpt]

        paths = sorted(paths)
        u, v = _collect_edges(pairwise)

        for path in tqdm(paths, desc="Plotting"):
            save_path = f"{path}.svg"
            if os.path.exists(save_path) and not args.overwrite_plots:
                continue

            state = torch.load(path, map_location=args.device)
            net.load_state_dict(state)

            table = net.lorentz_to_poincare()
            plt.figure(figsize=(args.plot_size, args.plot_size))

            if args.plot_graph and len(u) > 0:
                for a, b in zip(u, v):
                    plt.plot(table[[a, b], 0], table[[a, b], 1], color="black", marker="o", alpha=0.35)
            else:
                plt.scatter(table[1:, 0], table[1:, 1])

            plt.gca().set_xlim(-1, 1)
            plt.gca().set_ylim(-1, 1)
            plt.gca().add_artist(plt.Circle((0, 0), 1, fill=False, edgecolor="black"))
            plt.title(os.path.basename(path))
            plt.savefig(save_path)
            plt.close()

        return

    # ---------------- train mode

    graph_dataset = Graph(pairwise, sample_size=args.sample_size, batch_size=args.batch_size)
    dataloader = DataLoader(
        graph_dataset,
        shuffle=args.shuffle,
        batch_size=args.batch_size,
        num_workers=args.loader_workers,
        drop_last=False,
    )

    rsgd = RSGD(net.parameters(), learning_rate=args.learning_rate)
    run_name = f"{os.path.basename(args.dataset)} {datetime.utcnow().isoformat(timespec='seconds')}"

    writer = None
    if args.log:
        if SummaryWriter is None:
            print("tensorboardX not installed; skipping logging.")
        else:
            writer = SummaryWriter(os.path.join(args.logdir, run_name))

    with tqdm(total=args.epochs, ncols=90) as epoch_bar:
        for epoch in range(args.epochs):
            rsgd.learning_rate = args.learning_rate / args.burn_c if epoch < args.burn_epochs else args.learning_rate

            # one epoch
            for I, Ks in dataloader:
                I = I.to(args.device)
                Ks = Ks.to(args.device)

                rsgd.zero_grad()
                loss = net(I, Ks).mean()
                loss.backward()
                rsgd.step()

            if writer is not None and (epoch % args.log_step == 0):
                writer.add_scalar("loss", float(loss), epoch)
                writer.add_scalar("recon_acc", recon(net.get_lorentz_table(), pairwise), epoch)
                writer.add_scalar("table_test", net._test_table(), epoch)

            if epoch % args.save_step == 0:
                ckpt_path = os.path.join(args.savedir, f"{epoch} {run_name}.ckpt")
                torch.save(net.state_dict(), ckpt_path)

            epoch_bar.set_description(f"epoch={epoch} loss={float(loss):.4f} lr={rsgd.learning_rate:g}")
            epoch_bar.update(1)

    final_path = args.dataset.replace(".", "_") + "_lorentz_embeddings.pt"
    torch.save(net.state_dict(), final_path)
    print(f"Saved final model to: {final_path}")


if __name__ == "__main__":
    main()
