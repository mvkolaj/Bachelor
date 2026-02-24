"""
datasets.py

Utilities for loading graphs for Lorentz hyperbolic embeddings.

This file is compatible with the CLI in lorentz.py:
    python lorentz.py <dataset>

<dataset> can be:
  - a built-in name: "binary_tree", "quad_tree"
  - a path to a file:
      * .pkl  (pickle: numpy array or scipy sparse matrix)
      * .npy  (numpy dense matrix)
      * .npz  (scipy sparse saved via scipy.sparse.save_npz)
      * .csv/.tsv edge list with columns: source,target[,weight]
        (strings are factorized to indices; returns sparse matrix)
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp


# ------------------------- Built-ins (small sanity-check graphs)

def _make_binary_tree(depth: int = 10) -> sp.csr_matrix:
    n = sum(2 ** i for i in range(depth))
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        j = 2 * i + 1
        if j + 1 >= n:
            break
        mat[i, j] = 1.0
        mat[i, j + 1] = 1.0
    return sp.csr_matrix(mat)


def _make_quad_tree(depth: int = 4) -> sp.csr_matrix:
    n = sum(4 ** i for i in range(depth))
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        j = 4 * i + 1
        if j + 3 >= n:
            break
        mat[i, j : j + 4] = 1.0
    return sp.csr_matrix(mat)


BUILTINS: Dict[str, sp.csr_matrix] = {
    "binary_tree": _make_binary_tree(),
    "quad_tree": _make_quad_tree(),
}


# ------------------------- Loaders

def _as_csr(x) -> sp.csr_matrix:
    if sp.issparse(x):
        return x.tocsr().astype(np.float32)
    arr = np.asarray(x)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected a square 2D matrix, got shape={arr.shape}")
    return sp.csr_matrix(arr.astype(np.float32))


def load_pickle(path: str) -> sp.csr_matrix:
    obj = pickle.load(open(path, "rb"))
    return _as_csr(obj)


def load_npy(path: str) -> sp.csr_matrix:
    arr = np.load(path, allow_pickle=False)
    return _as_csr(arr)


def load_npz(path: str) -> sp.csr_matrix:
    mat = sp.load_npz(path)
    return _as_csr(mat)


@dataclass
class EdgeListLoadResult:
    matrix: sp.csr_matrix
    node_to_idx: Dict[str, int]


def load_edge_list(
    path: str,
    sep: Optional[str] = None,
    directed: bool = True,
    weight_col: str = "weight",
    source_col: str = "source",
    target_col: str = "target",
    binarize: bool = False,
) -> EdgeListLoadResult:
    """
    Load an edge list into a CSR adjacency matrix.

    The file must have at least columns [source_col, target_col].
    If weight_col is missing, weights default to 1.0.

    Notes:
      - If you want an undirected graph, set directed=False (we will symmetrize).
      - If binarize=True, all non-zero weights become 1.0.
    """
    import pandas as pd

    if sep is None:
        # heuristic: tab for .tsv, comma for .csv, else let pandas infer
        ext = os.path.splitext(path)[1].lower()
        sep = "\t" if ext in [".tsv", ".tab"] else ","

    df = pd.read_csv(path, sep=sep)

    if source_col not in df.columns or target_col not in df.columns:
        raise ValueError(
            f"Edge list must contain columns '{source_col}' and '{target_col}'. "
            f"Found: {list(df.columns)}"
        )

    if weight_col in df.columns:
        w = df[weight_col].astype(float).to_numpy()
    else:
        w = np.ones(len(df), dtype=float)

    src = df[source_col].astype(str).to_numpy()
    tgt = df[target_col].astype(str).to_numpy()

    # factorize nodes
    nodes = np.unique(np.concatenate([src, tgt]))
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    rows = np.array([node_to_idx[s] for s in src], dtype=np.int64)
    cols = np.array([node_to_idx[t] for t in tgt], dtype=np.int64)

    if binarize:
        w = np.where(np.asarray(w) > 0, 1.0, 0.0)

    n = len(nodes)
    mat = sp.coo_matrix((w, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()

    if not directed:
        mat = mat.maximum(mat.T)

    return EdgeListLoadResult(matrix=mat, node_to_idx=node_to_idx)


def get_dataset(name_or_path: str) -> sp.csr_matrix:
    """
    Main entry point used by lorentz.py.

    - If name_or_path matches a builtin name, returns that graph.
    - Otherwise, interprets it as a file path and loads it based on extension.
    """
    if name_or_path in BUILTINS:
        return BUILTINS[name_or_path]

    if not os.path.exists(name_or_path):
        raise FileNotFoundError(
            f"Dataset '{name_or_path}' not found. "
            f"Use a builtin ({', '.join(BUILTINS.keys())}) or provide a valid path."
        )

    ext = os.path.splitext(name_or_path)[1].lower()
    if ext == ".pkl":
        return load_pickle(name_or_path)
    if ext == ".npy":
        return load_npy(name_or_path)
    if ext == ".npz":
        return load_npz(name_or_path)
    if ext in [".csv", ".tsv", ".tab", ".txt"]:
        # default: edge list
        return load_edge_list(name_or_path, directed=True).matrix

    # fallback: try pickle
    return load_pickle(name_or_path)
