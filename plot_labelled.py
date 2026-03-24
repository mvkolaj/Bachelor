import matplotlib.pyplot as plt
import torch
import pickle
import numpy as np
from lorentz import Lorentz

# ── config ────────────────────────────────────────────────────────────────────
CKPT        = "ckpt/final_my_graph.ckpt"
VOCAB       = "matrix_vocab.pkl"
OUT         = "poincare_labelled.png"
DIM         = 2        # poincare_dim used during training
FIGSIZE     = (40, 40)
FONTSIZE    = 6
# set to None to label all genres, or a number e.g. 50 to label only top N
# genres by degree (most connected), leaving the rest as unlabelled dots
TOP_N_LABELS = None
# ──────────────────────────────────────────────────────────────────────────────

# load vocab
genres = pickle.load(open(VOCAB, "rb"))
n_items = len(genres)
print(f"Loaded {n_items} genres")

# load model and get poincare coordinates
net = Lorentz(n_items, DIM + 1)
net.load_state_dict(torch.load(CKPT, map_location="cpu"))
net.eval()

# skip index 0 (padding), get 2D poincare coords
table = net.lorentz_to_poincare()
coords = table[1:]   # shape (n_items, 2)
x = coords[:, 0]
y = coords[:, 1]

# optionally load matrix to compute degrees for top-N labelling
if TOP_N_LABELS is not None:
    try:
        adj = np.load("my_graph.npy")
        degrees = (adj > 0).sum(axis=1)
        top_indices = set(np.argsort(degrees)[-TOP_N_LABELS:])
        print(f"Labelling top {TOP_N_LABELS} genres by degree")
    except FileNotFoundError:
        print("my_graph.npy not found, labelling all genres")
        top_indices = set(range(n_items))
else:
    top_indices = set(range(n_items))

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# unit circle boundary
circle = plt.Circle((0, 0), 1, fill=False, edgecolor="black", linewidth=1.5)
ax.add_artist(circle)

# scatter all dots
ax.scatter(x, y, s=20, color="tomato", zorder=3, linewidths=0)

# labels with white background box (matching reference style)
for idx, genre in enumerate(genres):
    if idx in top_indices:
        ax.annotate(
            genre,
            (x[idx], y[idx]),
            fontsize=FONTSIZE,
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
            zorder=4,
            bbox={"fc": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1},
        )

ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Poincaré Disk — Genre Embeddings", fontsize=16, pad=15)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Saved to {OUT}")
