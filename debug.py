import numpy as np

adj = np.load("my_graph.npy")

print("Shape:", adj.shape)
print("Diagonal sum:", adj.diagonal().sum())
print("Is symmetric:", np.allclose(adj, adj.T))
print("Min value:", adj.min())
print("Max value:", adj.max())
print("Non-zero entries:", np.count_nonzero(adj))
print("Total entries:", adj.shape[0] ** 2)
print("Density:", np.count_nonzero(adj) / adj.shape[0] ** 2)
print("Nodes with zero rows:", (adj.sum(axis=1) == 0).sum())

# Check a few row samples
for i in [0, 1, 2]:
    row = adj[i]
    pos = np.where(row > 0)[0]
    neg = np.where(row == 0)[0]
    print(f"Node {i}: {len(pos)} positives, {len(neg)} negatives, max weight: {row.max():.4f}")