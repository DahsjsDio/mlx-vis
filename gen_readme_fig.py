"""Generate 2x3 comparison figure: top row dark, bottom row light.
Columns: UMAP, t-SNE, PaCMAP. All 500 iterations on full Fashion-MNIST 70K."""
import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from mlx_vis import UMAP, TSNE, PaCMAP
from mlx_vis.plot import _resolve_colors, _get_square_lims

print("Loading Fashion-MNIST 70K...")
fm = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="liac-arff")
X = fm.data.astype(np.float32) / 255.0
y = fm.target.astype(np.int32)
print(f"Loaded {X.shape}")

methods = [
    ("UMAP", lambda: UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42, n_epochs=500).fit_transform(X)),
    ("t-SNE", lambda: TSNE(n_components=2, perplexity=30, n_iter=500, random_state=42).fit_transform(X)),
    ("PaCMAP", lambda: PaCMAP(n_components=2, n_neighbors=10, random_state=42).fit_transform(X)),
]

results = []
for name, fn in methods:
    print(f"Running {name} (500 iter)...")
    t0 = time.time()
    Y = fn()
    elapsed = time.time() - t0
    print(f"  {name} done in {elapsed:.1f}s")
    results.append((name, Y, elapsed))

# Build 2x3 figure
fig, axes = plt.subplots(2, 3, figsize=(30, 20))

for col, (name, Y, elapsed) in enumerate(results):
    for row, theme in enumerate(["dark", "light"]):
        ax = axes[row, col]
        bg = "black" if theme == "dark" else "white"
        fg = "white" if theme == "dark" else "black"
        
        ax.set_facecolor(bg)
        
        c = _resolve_colors(y, None, len(Y), theme)
        c = np.array(c, dtype=np.float64)
        c[:, 3] = 0.6
        
        xlim, ylim = _get_square_lims(Y)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.scatter(Y[:, 0], Y[:, 1], s=1.0, c=c, edgecolors="none")
        ax.set_title(f"{name}  70K x 784  {elapsed:.1f}s", color=fg,
                     fontsize=16, pad=10, fontfamily="monospace")

# Top row black bg, bottom row white bg
for col in range(3):
    axes[0, col].set_facecolor("black")
    axes[1, col].set_facecolor("white")

fig.patch.set_facecolor("black")
# Split background: draw a white rectangle on bottom half
from matplotlib.patches import Rectangle
fig.patches.append(Rectangle((0, 0), 1, 0.5, transform=fig.transFigure,
                              facecolor="white", zorder=-1))
fig.patches.append(Rectangle((0, 0.5), 1, 0.5, transform=fig.transFigure,
                              facecolor="black", zorder=-1))

plt.subplots_adjust(wspace=0.05, hspace=0.1)
out = "/Users/hanxiao/.openclaw/workspace/mlx-vis-comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved {out}")
