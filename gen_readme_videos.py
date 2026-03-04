"""Generate 3 animation videos for README: UMAP, t-SNE, PaCMAP."""
import numpy as np
import time
from sklearn.datasets import fetch_openml

print("Loading Fashion-MNIST 70K...")
fm = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="liac-arff")
X = fm.data.astype(np.float32) / 255.0
y = fm.target.astype(np.int32)

np.random.seed(42)
viz_idx = np.random.choice(len(X), 15000, replace=False)
sub_labels = y[viz_idx]

from mlx_vis import UMAP, TSNE, PaCMAP
from mlx_vis.plot import animate_gpu

methods = [
    ("umap", "UMAP", lambda cb: UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42, n_epochs=500).fit_transform(X, epoch_callback=cb)),
    ("tsne", "t-SNE", lambda cb: TSNE(n_components=2, perplexity=30, n_iter=500, random_state=42).fit_transform(X, epoch_callback=cb)),
    ("pacmap", "PaCMAP", lambda cb: PaCMAP(n_components=2, n_neighbors=10, random_state=42).fit_transform(X, epoch_callback=cb)),
]

for fname, name, fn in methods:
    snaps, times = [], []
    t0 = time.time()
    def cb(epoch, Y_np, _t0=t0, _s=snaps, _ts=times):
        _s.append(Y_np.copy())
        _ts.append(time.time() - _t0)
    
    print(f"\nRunning {name}...")
    Y = fn(cb)
    elapsed = time.time() - t0
    print(f"  {name}: {elapsed:.1f}s, {len(snaps)} snapshots")
    
    sub_snaps = [s[viz_idx] for s in snaps]
    
    out = f"/Users/hanxiao/Documents/mlx-vis/{fname}-animation.mp4"
    t1 = time.time()
    nf = animate_gpu(sub_snaps, labels=sub_labels, timestamps=times,
        method_name=f"{fname}-mlx", dataset_name="Fashion-MNIST 70K",
        fps=120, theme="dark", save=out, bitrate=4000)
    print(f"  Render: {time.time()-t1:.1f}s, {nf} frames -> {out}")

print("\nDone!")
