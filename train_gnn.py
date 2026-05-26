#!/usr/bin/env python3
"""
Node classification on LastFM-Asia: SGC baseline, 2-layer GCN, 2-layer GraphSAGE.
Pure NumPy / SciPy / scikit-learn -- no PyTorch or PyG required.

Usage:
    python train_gnn.py
"""

import json, time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, eye
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# --- Config ------------------------------------------------------------------
DATA     = "lasftm_asia/"
SEED     = 42
SVD_DIM  = 128    # feature dims after TF-IDF + SVD
HIDDEN   = 256    # hidden layer width
EPOCHS   = 500
PATIENCE = 50
LR       = 1e-3   # Adam learning rate
WD       = 5e-4   # L2 weight decay
DROPOUT  = 0.5    # dropout between conv layers

rng = np.random.default_rng(SEED)

# --- 1. Load data ------------------------------------------------------------
print("--- 1. Loading data")
edges   = pd.read_csv(DATA + "lastfm_asia_edges.csv")
targets = pd.read_csv(DATA + "lastfm_asia_target.csv").sort_values("id").reset_index(drop=True)
with open(DATA + "lastfm_asia_features.json") as f:
    raw_feat = json.load(f)

N      = len(targets)
labels = targets["target"].values.astype(np.int64)
C      = int(labels.max()) + 1
print(f"    {N:,} nodes  {len(edges):,} edges  {C} classes")

# --- 2. Features: TF-IDF + SVD -----------------------------------------------
print(f"--- 2. Building features  (bag-of-artists -> TF-IDF -> SVD-{SVD_DIM})")

all_artists = sorted({int(a) for v in raw_feat.values() for a in v})
a2i         = {a: i for i, a in enumerate(all_artists)}
D           = len(a2i)

rows, cols = [], []
for node_str, artist_ids in raw_feat.items():
    n = int(node_str)
    for a in artist_ids:
        rows.append(n)
        cols.append(a2i[int(a)])

X_bin = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(N, D))

# IDF: down-weights globally popular artists (Zipf-distributed per EDA)
df      = np.asarray(X_bin.sum(axis=0)).flatten()
idf     = np.log((N + 1.0) / (df + 1.0)) + 1.0
X_tfidf = normalize(X_bin.multiply(idf), norm="l2")

svd = TruncatedSVD(n_components=SVD_DIM, random_state=SEED)
X   = svd.fit_transform(X_tfidf).astype(np.float32)   # (N, SVD_DIM)
print(f"    Explained variance: {svd.explained_variance_ratio_.sum():.3f}")

# --- 3. Adjacency matrices ---------------------------------------------------
print("--- 3. Building adjacency")
src = np.concatenate([edges["node_1"].values, edges["node_2"].values])
dst = np.concatenate([edges["node_2"].values, edges["node_1"].values])
A   = csr_matrix((np.ones(len(src), np.float32), (src, dst)), shape=(N, N))

# GCN: symmetric normalisation  A_hat = D^{-1/2}(A+I)D^{-1/2}
A_gcn  = (A + eye(N, format="csr")).astype(np.float32)
deg    = np.asarray(A_gcn.sum(1)).flatten()
Drt    = diags(1.0 / np.sqrt(deg), dtype=np.float32)
A_hat  = (Drt @ A_gcn @ Drt).astype(np.float32)

# GraphSAGE: row-normalised A (mean neighbour aggregation, no self-loop)
deg2   = np.asarray(A.sum(1)).flatten(); deg2[deg2 == 0] = 1.0
A_sage = (diags(1.0 / deg2, dtype=np.float32) @ A).astype(np.float32)

# Precompute fixed first-hop aggregations (X never changes)
AX_gcn       = np.asarray(A_hat  @ X, dtype=np.float32)   # (N, SVD_DIM)
AX_sage      = np.asarray(A_sage @ X, dtype=np.float32)   # (N, SVD_DIM)
# GraphSAGE layer-1 input: [self | mean-neighbours] -- also fixed
SAGE_AGG1    = np.hstack([X, AX_sage]).astype(np.float32) # (N, 2*SVD_DIM)

# --- 4. Stratified split  70 / 15 / 15 --------------------------------------
print("--- 4. Splitting  (stratified 70/15/15)")
idx              = np.arange(N)
tr_idx, tmp_idx  = train_test_split(idx, test_size=0.30, stratify=labels, random_state=SEED)
val_idx, te_idx  = train_test_split(tmp_idx, test_size=0.50,
                                     stratify=labels[tmp_idx], random_state=SEED)

tr  = np.zeros(N, bool); tr[tr_idx]   = True
val = np.zeros(N, bool); val[val_idx] = True
te  = np.zeros(N, bool); te[te_idx]   = True
print(f"    Train {tr.sum()} | Val {val.sum()} | Test {te.sum()}")

# Class weights for SGC baseline only (convex solver handles imbalance fine)
cw_balanced = compute_class_weight("balanced", classes=np.arange(C), y=labels[tr])
# GNN training: standard CE -- balanced weights cancel gradients near uniform init

# --- 5. Helpers --------------------------------------------------------------
def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=1, keepdims=True)

def ce_loss(probs, mask):
    """Standard (unweighted) cross-entropy on masked nodes."""
    p = np.clip(probs[mask, labels[mask]], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))

def dropout_fwd(H, p, training):
    """Inverted dropout. Returns (H_out, mask) -- mask needed for backprop."""
    if not training or p == 0.0:
        return H, None
    mask = (rng.random(H.shape) >= p).astype(np.float32)
    return H * mask / (1.0 - p), mask

def dropout_bwd(dH_out, mask, p):
    """Backward through inverted dropout. Pass mask=None when not training."""
    if mask is None:
        return dH_out
    return dH_out * mask / (1.0 - p)

def macro_f1(probs, mask):
    return f1_score(labels[mask], probs[mask].argmax(1), average="macro", zero_division=0)

# --- 6. Adam optimizer -------------------------------------------------------
class Adam:
    """Per-parameter Adam with weight decay (AdamW-style)."""
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, wd=5e-4):
        self.lr, self.beta1, self.beta2, self.eps, self.wd = lr, beta1, beta2, eps, wd
        self.t = 0
        self._m = {}
        self._v = {}

    def step(self, named_grads):
        """named_grads: dict of {name: (param_array, grad_array)}"""
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        for name, (param, grad) in named_grads.items():
            if name not in self._m:
                self._m[name] = np.zeros_like(param)
                self._v[name] = np.zeros_like(param)
            self._m[name] = self.beta1 * self._m[name] + (1.0 - self.beta1) * grad
            self._v[name] = self.beta2 * self._v[name] + (1.0 - self.beta2) * grad * grad
            m_hat = self._m[name] / bc1
            v_hat = self._v[name] / bc2
            # AdamW: weight decay applied directly to params, not to grad
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps) + self.lr * self.wd * param

# --- 7. GCN ------------------------------------------------------------------
class GCN:
    """
    2-layer GCN.
    Forward:  Z1 = A_hat @ X @ W1  (AX precomputed)
              H1 = relu(Z1),  dropout applied between layers
              Z2 = A_hat @ dropout(H1) @ W2
    """
    def __init__(self):
        self.W1  = (rng.standard_normal((SVD_DIM, HIDDEN)) * np.sqrt(2.0 / SVD_DIM)).astype(np.float32)
        self.W2  = (rng.standard_normal((HIDDEN,  C))      * np.sqrt(2.0 / HIDDEN)).astype(np.float32)
        self.opt = Adam(lr=LR, wd=WD)

    def forward(self, training=False):
        self._Z1      = AX_gcn @ self.W1                           # (N, HIDDEN)
        H1_raw        = np.maximum(0.0, self._Z1)
        H1, self._dm  = dropout_fwd(H1_raw, DROPOUT, training)    # dropout between layers
        self._H1      = H1
        AH1           = np.asarray(A_hat @ H1, dtype=np.float32)
        self._AH1     = AH1
        return softmax(AH1 @ self.W2)                              # (N, C)

    def backward(self, probs):
        n   = int(tr.sum())
        # Standard CE gradient (no class weights -- see code comment above)
        dZ2 = np.zeros((N, C), np.float32)
        d   = probs[tr].copy()
        d[np.arange(n), labels[tr]] -= 1.0
        d  /= n
        dZ2[tr] = d

        # Layer 2
        dW2  = self._AH1.T @ dZ2                                   # (HIDDEN, C)
        dAH1 = dZ2 @ self.W2.T                                     # (N, HIDDEN)

        # Back through A_hat (symmetric: A_hat.T = A_hat)
        dH1  = np.asarray(A_hat @ dAH1, dtype=np.float32)

        # Back through dropout
        dH1_raw = dropout_bwd(dH1, self._dm, DROPOUT)

        # Back through ReLU and layer 1
        dZ1 = dH1_raw * (self._Z1 > 0).astype(np.float32)
        dW1 = AX_gcn.T @ dZ1                                       # (SVD_DIM, HIDDEN)

        self.opt.step({"W1": (self.W1, dW1), "W2": (self.W2, dW2)})

# --- 8. GraphSAGE ------------------------------------------------------------
class GraphSAGE:
    """
    2-layer GraphSAGE (mean aggregator, concat self + neighbours).
    Layer 1 input  [X | A_sage@X]     (SAGE_AGG1, precomputed)
    Layer 2 input  [H1 | A_sage@H1]   (computed each step since H1 changes)
    """
    def __init__(self):
        self.W1  = (rng.standard_normal((2 * SVD_DIM, HIDDEN)) * np.sqrt(2.0 / (2 * SVD_DIM))).astype(np.float32)
        self.W2  = (rng.standard_normal((2 * HIDDEN,  C))      * np.sqrt(2.0 / (2 * HIDDEN))).astype(np.float32)
        self.opt = Adam(lr=LR, wd=WD)

    def forward(self, training=False):
        # Layer 1: concat(X, mean-neigh) -> linear -> ReLU
        self._Z1    = SAGE_AGG1 @ self.W1                          # (N, HIDDEN)
        H1_raw      = np.maximum(0.0, self._Z1)
        H1, self._dm = dropout_fwd(H1_raw, DROPOUT, training)
        self._H1    = H1

        # Layer 2: concat(H1, mean-neigh-H1) -> linear
        AH1         = np.asarray(A_sage @ H1, dtype=np.float32)
        agg2        = np.hstack([H1, AH1])                         # (N, 2*HIDDEN)
        self._agg2  = agg2
        self._AH1   = AH1
        return softmax(agg2 @ self.W2)                             # (N, C)

    def backward(self, probs):
        n   = int(tr.sum())
        dZ2 = np.zeros((N, C), np.float32)
        d   = probs[tr].copy()
        d[np.arange(n), labels[tr]] -= 1.0
        d  /= n
        dZ2[tr] = d

        # Layer 2
        dW2   = self._agg2.T @ dZ2                                 # (2*HIDDEN, C)
        dagg2 = dZ2 @ self.W2.T                                    # (N, 2*HIDDEN)

        # Split gradient: self path + neighbour path
        # Note: A_sage is NOT symmetric, so must use A_sage.T here
        dH1 = (dagg2[:, :HIDDEN]
               + np.asarray(A_sage.T @ dagg2[:, HIDDEN:], dtype=np.float32))

        # Back through dropout and ReLU
        dH1_raw = dropout_bwd(dH1, self._dm, DROPOUT)
        dZ1     = dH1_raw * (self._Z1 > 0).astype(np.float32)
        dW1     = SAGE_AGG1.T @ dZ1                                # (2*SVD_DIM, HIDDEN)

        self.opt.step({"W1": (self.W1, dW1), "W2": (self.W2, dW2)})

# --- 9. Training loop --------------------------------------------------------
def train_model(model, name):
    best_f1          = -1.0
    best_W1, best_W2 = model.W1.copy(), model.W2.copy()
    wait             = 0
    history          = {"loss": [], "val_f1": []}
    t0               = time.time()

    print(f"\n{'='*60}")
    print(f"  Training {name}")
    print(f"{'='*60}")

    for epoch in range(1, EPOCHS + 1):
        probs = model.forward(training=True)
        loss  = ce_loss(probs, tr)
        model.backward(probs)

        probs_eval = model.forward(training=False)
        vf1        = macro_f1(probs_eval, val)

        history["loss"].append(loss)
        history["val_f1"].append(vf1)

        if vf1 > best_f1:
            best_f1          = vf1
            best_W1, best_W2 = model.W1.copy(), model.W2.copy()
            wait             = 0
            marker           = " *"
        else:
            wait  += 1
            marker = ""

        if epoch % 25 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}  loss={loss:.4f}  val-F1={vf1:.4f}  best={best_f1:.4f}  wait={wait}{marker}")

        if wait >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    model.W1[:] = best_W1
    model.W2[:] = best_W2
    print(f"  Finished in {time.time()-t0:.1f}s  |  Best val macro-F1: {best_f1:.4f}")
    return history

# --- 10. Evaluation ----------------------------------------------------------
def evaluate(probs, name):
    pred = probs[te].argmax(1)
    true = labels[te]
    mf1  = f1_score(true, pred, average="macro",    zero_division=0)
    wf1  = f1_score(true, pred, average="weighted", zero_division=0)
    acc  = float((pred == true).mean())

    print(f"\n{'='*60}")
    print(f"  {name}  --  Test set")
    print(f"{'='*60}")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  Macro-F1      : {mf1:.4f}   <- primary metric (imbalanced classes)")
    print(f"  Weighted-F1   : {wf1:.4f}")
    print()
    print(classification_report(true, pred, zero_division=0))
    return pred, mf1

# --- 11. SGC baseline --------------------------------------------------------
def run_sgc(k=2):
    """k-hop Laplacian smoothing of features -> logistic regression (SGC)."""
    Xs = X.copy()
    for _ in range(k):
        Xs = np.asarray(A_hat @ Xs, dtype=np.float32)

    # LogReg with balanced weights works fine here (convex solver)
    cw_dict = {c: cw_balanced[c] for c in range(C)}
    clf = LogisticRegression(
        max_iter=1000, C=1.0, class_weight=cw_dict,
        solver="saga", random_state=SEED
    )
    clf.fit(Xs[tr], labels[tr])
    pred_te  = clf.predict(Xs[te])
    pred_val = clf.predict(Xs[val])

    mf1_val  = f1_score(labels[val], pred_val, average="macro", zero_division=0)
    mf1_test = f1_score(labels[te],  pred_te,  average="macro", zero_division=0)
    acc      = float((pred_te == labels[te]).mean())

    print(f"\n{'='*60}")
    print(f"  SGC-{k}  (logistic regression on {k}-hop smoothed features)")
    print(f"{'='*60}")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  Val  macro-F1 : {mf1_val:.4f}")
    print(f"  Test macro-F1 : {mf1_test:.4f}   <- primary metric")
    print()
    print(classification_report(labels[te], pred_te, zero_division=0))
    return pred_te, mf1_test

# --- 12. Plots ---------------------------------------------------------------
def plot_training(histories):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, h in histories.items():
        axes[0].plot(h["loss"],   label=name)
        axes[1].plot(h["val_f1"], label=name)
    axes[0].set(title="Training loss (CE)", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Validation macro-F1", xlabel="Epoch", ylabel="Macro-F1")
    for ax in axes:
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=110)
    print("  Saved: training_curves.png")

def plot_confusion(true, pred, title, fname):
    cm = confusion_matrix(true, pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                linewidths=0.3, cbar_kws={"label": "Count"})
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    plt.tight_layout()
    plt.savefig(fname, dpi=110)
    print(f"  Saved: {fname}")

def plot_comparison(results):
    names  = list(results.keys())
    scores = list(results.values())
    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = ["steelblue", "teal", "darkorange"][:len(names)]
    bars    = ax.bar(names, scores, color=colors, width=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
    ax.set(title="Test Macro-F1 comparison", ylabel="Macro-F1", ylim=(0, 1))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=110)
    print("  Saved: comparison.png")

# --- 13. Run -----------------------------------------------------------------
if __name__ == "__main__":
    results   = {}
    histories = {}
    preds     = {}

    # SGC baseline (fast, no gradient descent)
    preds["SGC-2"], results["SGC-2"] = run_sgc(k=2)

    # GCN
    gcn = GCN()
    histories["GCN"] = train_model(gcn, "GCN")
    probs_gcn = gcn.forward(training=False)
    preds["GCN"], results["GCN"] = evaluate(probs_gcn, "GCN")

    # GraphSAGE
    sage = GraphSAGE()
    histories["GraphSAGE"] = train_model(sage, "GraphSAGE")
    probs_sage = sage.forward(training=False)
    preds["GraphSAGE"], results["GraphSAGE"] = evaluate(probs_sage, "GraphSAGE")

    # Summary
    print(f"\n{'='*60}")
    print("  FINAL COMPARISON  (test set, macro-F1)")
    print(f"{'='*60}")
    for name, score in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:<15}  {score:.4f}")

    # Plots
    plot_training(histories)
    plot_confusion(labels[te], preds["GCN"],       "GCN - Test Confusion Matrix",       "confusion_gcn.png")
    plot_confusion(labels[te], preds["GraphSAGE"], "GraphSAGE - Test Confusion Matrix", "confusion_sage.png")
    plot_comparison(results)
