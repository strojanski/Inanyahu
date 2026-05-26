#!/usr/bin/env python3
"""
Node classification on LastFM-Asia.
Two feature variants x three models (SGC, GCN, GraphSAGE).

Feature variants
  freq_norm   : frequency-weighted + L2 normalise (no SVD)
  freq_svd128 : frequency-weighted + L2 + SVD-128  -- compact dense representation

For GNN (GCN / GraphSAGE) the high-dim variant (freq_norm) is compressed to
SVD-128 before training; SGC uses it at full dimensionality.

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
SVD_DIM  = 128
HIDDEN   = 256
EPOCHS   = 500
PATIENCE = 50
LR       = 1e-3
WD       = 5e-4
DROPOUT  = 0.5

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

# --- 2. Build three feature variants -----------------------------------------
print("--- 2. Building feature variants")

all_artists = sorted({int(a) for v in raw_feat.values() for a in v})
a2i         = {a: i for i, a in enumerate(all_artists)}
D           = len(a2i)

row_idx, col_idx = [], []
for node_str, artist_ids in raw_feat.items():
    n = int(node_str)
    for a in artist_ids:
        row_idx.append(n)
        col_idx.append(a2i[int(a)])

X_bin = csr_matrix(
    (np.ones(len(row_idx), np.float32), (row_idx, col_idx)), shape=(N, D)
)

# (a) inverse frequency-weighted + L2 normalise (no SVD)
df_vec   = np.asarray(X_bin.sum(axis=0)).flatten()
X_freq   = np.ones((N, D), dtype=np.float32) - normalize(X_bin.multiply(df_vec), norm="l2")   # sparse (N, D)

# (b) frequency-weighted + L2 + SVD-128
svd      = TruncatedSVD(n_components=SVD_DIM, random_state=SEED)
X_svd    = svd.fit_transform(X_freq).astype(np.float32)   # dense (N, 128)
print(f"    (a) freq_norm   : shape {X_freq.shape},  sparse")
print(f"    (b) freq_svd128 : shape {X_svd.shape},  dense  "
      f"(var explained: {svd.explained_variance_ratio_.sum():.3f})")

# --- 3. Adjacency matrices ---------------------------------------------------
print("--- 3. Building adjacency")
src = np.concatenate([edges["node_1"].values, edges["node_2"].values])
dst = np.concatenate([edges["node_2"].values, edges["node_1"].values])
A   = csr_matrix((np.ones(len(src), np.float32), (src, dst)), shape=(N, N))

A_gcn  = (A + eye(N, format="csr")).astype(np.float32)
deg    = np.asarray(A_gcn.sum(1)).flatten()
Drt    = diags(1.0 / np.sqrt(deg), dtype=np.float32)
A_hat  = (Drt @ A_gcn @ Drt).astype(np.float32)         # GCN symmetric norm

deg2   = np.asarray(A.sum(1)).flatten(); deg2[deg2 == 0] = 1.0
A_sage = (diags(1.0 / deg2, dtype=np.float32) @ A).astype(np.float32)  # SAGE row norm

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

cw_balanced = compute_class_weight("balanced", classes=np.arange(C), y=labels[tr])

# --- 5. Helpers --------------------------------------------------------------
def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=1, keepdims=True)

def ce_loss(probs):
    p = np.clip(probs[tr, labels[tr]], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))

def dropout_fwd(H, p, training):
    if not training or p == 0.0:
        return H, None
    mask = (rng.random(H.shape) >= p).astype(np.float32)
    return H * mask / (1.0 - p), mask

def dropout_bwd(dH, mask, p):
    return dH if mask is None else dH * mask / (1.0 - p)

def macro_f1(probs, mask):
    return f1_score(labels[mask], probs[mask].argmax(1), average="macro", zero_division=0)

# --- 6. Adam optimizer -------------------------------------------------------
class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, wd=5e-4):
        self.lr, self.beta1, self.beta2, self.eps, self.wd = lr, beta1, beta2, eps, wd
        self.t, self._m, self._v = 0, {}, {}

    def step(self, named_grads):
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        for name, (param, grad) in named_grads.items():
            if name not in self._m:
                self._m[name] = np.zeros_like(param)
                self._v[name] = np.zeros_like(param)
            self._m[name] = self.beta1 * self._m[name] + (1 - self.beta1) * grad
            self._v[name] = self.beta2 * self._v[name] + (1 - self.beta2) * grad * grad
            m_hat = self._m[name] / bc1
            v_hat = self._v[name] / bc2
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps) + self.lr * self.wd * param

# --- 7. GCN ------------------------------------------------------------------
class GCN:
    """
    2-layer GCN.  Accepts precomputed AX = A_hat @ X (fixed per feature set).
    """
    def __init__(self, AX_pre, in_dim):
        self.AX  = AX_pre                                          # (N, in_dim)
        self.W1  = (rng.standard_normal((in_dim, HIDDEN)) * np.sqrt(2.0 / in_dim)).astype(np.float32)
        self.W2  = (rng.standard_normal((HIDDEN, C))      * np.sqrt(2.0 / HIDDEN)).astype(np.float32)
        self.opt = Adam(lr=LR, wd=WD)

    def forward(self, training=False):
        self._Z1     = self.AX @ self.W1
        H1           = np.maximum(0.0, self._Z1)
        H1, self._dm = dropout_fwd(H1, DROPOUT, training)
        self._H1     = H1
        AH1          = np.asarray(A_hat @ H1, dtype=np.float32)
        self._AH1    = AH1
        return softmax(AH1 @ self.W2)

    def backward(self, probs):
        n   = int(tr.sum())
        dZ2 = np.zeros((N, C), np.float32)
        d   = probs[tr].copy(); d[np.arange(n), labels[tr]] -= 1.0; d /= n
        dZ2[tr] = d
        dW2     = self._AH1.T @ dZ2
        dH1     = np.asarray(A_hat @ (dZ2 @ self.W2.T), dtype=np.float32)
        dH1_raw = dropout_bwd(dH1, self._dm, DROPOUT)
        dZ1     = dH1_raw * (self._Z1 > 0).astype(np.float32)
        dW1     = self.AX.T @ dZ1
        self.opt.step({"W1": (self.W1, dW1), "W2": (self.W2, dW2)})

# --- 8. GraphSAGE ------------------------------------------------------------
class GraphSAGE:
    """
    2-layer GraphSAGE (mean aggregator, concat self + neighbours).
    Accepts precomputed AGG1 = concat(X, A_sage @ X) (fixed per feature set).
    """
    def __init__(self, AGG1_pre, in_dim):
        self.AGG1 = AGG1_pre                                       # (N, 2*in_dim)
        self.W1   = (rng.standard_normal((2 * in_dim, HIDDEN)) * np.sqrt(2.0 / (2 * in_dim))).astype(np.float32)
        self.W2   = (rng.standard_normal((2 * HIDDEN, C))      * np.sqrt(2.0 / (2 * HIDDEN))).astype(np.float32)
        self.opt  = Adam(lr=LR, wd=WD)

    def forward(self, training=False):
        self._Z1     = self.AGG1 @ self.W1
        H1           = np.maximum(0.0, self._Z1)
        H1, self._dm = dropout_fwd(H1, DROPOUT, training)
        self._H1     = H1
        AH1          = np.asarray(A_sage @ H1, dtype=np.float32)
        agg2         = np.hstack([H1, AH1])
        self._agg2   = agg2
        self._AH1    = AH1
        return softmax(agg2 @ self.W2)

    def backward(self, probs):
        n   = int(tr.sum())
        dZ2 = np.zeros((N, C), np.float32)
        d   = probs[tr].copy(); d[np.arange(n), labels[tr]] -= 1.0; d /= n
        dZ2[tr] = d
        dW2     = self._agg2.T @ dZ2
        dagg2   = dZ2 @ self.W2.T
        dH1     = (dagg2[:, :HIDDEN]
                   + np.asarray(A_sage.T @ dagg2[:, HIDDEN:], dtype=np.float32))
        dH1_raw = dropout_bwd(dH1, self._dm, DROPOUT)
        dZ1     = dH1_raw * (self._Z1 > 0).astype(np.float32)
        dW1     = self.AGG1.T @ dZ1
        self.opt.step({"W1": (self.W1, dW1), "W2": (self.W2, dW2)})

# --- 9. Training + evaluation ------------------------------------------------
def train_model(model, model_name):
    best_f1          = -1.0
    best_W1, best_W2 = model.W1.copy(), model.W2.copy()
    wait             = 0
    history          = {"loss": [], "val_f1": []}
    t0               = time.time()

    for epoch in range(1, EPOCHS + 1):
        probs = model.forward(training=True)
        loss  = ce_loss(probs)
        model.backward(probs)

        pe    = model.forward(training=False)
        vf1   = macro_f1(pe, val)
        history["loss"].append(loss)
        history["val_f1"].append(vf1)

        if vf1 > best_f1:
            best_f1          = vf1
            best_W1, best_W2 = model.W1.copy(), model.W2.copy()
            wait             = 0
            marker           = " *"
        else:
            wait += 1; marker = ""

        if epoch % 50 == 0 or epoch == 1:
            print(f"    epoch {epoch:4d}  loss={loss:.4f}  val-F1={vf1:.4f}  best={best_f1:.4f}  wait={wait}{marker}")

        if wait >= PATIENCE:
            print(f"    Early stop at epoch {epoch}")
            break

    model.W1[:] = best_W1
    model.W2[:] = best_W2
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s  |  best val macro-F1: {best_f1:.4f}")
    return history

def evaluate_probs(probs, model_name, feat_name, print_report=False):
    pred = probs[te].argmax(1)
    true = labels[te]
    mf1  = f1_score(true, pred, average="macro",    zero_division=0)
    acc  = float((pred == true).mean())
    if print_report:
        print(f"\n  [{feat_name} / {model_name}]  acc={acc:.4f}  macro-F1={mf1:.4f}")
        print(classification_report(true, pred, zero_division=0))
    return pred, mf1, acc

def run_sgc(X_feat, feat_name, k=2):
    """k-hop smoothing + logistic regression. Accepts sparse or dense X."""
    Xs = X_feat.copy() if not hasattr(X_feat, 'toarray') else X_feat.toarray().astype(np.float32)
    for _ in range(k):
        Xs = np.asarray(A_hat @ Xs, dtype=np.float32)
    cw_dict = {c: cw_balanced[c] for c in range(C)}
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight=cw_dict,
                             solver="saga", random_state=SEED)
    clf.fit(Xs[tr], labels[tr])
    pred_te  = clf.predict(Xs[te])
    pred_val = clf.predict(Xs[val])
    mf1_test = f1_score(labels[te],  pred_te,  average="macro", zero_division=0)
    mf1_val  = f1_score(labels[val], pred_val, average="macro", zero_division=0)
    acc      = float((pred_te == labels[te]).mean())
    return pred_te, mf1_test, acc

# --- 10. Run all experiments -------------------------------------------------
all_results = {}   # {(feat_name, model_name): (macro_f1, accuracy)}
all_histories = {} # {(feat_name, model_name): history}
all_preds     = {} # {(feat_name, model_name): pred}

# Experiment configs:
# SGC runs on all three feature sets (sparse-OK)
# GCN / SAGE run on the two SVD-compressed versions
GNN_FEAT = {
    "freq_svd128": X_svd,    # freq+L2 -> SVD-128
}
SGC_FEAT = {
    "freq_norm":   X_freq,   # sparse freq+L2 (N, D)
    "freq_svd128": X_svd,    # dense SVD-128
}

print("\n" + "="*60)
print("  SGC baseline (all feature sets)")
print("="*60)
for fname, Xf in SGC_FEAT.items():
    print(f"\n  SGC | {fname}")
    pred, mf1, acc = run_sgc(Xf, fname)
    all_results[(fname, "SGC")] = (mf1, acc)
    all_preds[(fname, "SGC")]   = pred
    print(f"    acc={acc:.4f}  macro-F1={mf1:.4f}")

print("\n" + "="*60)
print("  GCN  (freq+L2->SVD-128)")
print("="*60)
for fname, Xf in GNN_FEAT.items():
    print(f"\n  GCN | {fname}")
    AX_pre = np.asarray(A_hat @ Xf, dtype=np.float32)
    gcn = GCN(AX_pre, in_dim=SVD_DIM)
    hist = train_model(gcn, f"GCN/{fname}")
    probs = gcn.forward(training=False)
    pred, mf1, acc = evaluate_probs(probs, "GCN", fname)
    all_results[(fname, "GCN")]   = (mf1, acc)
    all_histories[(fname, "GCN")] = hist
    all_preds[(fname, "GCN")]     = pred

print("\n" + "="*60)
print("  GraphSAGE  (freq+L2->SVD-128)")
print("="*60)
for fname, Xf in GNN_FEAT.items():
    print(f"\n  GraphSAGE | {fname}")
    AX_s  = np.asarray(A_sage @ Xf, dtype=np.float32)
    AGG1  = np.hstack([Xf, AX_s]).astype(np.float32)
    sage = GraphSAGE(AGG1, in_dim=SVD_DIM)
    hist = train_model(sage, f"SAGE/{fname}")
    probs = sage.forward(training=False)
    pred, mf1, acc = evaluate_probs(probs, "GraphSAGE", fname)
    all_results[(fname, "GraphSAGE")]   = (mf1, acc)
    all_histories[(fname, "GraphSAGE")] = hist
    all_preds[(fname, "GraphSAGE")]     = pred

# --- 11. Summary table -------------------------------------------------------
print("\n" + "="*68)
print(f"  {'FEATURE SET':<20}  {'MODEL':<12}  {'Macro-F1':>10}  {'Accuracy':>10}")
print("="*68)
for (fname, mname), (mf1, acc) in sorted(all_results.items(), key=lambda x: (-x[1][0], x[0])):
    print(f"  {fname:<20}  {mname:<12}  {mf1:>10.4f}  {acc:>10.4f}")

# --- 12. Plots ---------------------------------------------------------------
def plot_training(histories, fname):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for (fn, mn), h in histories.items():
        if fn != fname:
            continue
        axes[0].plot(h["loss"],   label=mn)
        axes[1].plot(h["val_f1"], label=mn)
    axes[0].set(title=f"Training loss [{fname}]", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title=f"Val macro-F1 [{fname}]",  xlabel="Epoch", ylabel="Macro-F1")
    for ax in axes:
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = f"training_{fname}.png"
    plt.savefig(out, dpi=110); plt.close()
    print(f"  Saved: {out}")

def plot_comparison(all_results):
    models = ["SGC", "GCN", "GraphSAGE"]
    feats  = sorted({fn for fn, _ in all_results})
    x      = np.arange(len(feats))
    width  = 0.25
    colors = ["steelblue", "teal", "darkorange"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (model, color) in enumerate(zip(models, colors)):
        scores = [all_results.get((fn, model), (0, 0))[0] for fn in feats]
        bars   = ax.bar(x + i * width, scores, width, label=model, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(feats, rotation=12, ha="right")
    ax.set(title="Test Macro-F1 by feature set and model",
           ylabel="Macro-F1", ylim=(0, 1))
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=110); plt.close()
    print("  Saved: comparison.png")

def plot_confusion(true, pred, title, fname):
    cm = confusion_matrix(true, pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                linewidths=0.3, cbar_kws={"label": "Count"})
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    plt.tight_layout()
    plt.savefig(fname, dpi=110); plt.close()
    print(f"  Saved: {fname}")

print("\n--- Generating plots")
for fname in GNN_FEAT:
    plot_training(all_histories, fname)

plot_comparison(all_results)

# Confusion matrices for best model per feature set
for fname in GNN_FEAT:
    best_model = max(
        [(mname, all_results[(fname, mname)][0])
         for mname in ["GCN", "GraphSAGE"] if (fname, mname) in all_results],
        key=lambda x: x[1]
    )[0]
    plot_confusion(labels[te], all_preds[(fname, best_model)],
                   f"{best_model} [{fname}] - Confusion Matrix",
                   f"confusion_{fname}_{best_model.lower()}.png")
