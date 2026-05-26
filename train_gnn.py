#!/usr/bin/env python3
"""
Node classification on LastFM-Asia using GraphSAGE.

Features: frequency-weighted + L2 + SVD-128

Usage:
    python train_gnn.py
"""

import json, time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# --- Config ------------------------------------------------------------------
DATA     = "lasftm_asia/"
SEED     = 42
SVD_DIM  = 2048
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
inv_df = 1.0 / np.maximum(df_vec, 1)
X_freq = normalize(X_bin.multiply(inv_df), norm="l2")

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

# --- 7. GraphSAGE ------------------------------------------------------------
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

# --- 9. Run GraphSAGE --------------------------------------------------------
print("\n" + "="*60)
print("  GraphSAGE  (freq+L2->SVD-128)")
print("="*60)

AX_s = np.asarray(A_sage @ X_svd, dtype=np.float32)
AGG1 = np.hstack([X_svd, AX_s]).astype(np.float32)
sage = GraphSAGE(AGG1, in_dim=SVD_DIM)
hist = train_model(sage, "GraphSAGE")
probs = sage.forward(training=False)
pred, mf1, acc = evaluate_probs(probs, "GraphSAGE", "freq_svd128", print_report=True)

# --- 10. Bootstrap confidence intervals --------------------------------------
N_BOOT   = 2000
true_te  = labels[te]
n_te     = len(true_te)
boot_acc, boot_f1 = np.empty(N_BOOT), np.empty(N_BOOT)
for i in range(N_BOOT):
    idx = rng.integers(0, n_te, size=n_te)
    boot_acc[i] = (pred[idx] == true_te[idx]).mean()
    boot_f1[i]  = f1_score(true_te[idx], pred[idx], average="macro", zero_division=0)

ci_acc = np.percentile(boot_acc, [2.5, 97.5])
ci_f1  = np.percentile(boot_f1,  [2.5, 97.5])
print(f"\n  Bootstrap (n={N_BOOT})  95% CI")
print(f"  Accuracy : {boot_acc.mean():.4f}  [{ci_acc[0]:.4f}, {ci_acc[1]:.4f}]"
      f"  (±{(ci_acc[1]-ci_acc[0])/2:.4f})")
print(f"  Macro-F1 : {boot_f1.mean():.4f}  [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}]"
      f"  (±{(ci_f1[1]-ci_f1[0])/2:.4f})")

# --- 11. Plots ---------------------------------------------------------------
print("\n--- Generating plots")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(hist["loss"])
axes[1].plot(hist["val_f1"])
axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
axes[1].set(title="Val macro-F1",  xlabel="Epoch", ylabel="Macro-F1")
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("training_graphsage.png", dpi=110); plt.close()
print("  Saved: training_graphsage.png")

cm = confusion_matrix(labels[te], pred)
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            linewidths=0.3, cbar_kws={"label": "Count"})
ax.set(xlabel="Predicted", ylabel="True", title="GraphSAGE [freq_svd128] - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_graphsage.png", dpi=110); plt.close()
print("  Saved: confusion_graphsage.png")
