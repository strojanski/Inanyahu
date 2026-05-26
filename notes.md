# LastFM-Asia Node Classification — Notes

## 1. Dataset

The LastFM-Asia dataset is a **social network** where:

- **Nodes** (7,624): LastFM users from Asian countries.
- **Edges** (27,806): undirected mutual-follower relationships.
- **Node features**: a set of artist IDs representing the artists each user likes (bag-of-artists representation).
- **Node labels**: country of origin, encoded as integers 0–17 (18 classes).

The task is **transductive node classification**: predict the country label of each node. The entire graph (all nodes and edges) is visible during training; only the labels are split.

Key properties discoveredP in EDA that directly inform model design:

| Property | Value | Implication |
|---|---|---|
| Edge homophily h | 0.874 | Neighbours share the same country label 87% of the time. GNN message-passing is well-suited. |
| Class imbalance (max/min) | 98x | Macro-F1 is the right metric; accuracy is misleading. |
| Feature sparsity | 95% | Raw binary features need reweighting; popular artists carry little country signal. |
| Artist popularity | Zipf law (gamma=0.78) | Common artists appear in nearly every user's list; IDF reweighting needed. |
| Avg. path length | ~5.1 hops | 2-3 GNN layers can propagate signal across most of the graph. |
| Avg. clustering | 0.219 | High local clustering; deeper GNNs risk over-smoothing. |

---

## 2. Feature Engineering

### Raw Features

Each node has a JSON list of integer artist IDs. These are converted to a **binary matrix** X of shape (N=7624, D=7842) where X[i,a] = 1 if user i likes artist a. This matrix is 95% sparse.

### TF-IDF Reweighting

Raw counts give equal weight to globally popular artists (appearing in thousands of profiles) and rare artists (appearing in only a few). A globally popular artist is uninformative for country prediction because everyone likes them regardless of country.

IDF (Inverse Document Frequency) reweights each artist by how uniquely they appear:

```
idf[a] = log( (N + 1) / (df[a] + 1) ) + 1
```

where `df[a]` is the number of users who like artist `a`. Artists liked by everyone get low IDF; niche artists get high IDF. After multiplying by IDF, each user's feature vector is L2-normalised to unit length.

### SVD Dimensionality Reduction

After TF-IDF, the feature matrix is still (7624, 7842). Truncated SVD (equivalent to PCA for sparse matrices) compresses this to (7624, 128). This:
- Makes training significantly faster (128 dimensions vs 7842).
- Removes noise from very rare artists.
- Produces a dense matrix that matrix multiplication handles efficiently.

The 128 components explain ~30% of total variance, which is reasonable for a 7842-dimensional sparse matrix.

---

## 3. Graph Representation

Two versions of the normalised adjacency matrix are built:

### GCN Adjacency (A_hat)

Standard Kipf & Welling symmetric normalisation:

```
A_hat = D^{-1/2} (A + I) D^{-1/2}
```

Where:
- `A` is the raw adjacency matrix (binary, symmetric, no self-loops).
- `A + I` adds self-loops so each node attends to itself as well as its neighbours.
- `D` is the degree matrix (diagonal) of `A + I`.
- `D^{-1/2} ... D^{-1/2}` normalises so that row sums are bounded — high-degree nodes don't dominate aggregation.

A_hat is **symmetric** (A_hat.T = A_hat), which simplifies backpropagation.

### GraphSAGE Adjacency (A_sage)

Row-normalised adjacency **without** self-loops:

```
A_sage = D^{-1} A
```

Each row sums to 1. Multiplying `A_sage @ H` computes the **mean** of each node's neighbours' representations. GraphSAGE then concatenates this mean with the node's own representation (see Section 5).

---

## 4. Data Split

The split is **stratified** 70/15/15 (train/val/test). Stratification means each class gets the same proportion of its samples in each split. This is required because the class imbalance is 98:1 — a random split would leave some minority classes with 0 or 1 test examples.

```
Train:  5,336 nodes
Val:    1,144 nodes
Test:   1,144 nodes
```

In the transductive setting, all 7,624 nodes are visible in every forward pass (the graph is unchanged). Only the labels are hidden for val/test nodes.

---

## 5. Models

### SGC (Simple Graph Convolutional) — Baseline

SGC (Wu et al., 2019) is the simplest possible graph model: smooth the features with the graph, then fit a linear classifier.

**Step 1 — k-hop smoothing:**

```
X_smooth = A_hat @ A_hat @ X    (k=2 hops)
```

Each node's feature vector becomes a weighted average of its 2-hop neighbourhood. Because homophily is 0.874, this averaging mixes mostly same-country users, making the features more separable.

**Step 2 — Logistic Regression:**

A multinomial logistic regression is fitted on `X_smooth[train]` and evaluated on `X_smooth[test]`. Balanced class weights are used because logistic regression with an L-BFGS/SAGA solver is a convex problem — the balanced weights shift the decision boundary toward minority classes without causing gradient instability.

SGC has **no learnable parameters in the graph part** — it is equivalent to a GCN with the non-linearities removed. It is a strong baseline precisely because the graph smoothing (which is the expensive/complex part of GNNs) can be precomputed.

**Test macro-F1: 0.621**

---

### GCN (Graph Convolutional Network) — Kipf & Welling (2017)

GCN adds **learnable nonlinear transformations** at each propagation step.

**Forward pass (2 layers):**

```
Layer 1:  H1 = ReLU( A_hat @ X @ W1 )
Layer 2:  Z  = A_hat @ dropout(H1) @ W2
probs    = softmax(Z)
```

Where:
- `W1` has shape (128, 256): projects raw features into a hidden representation.
- `W2` has shape (256, 18): projects the hidden representation to class logits.
- `A_hat @ X` (layer 1) and `A_hat @ H1` (layer 2) aggregate neighbour information.
- `dropout(H1)` randomly zeros 50% of hidden units during training (reduces overfitting).
- `softmax` converts logits to class probabilities.

`A_hat @ X` is **precomputed once** before training since X never changes. `A_hat @ H1` must be recomputed each step because H1 changes as weights update.

**Loss function:**

Standard cross-entropy over train nodes:

```
loss = -(1/n_train) * sum_{i in train} log( probs[i, y[i]] )
```

No class weights are used. (See Section 6 for why.)

**Backpropagation:**

Because the graph aggregation `A_hat @ H` is a linear operation, gradients flow through it as `A_hat.T @ dH = A_hat @ dH` (since A_hat is symmetric). The gradient from loss-on-train-nodes propagates to all nodes via the sparse matrix product:

```
d(loss)/d(H1) = A_hat @ ( d(loss)/d(Z) @ W2.T )
```

This means: even though loss is computed only on train nodes, gradients propagate to all nodes through the graph structure. This is what makes GCNs transductive — the full graph participates in every update.

**Test macro-F1: 0.601**

---

### GraphSAGE (Hamilton et al., 2017)

GraphSAGE uses a **concatenation** aggregation instead of sum. At each layer, each node combines its own representation with the mean of its neighbours', giving the model more expressive power to distinguish between self and neighbourhood.

**Forward pass (2 layers):**

```
Layer 1:  agg1 = concat( X,  A_sage @ X )       shape: (N, 256)
          H1   = ReLU( agg1 @ W1 )               shape: (N, 256) [hidden=256]

Layer 2:  agg2 = concat( H1, A_sage @ H1 )       shape: (N, 512)
          Z    = agg2 @ W2                        shape: (N, 18)
probs    = softmax(Z)
```

Where:
- `A_sage @ X` computes the **mean of each node's neighbours' features**.
- `concat` keeps both the node's own features AND its neighbourhood mean, rather than merging them.
- `W1` has shape (256, 256): input is 2*128=256 (self + mean-neigh of 128-dim features).
- `W2` has shape (512, 18): input is 2*256=512 (self + mean-neigh of 256-dim hidden).

The concatenation rather than sum means the model can learn different weights for "what I know about myself" vs "what my neighbourhood tells me". This is why GraphSAGE outperforms GCN here.

`concat(X, A_sage @ X)` for layer 1 is precomputed since both X and A_sage are fixed. Layer 2's concat must be recomputed each step.

**Backpropagation:**

The gradient through the neighbour path uses `A_sage.T` (NOT `A_sage`) because A_sage is row-normalised and therefore asymmetric:

```
d(loss)/d(H1) = d(agg2)/d(H1_self)   +   A_sage.T @ d(agg2)/d(A_sage@H1)
             = dagg2[:, :HIDDEN]      +   A_sage.T @ dagg2[:, HIDDEN:]
```

**Test macro-F1: 0.733** — best of the three models.

---

## 6. Optimizer and Training

### Adam Optimizer

Vanilla SGD (weight -= lr * gradient) does not work well for GNNs because:
- Gradient magnitudes vary enormously across layers (graph convolution diffuses and scales gradients).
- Near initialisation, loss landscape has high curvature.

Adam maintains **per-parameter moving averages** of the gradient (first moment m) and squared gradient (second moment v):

```
m = 0.9 * m + 0.1 * gradient          (exponential moving average)
v = 0.999 * v + 0.001 * gradient^2    (exponential moving average of squared grad)

param -= lr * (m / sqrt(v) + eps)      (update)
```

The `sqrt(v)` denominator adapts the effective learning rate: parameters with large gradients get smaller updates (prevents explosion), parameters with tiny gradients get larger updates (prevents vanishing). Weight decay (`param -= lr * wd * param`) penalises large weights to reduce overfitting.

### Why No Balanced Class Weights in the GNN

With balanced class weights, each class contributes equally to the gradient regardless of size. This sounds beneficial, but it creates a problem at initialisation:

When a neural network is randomly initialised, all class probabilities are approximately equal (~1/18). With balanced weights, the positive and negative gradient contributions from each class cancel out exactly — the gradient for every weight is mathematically zero. The model cannot start learning.

Logistic regression avoids this because it is a convex problem: the solver (L-BFGS/SAGA) does not require a gradient with a specific initial magnitude to converge. For gradient descent, the model must be in a non-symmetric state before balanced weights help.

**Solution used here:** train GNNs with standard CE (no class weights), evaluate with macro-F1. Macro-F1 is the primary metric precisely because it gives equal weight to each class regardless of class size, achieving the same goal as balanced weights — without the training instability.

### Early Stopping

After each epoch, the model is evaluated on the validation set with macro-F1. If val macro-F1 does not improve for 50 consecutive epochs, training stops and the best checkpoint is restored. This prevents overfitting to the training set while not requiring a fixed number of epochs.

---

## 7. Results Summary

| Model | Accuracy | Macro-F1 | Weighted-F1 | Epochs |
|---|---|---|---|---|
| GraphSAGE | 86.8% | **0.733** | 0.862 | 492 |
| GCN | 85.2% | 0.601 | 0.835 | 500 |
| SGC-2 (baseline) | 75.5% | 0.621 | 0.782 | — |

**Macro-F1** is the primary metric because 6 of 18 classes have fewer than 15 test examples; accuracy and weighted-F1 are dominated by the majority classes.

GraphSAGE outperforms GCN despite using the same number of layers because the concatenation aggregation preserves more information (self vs. neighbour) than the symmetric normalisation used in GCN.

SGC-2 is a competitive baseline: it matches GCN on macro-F1 and runs 10x faster because it requires no gradient descent, just one matrix multiplication followed by logistic regression. The gap between SGC and the learned models shows that the nonlinear transformations in W1 provide measurable benefit.

---

## 8. File Structure

```
train_gnn.py          - main training script
eda.ipynb             - exploratory data analysis notebook
lasftm_asia/
    lastfm_asia_edges.csv       - edge list (node_1, node_2)
    lastfm_asia_features.json   - {node_id: [artist_id, ...]}
    lastfm_asia_target.csv      - (id, target) node labels
    README.txt                  - dataset citation

training_curves.png   - loss and val macro-F1 per epoch
comparison.png        - final test macro-F1 bar chart
confusion_gcn.png     - GCN confusion matrix (18x18)
confusion_sage.png    - GraphSAGE confusion matrix (18x18)
```
