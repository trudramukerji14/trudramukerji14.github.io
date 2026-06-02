---
title: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
date: 2026-06-02
draft: false
math: true
tags: ["machine-learning", "quantization", "llms"]
description: "A walkthrough of GPTQ"
---
In this article, I'd like to explore [GPTQ](https://arxiv.org/abs/2210.17323), one-shot weight quantization method introduced by Frantar et al. based on approximate second-order information. This work dealt with the challenging task of quantizing GPT models with 175 billion parameters while preserving a reasonable amount of accuracy despite the extreme quantization involved. This post describes quantization, the OBQ framework, and the motivation and choices that led to GPTQ. Along the way I include a worked example, a PyTorch implementation applied to GPT-2, and perplexity results on WikiText-2. 

## Background


### The Quantization Problem
Let's recall the problem of quantization, we define $$Q_{sym} = \{sq : q \in \{-2^{b-1}+1, \ldots, 2^{b-1}-1\}\}$$ to be the set of finite symmetric set of $b$-bit integers scaled by $s$. We can also define asymmetric analog as
$$Q_{asym} = \{sq : q \in \{0, \ldots, 2^{b}-1\}\}.$$ We abuse notation and let $Q$ refer to either one of these two sets. Given a real number $r$, we let $\hat{r}$ be its quantization, that is a nearest member of $Q$ to $r$. 

Let $W \in \mathbb{R}^{d_{row} \times d_{col}}$, in quantization, we are looking to find $\hat{W} \in Q^{d_{row} \times d_{col}}$ such that the loss such that the loss
$$L(\hat{W}) = \|WX - \hat{W}X \|_{2}^{2}$$
is minimized for $X \in \mathbb{R}^{d_{col} \times n}$. Alternatively, we can set $\delta = (W-\hat{W})$ and view this as function of $\delta$. We can actually view this loss as the $L_{2}^{2}$ of each of the rows
$$
L(W - \delta) = \sum^{d_{row}}_{{k}} \|w_{k}X - (w_{k}-\delta_{k})X \|_{2}^{2}.
$$
where $\delta_{k}= (w_{k}-\hat{w}_{k})$. Though the exact computation of this problem is NP-Hard, approximate solutions to this problem has a natural application in deep learning in trying to quantify the weights of a trained uncompressed neural network that is still accurate enough to be used during inference. We dicuss a few thoughts in this direction before arriving at GPTQ. 

## Optimal Brain Quantization (OBQ)

The ideas in GPTQ were preceeding by previous work in the form of the Optimal Brain Quantization (OBQ) framework for quantization. This OBQ framework combined weight pruning framework in earlier work by LeCun et all and later Hassibi and Stork, who introduced Optimal Brain Surgeon (OBS) with approaches like AdaPrune by  that break the quantization of the entire weight space into layer by layer subproblems.  

That is, setting $\delta = (W-\hat{W})$. we can actually view the problem above as minimizing the $L_{2}^{2}$ of each of the rows $w_{k}$ of $W$
$$L(W - \delta) = \sum^{d_{row}}_{{k}} \|w_{k}X - (w_{k}-\delta_{k})X \|_{2}^{2}.$$
where $\delta_{k}= (w_{k}-\hat{w}_{k})$. Simplifying each term of the sum to $\|\delta_{k}X\|^{2}_{2}$, we obtain that we are summing $\delta_{k}XX^{T}\delta_{k}^{T}$ and realize that $XX^{T}$ serves as the Hessian of the per row loss with respect to $\delta_{k}$.

As a brief aside, in the OBS framework, we instead have a general network loss $\ell$ and write the Taylor expansion around the pre-trained weights $w_{k}$:
$$\ell(w_{k} + \delta_{k}) = \ell(w_{k}) + \nabla_{w_{k}}\ell \cdot \delta_{k}^{T} + \frac{1}{2}\delta_{k}\mathbf{H}\delta_{k}^{T} + \mathcal{O}(\|\delta_{k}\|^{3})$$
where $\nabla_{w_{k}}\ell = 0$ is assumed (the network has converged to a local minimum) and the cubic remainder is discarded. In OBQ, by contrast, both of these hold exactly: $\ell(0) = 0$, $\nabla_{\delta_{k}}\ell|_{\delta_{k}=0} = 0$ by direct computation, and the loss is exactly quadratic with no higher-order terms.

 Suppose in our minimization of the $k$th row, we apply the constraint where $w_{kq}$ is adjusted to its quantized value $\hat{w}_{kq}$. This means we set the $q$-th coordinate of $\delta_{k}$ to be $\hat{\delta}_{kq} =\hat{w}_{kq} - w_{kq}$. This means we are solving the optimization problem

$$
\begin{aligned}
&\text{minimize} && \frac{1}{2}\delta_{k}^{T}\mathbf{H}\delta_{k} \\
&\text{subject to} && e_{q}^{T}\delta_{k} - \hat{\delta}_{kq} = 0
\end{aligned}
$$

This prompts us to consider the Lagrangian $$\mathcal{L}(\delta_{k}) = \frac{1}{2}\delta^{T}_{k}\mathbf{H}\delta_{k} + \lambda(e_{q}^{T}\delta_{k} - \hat{\delta_{kq}}).$$ Taking $\nabla_{\delta_{k}}\mathcal{L} = \mathbf{H}\delta_{k} + \lambda e_{q}$ and setting it to zero, we obtain a minimizer at $$\delta^{*}_{k} = -\lambda \mathbf{H}^{-1}e_{q}.$$ 
Substituting this into the constraint $e_{q}^{T}\delta_{k} = \hat{\delta}_{kq}$, we get $-\lambda [\mathbf{H}^{-1}]_{qq} = \hat{\delta_{kq}}$ and $\lambda = -\frac{\hat{\delta_{kq}}}{[\mathbf{H}^{-1}]_{qq}}$.  This leads to
$$\delta^{*}_{k} = \frac{\hat{\delta_{kq}}}{[\mathbf{H}^{-1}]_{qq}} \mathbf{H}^{-1}e_{q}$$
Also, substituting $\delta^{*}_{k}$ into $\frac{1}{2}\delta_{k}^{T}\mathbf{H}\delta_{k}$ gives the saliency score at row $k$ and column $q$: 
$$ Saliency_{kq}= \frac{\hat{\delta}_{kq}^{2}}{2[\mathbf{H}^{-1}]_{qq}}.$$
This computation motivates the OBQ updates given in (2) of the GPTQ paper for a given row $k$ and set of available column indices $F$:
\begin{equation}
    w_{kq} = \text{argmin}_{\{w_{kc}: c \in  F \}} \frac{(\hat{w}_{kq}-w_{kc})^{2}}{2[\mathbf{H}^{-1}]_{cc}} \tag{2}
\end{equation}
and associated update
\begin{equation}
    \delta_{F} = \frac{(\hat{w_{kq}}-w_{kq})}{2[\mathbf{H}^{-1}]_{qq}} \mathbf{H}^{-1}e_{q} \tag{3}
\end{equation}
We are essentially considering all possible column indices for an optimal update $\delta^{*}$ where the Saliency score is the loss that is incurred from that update.

### OBQ Algorithm and Complexity

We now state the OBQ algorithm:

<div style="border:1px solid #888; border-radius:4px; padding:1em 1.5em; margin:1.5em 0; line-height:2;">
<strong>Algorithm: OBQ</strong><br>
<strong>Input:</strong> $\mathbf{W} \in \mathbb{R}^{d_{\text{row}} \times d_{\text{col}}}$, $\mathbf{H}^{-1}$ &nbsp;&nbsp;&nbsp; <strong>Output:</strong> Quantized $\hat{\mathbf{W}}$<br><br>
1 &nbsp;&nbsp; <strong>for</strong> $i = 0, 1, \ldots, d_{\text{row}}-1$ <strong>do</strong><br>
2 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $F \gets \{0, 1, \ldots, d_{\text{col}}-1\}$<br>
3 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>while</strong> $|F| > 0$ <strong>do</strong><br>
4 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $q \gets \operatorname{argmin}_{c \in F} \dfrac{(\operatorname{quant}(w_{ic}) - w_{ic})^{2}}{2[\mathbf{H}^{-1}_F]_{cc}}$<br>
5 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\hat{w}_{iq} \gets \operatorname{quant}(w_{iq})$<br>
6 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\delta_q \gets \hat{w}_{iq} - w_{iq}$<br>
7 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>for</strong> $j \in F \setminus \{q\}$ <strong>do</strong><br>
8 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $w_{ij} \gets w_{ij} - \dfrac{\delta_q}{[\mathbf{H}^{-1}_F]_{qq}}\,[\mathbf{H}^{-1}_F]_{qj}$<br>
9 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>end for</strong><br>
10 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\mathbf{H}^{-1}_F \gets$ remove row and column $q$ from $\mathbf{H}^{-1}_F$<br>
11 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $F \gets F \setminus \{q\}$<br>
12 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>end while</strong><br>
13 &nbsp;&nbsp; <strong>end for</strong>
</div>

Here we can give an estimate of the complexity. The step obtaining $\mathbf{H}^{-1}_{F}$ after removing row and column $q$ is given by equation (3) of the GPTQ paper:

\begin{equation}
    \mathbf{H}^{-1}_{-q} = \left( \mathbf{H}^{-1} - \frac{1}{[\mathbf{H}^{-1}_{F}]_{qq}}\mathbf{H}^{-1}_{:q}\mathbf{H}^{-1}_{q:} \right)_{-q} \tag{4}
\end{equation}

Here $\mathbf{H}^{-1}_{:q}\mathbf{H}^{-1}_{q:}$ denotes the outer product of the $q$-th column of $\mathbf{H}^{-1}$ with itself. This step is $O(d_{\text{col}}^{2})$ per column, giving $O(d_{\text{col}}^{3})$ per row and $O(d_{\text{row}} d_{\text{col}}^{3})$ complexity total.
## GPTQ

The GPTQ algorithm improves upon OBQ via three modifications.

**1. Fixed column order:** Instead of finding the optimal column $q$ via (2), we quantize columns in fixed order $0, 1, \ldots, d_{\text{col}}-1$. Since every row processes columns in the same order, all rows share a single $\mathbf{H}^{-1}$.

For example, at step $0$ we quantize column $0$ across all rows simultaneously. Each row $r$ has quantization error $\delta_0^{(r)} = \operatorname{quant}(w_{r0}) - w_{r0}$, and the optimal compensation to the remaining weights in row $r$ is:

$$\boldsymbol{\delta}^{(r)}_{0} = \frac{\delta_0^{(r)}}{[\mathbf{H}^{-1}]_{00}} \mathbf{H}^{-1}_{:,0}$$

where $\mathbf{H}^{-1}_{:,0}$ is the first column of $\mathbf{H}^{-1}$. Stacking these as rows gives the full update:

\begin{equation}
    \boldsymbol{\Delta}^{*} = \frac{1}{[\mathbf{H}^{-1}]_{00}} \mathbf{E}_0\, (\mathbf{H}^{-1}_{:,0})^{\top} \in \mathbb{R}^{d_{\text{row}} \times d_{\text{col}}} \tag{5}
\end{equation}

where $\mathbf{E}_{0} = (\boldsymbol{\delta}_0^{(0)},\, \boldsymbol{\delta}_0^{(1)},\, \ldots,\, \boldsymbol{\delta}_0^{(d_{\text{row}})})^{\top} \in \mathbb{R}^{d_{\text{row}} \times 1}$ is the vector of per-row quantization errors. The remaining weights are updated as $\mathbf{W} \leftarrow \mathbf{W} - \boldsymbol{\Delta}^{*}$. Each row of $\boldsymbol{\Delta}^{*}$ shares the same $\mathbf{H}^{-1}$ terms, which is the key efficiency gain.

Moving to the next column, we apply equation (4) to remove the $0$th row and column from $\mathbf{H}^{-1}$, costing $O(d_{\text{col}}^{2})$, then apply the weight update to $\mathbf{W}_{:,1:} \in \mathbb{R}^{d_{\text{row}} \times (d_{\text{col}}-1)}$ in $O(d_{\text{row}} d_{\text{col}})$. Repeating for all $d_{\text{col}}$ columns gives total complexity $O(d_{\text{row}} d_{\text{col}}^{2}) + O(d_{\text{col}}^{3})$.

**2. Lazy batch update:** The per-column updates in (4) and (5) are GEMV operations with low compute-to-memory ratios. GPTQ resolves this with batching.

We fix a block size $B$ and process columns in groups. For a block starting at column $i$ with index set $Q = \{i, \ldots, \min(i+B-1,\, d_{\text{col}}-1)\}$, we apply the rank-1 updates of (5) *only to columns inside $Q$*, leaving trailing columns untouched. As we iterate through the block we accumulate scaled errors:

$$\mathbf{E} \in \mathbb{R}^{d_{\text{row}} \times B}, \qquad \mathbf{E}_{:,\, j-i} = \frac{\mathbf{W}_{:,j} - \operatorname{quant}(\mathbf{W}_{:,j})}{[\mathbf{H}^{-1}]_{jj}}, \quad j \in Q.$$

After the block is fully processed, all $B$ errors are propagated to the trailing region in one matrix multiplication:

\begin{equation}
    \mathbf{W}_{:,(i+B):} \gets \mathbf{W}_{:,(i+B):} - \mathbf{E} \cdot \mathbf{H}^{-1}_{i:(i+B),\,(i+B):} \tag{6}
\end{equation}

where $\mathbf{W}_{:,(i+B):} \in \mathbb{R}^{d_{\text{row}} \times (d_{\text{col}} - i - B)}$, $\mathbf{E} \in \mathbb{R}^{d_{\text{row}} \times B}$, and $\mathbf{H}^{-1}_{i:(i+B),\,(i+B):} \in \mathbb{R}^{B \times (d_{\text{col}} - i - B)}$.

Update (6) is mathematically equivalent to applying (5) column-by-column to the trailing region. The total flop count is unchanged, but the memory access pattern improves: instead of $B$ low-arithmetic-intensity GEMV passes through HBM, we make a single rank-$B$ GEMM pass. Each trailing weight is loaded once and participates in $B$ multiply-adds, raising arithmetic intensity from $O(1)$ to $O(B)$ flops per byte. In practice this yields roughly an order-of-magnitude speedup on large layers.

The multi-column generalizations of (4) and (5) are:

$$\boldsymbol{\delta}_{-Q} = -(\mathbf{w}_Q - \operatorname{quant}(\mathbf{w}_Q)) \, ([\mathbf{H}^{-1}]_{QQ})^{-1} \, \mathbf{H}^{-1}_{Q,\,-Q}$$

$$\mathbf{H}^{-1}_{-Q} = \big(\mathbf{H}^{-1} - \mathbf{H}^{-1}_{:,\,Q} \, ([\mathbf{H}^{-1}]_{QQ})^{-1} \, \mathbf{H}^{-1}_{Q,\,:}\big)_{-Q}$$

The Cholesky reformulation below eliminates the second update entirely.

**3. Cholesky reformulation:** Repeatedly applying the block Schur complement (4) accumulates floating-point error and can drive $\mathbf{H}^{-1}$ indefinite — the GPTQ paper reports this almost certainly occurs on at least a few layers for models above a few billion parameters.

The key observation is that at step $j$, the algorithm only needs row $j$ (from the diagonal onward) of the current shrunken inverse $\mathbf{H}^{-1}_{F_j}$ where $F_j = \{j, \ldots, d_{\text{col}}-1\}$. The sequence of Schur complement updates producing these rows is — up to a row-scaling by $([\mathbf{H}^{-1}]_{jj})^{1/2}$ — exactly Cholesky factorization. So if we compute the upper Cholesky factor $\mathbf{U}$ of $\mathbf{H}^{-1}$ upfront (with $\mathbf{H}^{-1} = \mathbf{U}^{\top}\mathbf{U}$), then row $j$ of $\mathbf{U}$ from column $j$ onward gives exactly the coefficients needed at step $j$.

This has three consequences. First, $\mathbf{H}^{-1}$ is inverted once before the loop and never updated again — the block Schur complement update is eliminated entirely. Second, one optimized Cholesky kernel replaces $d_{\text{col}}/B$ Schur-complement updates. Third, mild dampening ($\mathbf{H} \gets \mathbf{H} + \lambda\mathbf{I}$, typically $\lambda = 1\%$ of the mean diagonal) makes the upfront Cholesky numerically robust even on very large models.

The resulting algorithm is:

<div style="border:1px solid #888; border-radius:4px; padding:1em 1.5em; margin:1.5em 0; line-height:2;">
<strong>Algorithm 2: GPTQ</strong><br>
<strong>Input:</strong> $\mathbf{W} \in \mathbb{R}^{d_{\text{row}} \times d_{\text{col}}}$, $\mathbf{H}^{-1}$, block size $B$ &nbsp;&nbsp;&nbsp; <strong>Output:</strong> Quantized $\hat{\mathbf{Q}}$<br><br>
1 &nbsp;&nbsp; $\mathbf{Q} \gets \mathbf{0}_{d_{\text{row}} \times d_{\text{col}}}$<br>
2 &nbsp;&nbsp; $\mathbf{H}^{-1} \gets \operatorname{Cholesky}(\mathbf{H}^{-1})^{\top}$<br>
3 &nbsp;&nbsp; <strong>for</strong> $i = 0, B, 2B, \ldots$ <strong>do</strong><br>
4 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\mathbf{E} \gets \mathbf{0}_{d_{\text{row}} \times B}$<br>
5 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>for</strong> $j = i, \ldots, i + B - 1$ <strong>do</strong><br>
6 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\mathbf{Q}_{:,j} \gets \operatorname{quant}(\mathbf{W}_{:,j})$ &nbsp;&nbsp; <em>// quantize column</em><br>
7 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\mathbf{E}_{:,j-i} \gets (\mathbf{W}_{:,j} - \mathbf{Q}_{:,j})\,/\,[\mathbf{H}^{-1}]_{jj}$ &nbsp;&nbsp; <em>// scaled error</em><br>
8 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\mathbf{W}_{:,j:(i+B)} \gets \mathbf{W}_{:,j:(i+B)} - \mathbf{E}_{:,j-i} \cdot \mathbf{H}^{-1}_{j,\,j:(i+B)}$ &nbsp;&nbsp; <em>// update within block</em><br>
9 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>end for</strong><br>
10 &nbsp;&nbsp;&nbsp; $\mathbf{W}_{:,(i+B):} \gets \mathbf{W}_{:,(i+B):} - \mathbf{E} \cdot \mathbf{H}^{-1}_{i:(i+B),\,(i+B):}$ &nbsp;&nbsp; <em>// lazy propagation</em><br>
11 &nbsp;&nbsp; <strong>end for</strong>
</div>

### A worked example of GPTQ

Let's consider a $2 \times 6$ matrix with block size $B=3$:

<div style="overflow-x:auto; margin:1em 0;">
<table style="border-collapse:collapse; margin:0 auto; font-size:0.95em;">
<tbody>
<tr>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#c8e6c9;">$w_{00}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#c8e6c9;">$w_{01}$</td>
  <td style="border:1px solid #aaa; border-right:3px solid #555; padding:5px 14px; background:#c8e6c9;">$w_{02}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#fff9c4;">$w_{03}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#fff9c4;">$w_{04}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#fff9c4;">$w_{05}$</td>
</tr>
<tr>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#c8e6c9;">$w_{10}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#c8e6c9;">$w_{11}$</td>
  <td style="border:1px solid #aaa; border-right:3px solid #555; padding:5px 14px; background:#c8e6c9;">$w_{12}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#fff9c4;">$w_{13}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#fff9c4;">$w_{14}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#fff9c4;">$w_{15}$</td>
</tr>
</tbody>
</table>
<div style="text-align:center; margin-top:6px; font-size:0.85em;">
<span style="background:#c8e6c9; padding:2px 8px; border-radius:3px; margin-right:8px;">Block 1: cols {0,1,2}</span>
<span style="background:#fff9c4; padding:2px 8px; border-radius:3px;">Remaining: cols {3,4,5}</span>
</div>
</div>

**Step 1: Quantize column 0.**

$$\hat{w}_{r0} = \operatorname{quant}(w_{r0}), \qquad \delta_0^{(r)} = \hat{w}_{r0} - w_{r0}$$

Update columns 1 and 2 **only** (columns 3–5 untouched):

<div style="overflow-x:auto; margin:1em 0;">
<table style="border-collapse:collapse; margin:0 auto; font-size:0.95em;">
<tbody>
<tr>
  <td style="border:1px solid #aaa; border-right:3px solid #555; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{00}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#c8e6c9;">$w_{01}^{\prime}$</td>
  <td style="border:1px solid #aaa; border-right:3px solid #555; padding:5px 14px; background:#c8e6c9;">$w_{02}^{\prime}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#fff9c4;">$w_{03}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#fff9c4;">$w_{04}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#fff9c4;">$w_{05}$</td>
</tr>
<tr>
  <td style="border:1px solid #aaa; border-right:3px solid #555; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{10}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#c8e6c9;">$w_{11}^{\prime}$</td>
  <td style="border:1px solid #aaa; border-right:3px solid #555; padding:5px 14px; background:#c8e6c9;">$w_{12}^{\prime}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#fff9c4;">$w_{13}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#fff9c4;">$w_{14}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#fff9c4;">$w_{15}$</td>
</tr>
</tbody>
</table>
</div>

$$\begin{aligned}
w_{r1}^{\prime} &= w_{r1} - \delta_0^{(r)}\, \frac{[\mathbf{H}^{-1}]_{0,1}}{[\mathbf{H}^{-1}]_{0,0}} \\[4pt]
w_{r2}^{\prime} &= w_{r2} - \delta_0^{(r)}\, \frac{[\mathbf{H}^{-1}]_{0,2}}{[\mathbf{H}^{-1}]_{0,0}}
\end{aligned}$$

**Step 2: Quantize column 1** (using updated $w_{r1}^{\prime}$).

$$\hat{w}_{r1} = \operatorname{quant}(w_{r1}^{\prime}), \qquad \delta_1^{(r)} = \hat{w}_{r1} - w_{r1}^{\prime}$$

Update column 2 **only**:

<div style="overflow-x:auto; margin:1em 0;">
<table style="border-collapse:collapse; margin:0 auto; font-size:0.95em;">
<tbody>
<tr>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{00}$</td>
  <td style="border:1px solid #aaa; border-right:3px solid #555; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{01}$</td>
  <td style="border:1px solid #aaa; border-right:3px solid #555; padding:5px 14px; background:#c8e6c9;">$w_{02}^{\prime\prime}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#fff9c4;">$w_{03}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#fff9c4;">$w_{04}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#fff9c4;">$w_{05}$</td>
</tr>
<tr>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{10}$</td>
  <td style="border:1px solid #aaa; border-right:3px solid #555; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{11}$</td>
  <td style="border:1px solid #aaa; border-right:3px solid #555; padding:5px 14px; background:#c8e6c9;">$w_{12}^{\prime\prime}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#fff9c4;">$w_{13}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#fff9c4;">$w_{14}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#fff9c4;">$w_{15}$</td>
</tr>
</tbody>
</table>
</div>

$$w_{r2}^{\prime\prime} = w_{r2}^{\prime} - \delta_1^{(r)}\, \frac{[\mathbf{H}^{-1}]_{1,2}}{[\mathbf{H}^{-1}]_{1,1}}$$

**Step 3: Quantize column 2** (using twice-updated $w_{r2}^{\prime\prime}$).

$$\hat{w}_{r2} = \operatorname{quant}(w_{r2}^{\prime\prime}), \qquad \delta_2^{(r)} = \hat{w}_{r2} - w_{r2}^{\prime\prime}$$

<div style="overflow-x:auto; margin:1em 0;">
<table style="border-collapse:collapse; margin:0 auto; font-size:0.95em;">
<tbody>
<tr>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{00}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{01}$</td>
  <td style="border:1px solid #aaa; border-right:3px solid #555; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{02}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#fff9c4;">$w_{03}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#fff9c4;">$w_{04}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#fff9c4;">$w_{05}$</td>
</tr>
<tr>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{10}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{11}$</td>
  <td style="border:1px solid #aaa; border-right:3px solid #555; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{12}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#fff9c4;">$w_{03}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#fff9c4;">$w_{04}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#fff9c4;">$w_{05}$</td>
</tr>
</tbody>
</table>
</div>

Columns $\{3,4,5\}$ are owed corrections from quantizing columns $\{0,1,2\}$. Apply all at once:

$$\mathbf{W}_{:,3:5} \;\leftarrow\; \mathbf{W}_{:,3:5} \;-\; \underbrace{\mathbf{E}_{:,0:2}}_{\text{errors}} \underbrace{\bigl(\mathbf{H}^{-1}_{0\!:\!2,\,0\!:\!2}\bigr)^{-1} \mathbf{H}^{-1}_{0\!:\!2,\,3\!:\!5}}_{\text{propagation}}$$

where the error matrix is

$$\mathbf{E}_{:,0:2} = \begin{pmatrix} \delta_0^{(0)} & \delta_1^{(0)} & \delta_2^{(0)} \\ \delta_0^{(1)} & \delta_1^{(1)} & \delta_2^{(1)} \end{pmatrix}$$

After propagation:

<div style="overflow-x:auto; margin:1em 0;">
<table style="border-collapse:collapse; margin:0 auto; font-size:0.95em;">
<tbody>
<tr>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{00}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{01}$</td>
  <td style="border:1px solid #aaa; border-right:3px solid #555; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{02}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#bbdefb;">$w_{03}^{\star}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#bbdefb;">$w_{04}^{\star}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#bbdefb;">$w_{05}^{\star}$</td>
</tr>
<tr>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{10}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{11}$</td>
  <td style="border:1px solid #aaa; border-right:3px solid #555; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{12}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#bbdefb;">$w_{13}^{\star}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#bbdefb;">$w_{14}^{\star}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#bbdefb;">$w_{15}^{\star}$</td>
</tr>
</tbody>
</table>
<div style="text-align:center; margin-top:6px; font-size:0.85em;">
<span style="background:#bbdefb; padding:2px 8px; border-radius:3px;">Block 2: cols {3,4,5} — now corrected, ready to quantize</span>
</div>
</div>

Repeat the same sequential procedure on columns $\{3,4,5\}$. No lazy propagation needed (no remaining columns).

<div style="overflow-x:auto; margin:1em 0;">
<table style="border-collapse:collapse; margin:0 auto; font-size:0.95em;">
<tbody>
<tr>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{00}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{01}$</td>
  <td style="border:1px solid #aaa; border-right:3px solid #555; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{02}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{03}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{04}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{05}$</td>
</tr>
<tr>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{10}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{11}$</td>
  <td style="border:1px solid #aaa; border-right:3px solid #555; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{12}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{13}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{14}$</td>
  <td style="border:1px solid #aaa; padding:5px 14px; background:#e0e0e0; color:#777;">$\hat{w}_{15}$</td>
</tr>
</tbody>
</table>
</div>


### Basic implementation in PyTorch

Just to have some fun, I ended up implementing a version of this algorithm in PyTorch. Sadly, it took longer than expected due to clumsy sign errors. But I was able apply this algorithm to GPT2 available via Huggingface's tranformer package. The inputs were constructed from the [Salesforce wikidata dataset](https://huggingface.co/datasets/Salesforce/wikitext). The full notebook is available on [Google Colab](https://colab.research.google.com/drive/1GYosMSkNuJQO7_eHhOrJX2nADnYiInVw#scrollTo=evWVk2zrBwvM). Below is the core quantization function:

```python
def gptq_quantize_layer(W, X, n_bits=8, block_size=128, damp=0.01):
    W = W.clone().float()
    n_row, n_col = W.shape

    # dampen the Hessian diagonal before factoring, for numerical stability
    H = 2*(X@X.T)/X.shape[1]
    H += damp * torch.diag(H).mean() * torch.eye(n_col, device=H.device)

    # factor, invert, then take the upper-Cholesky factor for stable row reads
    L = torch.linalg.cholesky(H)
    H_inv = torch.cholesky_inverse(L)
    H_inv_chol = torch.linalg.cholesky(H_inv, upper=True)

    W_orig = W.clone()
    scale = W_orig.abs().amax(dim=1, keepdim=True) / (2**(n_bits - 1) - 1)
    scale = scale.clamp(min=1e-10)

    def quantize(w_col):
        qmax, qmin = 2**(n_bits - 1) - 1, -2**(n_bits - 1)
        level = torch.round(w_col.unsqueeze(1) / scale).clamp(qmin, qmax)
        return (level * scale).squeeze(1)

    Q = torch.zeros_like(W)

    for block_start in range(0, n_col, block_size):
        block_end = min(block_start + block_size, n_col)
        E = torch.zeros(n_row, block_end - block_start, device=W.device)

        for j in range(block_start, block_end):
            q = quantize(W[:, j])
            err = (W[:, j] - q) / H_inv_chol[j, j]
            Q[:, j] = q
            E[:, j - block_start] = err
            W[:, j+1:block_end] -= err.unsqueeze(1) * H_inv_chol[j, j+1:block_end]

        if block_end < n_col:
            W[:, block_end:] -= E @ H_inv_chol[block_start:block_end, block_end:]

    return Q
```

I was happy to see that GPT-2 small holds up well under 4-bit quantization on WikiText-2:
 
| Configuration | Perplexity | Δ vs. baseline |
|---|---|---|
| fp32 baseline | 39.10 | — |
| 4-bit, first 4 blocks | 40.04 | +0.94 |
| 4-bit, all 12 blocks | 42.88 | +3.78 |

 **A note on the baseline.** The fp32 perplexity here (39.1) is higher than the ~29–30 often quoted for GPT-2 small on WikiText-2, because of the sequence length and sampling scheme used in this evaluation. The absolute numbers are sensitive to those choices; the *gap* between quantized and baseline is the quantity that transfers across setups, so that's what to focus on.

 **Some results from the original paper:**
 The original GPTQ paper measured perplexity with respect to the [OPT benchmark](https://arxiv.org/abs/2205.01068) of models showing the impressive gains as shown from the figure below from the original paper demonstrating the efficacy of the method.  

 ![OPT Perplexity vs Bit Width](/OPT_Perplexity_GPTQ.png)

## Further work 
GPTQ kicked off a wave of post-training quantization methods — AWQ, which protects the most salient weight channels by scaling rather than treating all weights equally, and SmoothQuant, which shifts quantization difficulty from activations to weights, are two widely used successors which could be fun exploring from here.

## References

1. Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.](https://arxiv.org/abs/2210.17323) *arXiv:2210.17323.*

2. LeCun, Y., Denker, J., & Solla, S. (1989). [Optimal Brain Damage.](https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html) *NeurIPS 2.*

3. Hassibi, B., & Stork, D. (1993). [Second Order Derivatives for Network Pruning: Optimal Brain Surgeon.](https://proceedings.neurips.cc/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html) *NeurIPS 5.*

4. Hubara, I., Chmiel, B., Island, M., Banner, R., Naor, J., & Soudry, D. (2021). [Accelerated Sparse Neural Training: A Provable and Efficient Method to Find N:M Transposable Masks.](https://arxiv.org/abs/2102.08124) *NeurIPS 34.* (AdaPrune)

5. Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., ... & Zettlemoyer, L. (2022). [OPT: Open Pre-trained Transformer Language Models.](https://arxiv.org/abs/2205.01068) *arXiv:2205.01068.*

6. Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., & Han, S. (2023). [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.](https://arxiv.org/abs/2306.00978) *arXiv:2306.00978.*

7. Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., & Han, S. (2022). [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.](https://arxiv.org/abs/2211.10438) *arXiv:2211.10438.*

