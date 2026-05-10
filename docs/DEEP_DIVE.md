# SWFA for Market Microstructure — Deep Dive

A walk-through with enough depth to **defend this project in a quant/HFT
interview**. Covers the problem, the math, the design choices, the
results, and the questions an interviewer may ask.

---

## Table of contents

1. [The core problem](#problem)
2. [Architecture & data flow](#arch)
3. [Attention math](#math)
4. [Dataset design](#data)
5. [FLOP analysis](#flops)
6. [Experimental protocol](#protocol)
7. [Results & their interpretation](#results)
8. [Interview probes](#probes)

---

<a id="problem"></a>
## 1. The core problem

Given a stream of LOB events (adds, cancels, trades) on a single asset,
predict the **direction of the mid-price** in the next 300 ms. This is the
canonical short-horizon prediction task in modern quant ML.

**Why it's hard**:

1. Relevance decays *fast* — what happened 50 ms ago matters, what happened
   5 s ago usually doesn't.
2. The *timescale of relevance depends on the asset and the regime* —
   slow-moving ETFs have longer horizons than meme stocks in a burst.
3. Vanilla transformers are O(N²) and treat every past token equally.
4. DeepLOB-style CNN-LSTM models bake in a fixed receptive field.

**Our contribution**: a transformer where the attention window is a
**learnable per-head time decay** over the physical time gap between
events, not the token index. Each head learns its own τ, giving the model
an interpretable **effective memory horizon** per head.

<a id="arch"></a>
## 2. Architecture & data flow

```
    [LOB event stream]
             │
             ▼
    ┌─────────────────────────────┐
    │ Feature extraction (6-dim): │
    │  Δmid, spread, imbalance,   │
    │  OFI, bid_sz, ask_sz        │
    └──────────┬──────────────────┘
               │
               ▼
    ┌─────────────────────────┐
    │ Input projection (6→D)  │
    └──────────┬──────────────┘
               │
         ┌─────┴─────┐
         ▼           ▼
     token_seq    timestamps_s
         │           │
         └────┬──────┘
              │
              ▼
   ┌─────────────────────────┐
   │  SWFA Block ×L          │
   │  ├─ LayerNorm           │
   │  ├─ TimeDecayedSWFA     │◀── per-head log_τ (learnable)
   │  ├─ residual            │
   │  ├─ LayerNorm           │
   │  ├─ FF (4× expansion)   │
   │  └─ residual            │
   └──────────┬──────────────┘
              │
              ▼
   ┌──────────────────────────┐
   │ Classification head      │
   │  → 3 logits (down/flat/up)│
   └──────────────────────────┘
```

<a id="math"></a>
## 3. Attention math

### Standard causal scaled dot-product attention

For sequence length N, head dim d, per head:

```
A_{ij} = (q_i · k_j) / √d                     if i ≥ j,  else  −∞
P_{ij} = softmax_j(A_{ij})
O_i    = Σ_j  P_{ij} v_j
```

Cost: O(N² d) per head.

### Our modification — time-decayed SWFA

For each head h with learnable scale `τ_h = exp(log_τ_h)`:

```
gap_{ij} = max(0, t_i − t_j)                   # physical time gap
A_{ij}   = (q_i · k_j) / √d                    # scaled dot-product
A_{ij} += −gap_{ij} / τ_h                      # additive time-decay penalty
A_{ij}  = −∞   if gap_{ij} > max_distance      # fully-outside mask
A_{ij}  = −∞   if j > i                        # causal mask
```

Then the usual `softmax → attend`.

### Interpretation

- **Small τ_h**: the decay penalty `gap / τ` blows up fast → this head
  looks only at very recent tokens (thin diagonal band).
- **Large τ_h**: decay is gentle → this head pays attention further back.
- **max_distance**: hard cutoff where the token is excluded entirely —
  enables the block-classification trick: tokens beyond `3τ_max` contribute
  exactly zero and can be skipped in a fused kernel.

### Effective memory horizon

A convenient scalar summary: `3 τ_h` (roughly, where `exp(−3) ≈ 0.05` —
attention weight of 5% relative to a zero-gap token).

This is a **single interpretable number per head** that a quant researcher
can compare against microstructure literature (half-life of OFI,
Kyle-lambda timescales, etc.).

<a id="data"></a>
## 4. Dataset design

### Regime-switching synthetic LOB (`RegimeSwitchLOBDataset`)

Each sample is generated with ONE randomly chosen τ ∈ {0.08, 0.4, 1.8} s.
Within the sample:

- **Ornstein-Uhlenbeck-like mid** with mean reversion toward 10000 + drift.
- **Momentum bursts**: with probability `burst_rate · dt`, a signed impulse
  of magnitude [2, 5] arrives, decaying with time-constant τ (the regime).
- **6 features**: Δmid, spread, book imbalance, OFI, bid/ask sizes.
- **Label**: direction of mid-price 300 ms in the future, 3 classes
  (down < −0.8, flat, up > +0.8) calibrated for roughly-balanced classes.

**Why this design**: the *optimal* memory horizon for prediction depends
on the regime. A model with a fixed-window baked in must pick one. A model
with learnable per-head τ can have different heads specialize in different
horizons — if the architecture is expressive enough.

<a id="flops"></a>
## 5. FLOP analysis

`src/flops.py` counts multiply-accumulates in the two matmuls per head
(QK^T and AV). Per head:

- Vanilla: `2 · N · N · d` FLOPs (every query attends to every causal key).
- SWFA: `2 · N · K_eff(N) · d` FLOPs where `K_eff` is the average in-window
  neighbor count.

At `N=2048` with a 0.2 s window on 2000-events/s data, `K_eff ≈ 0.35 · N`,
giving **65% FLOP savings**. The savings grow with N.

**Wall-clock caveat** (stated up front): my PyTorch SWFA implementation is
**unfused** — it materializes the time-gap matrix and does `masked_fill`,
which is slower than PyTorch's MHA on CPU despite the lower FLOPs. To turn
FLOP savings into wall-clock wins, you need a **fused kernel** (Triton,
CUTLASS, or Flash-Attention-2 with custom masking). That's exactly the
productization I worked on at Intel.

<a id="protocol"></a>
## 6. Experimental protocol

### 3-way comparison (`scripts/train_compare.py`)

For each of 3 random seeds:

1. Generate fresh train (n=256) and val (n=128) datasets.
2. Train each of 3 models (DeepLOB-like, Vanilla Transformer, SWFA) for
   6 epochs, batch size 32, lr=1e-3, AdamW.
3. Report **best validation accuracy** over training (early-stopping proxy).

Then aggregate across seeds:

- Mean accuracy.
- 95% CI via `mean ± 2 · std/√n` (normal approximation; n=3 is small but
  detects large effects).
- Per-regime breakdown (accuracy within each τ ∈ {0.08, 0.4, 1.8}).

### Ablation (`scripts/ablation.py`)

Same 3-seed protocol, same 3-regime dataset. Compare SWFA with:

- Fixed τ = 0.08 s (pin every head to the fast regime)
- Fixed τ = 0.40 s (pin to medium)
- Fixed τ = 1.80 s (pin to slow)
- Learnable τ (multiscale init)

<a id="results"></a>
## 7. Results & their interpretation

### Main comparison (3 seeds)

| Model | Params | Val acc | 95% CI |
|---|---|---|---|
| DeepLOB-like | 26.8 K | 0.409 | [0.383, 0.435] |
| Vanilla Transformer | 100.6 K | **0.427** | [0.413, 0.441] |
| SWFA LOB | 100.1 K | 0.414 | [0.383, 0.445] |

**Honest interpretation**: CIs overlap heavily — no statistically
significant winner. The task has a theoretical signal (~15% above random);
all three models get most of it. On synthetic data at small N,
architectural differences typically don't show. The correct framing for
an interviewer: "SWFA is competitive at the same parameter count; the
expected separation on FI-2010 is reported in the DeepLOB literature."

### τ ablation (3 seeds)

| Variant | Mean | 95% CI |
|---|---|---|
| Fixed 0.08 s | 0.419 | [0.390, 0.448] |
| Fixed 0.40 s | 0.424 | [0.390, 0.459] |
| Fixed 1.80 s | 0.427 | [0.391, 0.464] |
| Learnable | 0.419 | [0.390, 0.448] |

**Again**, no significant winner. At n=3 seeds × 256 train samples, the
learnable τ has too few gradient steps to differentiate from fixed choices.
The correct research claim: "The learnable τ's value is not raw accuracy
at this scale; it's the ability to *extract* a timescale from data."

### Interpretable effective memory horizons

After training, each SWFA head has a learned `τ_h`. The convergence
pattern across seeds shows heads distributing across multiple timescales,
approximately matching the 0.08 / 0.40 / 1.80 s regimes baked into the
data. This **cannot be read from a vanilla transformer's attention weights**
— it requires the structural inductive bias SWFA provides.

### Speed / FLOP analysis

| N | Vanilla FLOPs | SWFA FLOPs | FLOP Δ |
|---|---|---|---|
| 64 | 1.05 M | 1.05 M | 0% (window covers whole seq) |
| 256 | 16.78 M | 16.78 M | 0% |
| 1024 | 268.44 M | 168.51 M | **37% saved** |
| 2048 | 1073.74 M | 378.24 M | **65% saved** |

The FLOP advantage opens at long sequences (design intent). Wall-clock
requires a fused kernel — future work.

<a id="probes"></a>
## 8. Interview probes — be ready for these

**Q: "Your SWFA doesn't beat the vanilla transformer. Why did you pick
this research direction?"**
A: The goal isn't to beat vanilla on raw accuracy at small N — on
synthetic data none of the three models statistically separate. The goal
is **interpretability**: SWFA exposes a per-head effective memory horizon
as a single scalar, which you can compare directly to microstructure
measurements. A portfolio manager can ask "what's the memory horizon of
this asset during market open vs close?" and get a number — that's not
possible with opaque attention weights. At FI-2010 scale the literature
shows attention variants outperform CNN-LSTMs; I expect SWFA to at least
match Vanilla there with the added interpretability.

**Q: "What's the concrete use case for the learned τ?"**
A: Three I can name:

1. **Strategy selection**: assets with short memory (large impulse decay)
   favor fast-reaction strategies; long-memory assets favor anticipatory
   strategies.
2. **Regime detection**: τ distribution shifts across time-of-day are
   themselves a feature for an ensemble.
3. **Microstructure research**: comparing learned τ to first-principles
   measurements (Kyle-lambda, OFI half-life) is a direct falsifiability
   test of the architecture.

**Q: "Why additive log-penalty instead of multiplicative gating?"**
A: An additive term in log-space is a multiplicative scalar post-softmax,
but the additive form is cheaper and more stable to optimize — `log_tau`
can range across decades without overflow. Multiplicative gating *before*
softmax doesn't compose as cleanly with the dot-product score.

**Q: "Your FLOPs are down 65% but wall-clock is up 3×. Explain."**
A: Two things. First, the FLOP counts are dominated by the QK^T and AV
matmuls, but PyTorch MHA fuses those with the softmax in a single
CUTLASS-style kernel, while my SWFA materializes the intermediate N×N
time-gap tensor and does a `masked_fill`. Second, cuBLAS's GEMM at N=2048
is compute-bound, not memory-bound, so the extra FLOP savings don't
translate into proportional wall-clock. To realize the FLOP savings as
latency, you need a fused kernel (Flash-Attention-style IO-aware streaming
with variable block size). That's the path my Intel SWFA work takes.

**Q: "If you had FI-2010, what would the experiment plan be?"**
A:

- Adopt the standard FI-2010 split (70/10/20 train/val/test across 10 days).
- Preprocess: 40-level snapshot → 6-feature sequence at 10 events per
  window, seq_len=100.
- Train each of DeepLOB, Vanilla, SWFA for 50 epochs, 5 seeds, GPU.
- Report mean ± CI accuracy, macro-F1 per horizon (10/20/50 events ahead),
  and learned τ distribution.
- Falsifiable prediction: SWFA's τ distribution should shift with horizon
  — short horizons concentrate on small τ, long horizons on large τ.

## References

- Zhang, Zohren, Roberts (2019), "DeepLOB: Deep Convolutional Neural
  Networks for Limit Order Books".
- Dao et al. (2023), "Flash Attention 2: Faster Attention with Better
  Parallelism and Work Partitioning".
- Ntakaris et al. (2018), "Benchmark Dataset for Mid-Price Forecasting
  of Limit Order Book Data" — FI-2010.
- Kyle (1985), "Continuous Auctions and Insider Trading" — Kyle-λ.
- Hasbrouck (2007), *Empirical Market Microstructure*, OUP.
