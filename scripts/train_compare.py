"""Train and compare SWFA vs DeepLOB-like vs vanilla transformer on synthetic LOB.

Reports accuracy, macro-F1, effective memory horizons for SWFA heads.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import SyntheticLOBDataset, Config, LOBTransformer, DeepLOBLike, VanillaTransformerLOB


def run_epoch(model, loader, loss_fn, optim, device, train: bool):
    model.train(train)
    total_loss = 0.0
    correct = 0
    n = 0
    per_class_correct = np.zeros(3)
    per_class_total = np.zeros(3)

    for feats, t, y in loader:
        feats, t, y = feats.to(device), t.to(device), y.to(device)
        if train:
            optim.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(feats, t)
            loss = loss_fn(logits, y)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

        pred = logits.argmax(-1)
        correct += int((pred == y).sum())
        n += y.size(0)
        total_loss += float(loss) * y.size(0)
        for c in range(3):
            mask = y == c
            per_class_total[c] += int(mask.sum())
            per_class_correct[c] += int(((pred == y) & mask).sum())

    acc = correct / max(n, 1)
    recall = per_class_correct / np.maximum(per_class_total, 1)
    return {"loss": total_loss / max(n, 1), "acc": acc, "per_class_recall": recall}


def macro_f1_from_conf(model, loader, device) -> tuple[float, np.ndarray]:
    model.eval()
    conf = np.zeros((3, 3), dtype=int)
    with torch.no_grad():
        for feats, t, y in loader:
            feats, t, y = feats.to(device), t.to(device), y.to(device)
            pred = model(feats, t).argmax(-1)
            for yi, pi in zip(y.tolist(), pred.tolist()):
                conf[yi, pi] += 1
    f1s = []
    for c in range(3):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        f1s.append(f1)
    return float(np.mean(f1s)), conf


def train_model(name, model, train_loader, val_loader, device, epochs=6, lr=1e-3):
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n--- {name} ({n_params:,} params) ---")

    best_acc = 0.0
    for e in range(1, epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, loss_fn, optim, device, train=True)
        va = run_epoch(model, val_loader, loss_fn, optim, device, train=False)
        best_acc = max(best_acc, va["acc"])
        print(f"  epoch {e:02d}  train_acc={tr['acc']:.3f}  val_acc={va['acc']:.3f}  "
              f"val_loss={va['loss']:.3f}  [{time.time()-t0:.1f}s]")
    f1, conf = macro_f1_from_conf(model, val_loader, device)
    return {"name": name, "params": n_params, "best_val_acc": best_acc,
            "macro_f1": f1, "confusion": conf}


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = Config(seq_len=48)
    train_ds = SyntheticLOBDataset(n=512, cfg=cfg, seed=0)
    val_ds = SyntheticLOBDataset(n=128, cfg=cfg, seed=9999)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    results = []
    results.append(train_model(
        "DeepLOB-like", DeepLOBLike(num_features=cfg.num_features), train_loader, val_loader, device))
    results.append(train_model(
        "Vanilla Transformer", VanillaTransformerLOB(num_features=cfg.num_features),
        train_loader, val_loader, device))

    swfa = LOBTransformer(num_features=cfg.num_features, max_distance_s=2.0)
    results.append(train_model("SWFA LOB", swfa, train_loader, val_loader, device))

    print("\n=== Summary ===")
    print(f"  {'Model':<22}  {'params':>8}  {'best val acc':>12}  {'macro-F1':>10}")
    for r in results:
        print(f"  {r['name']:<22}  {r['params']:>8,}  {r['best_val_acc']:>11.3f}   {r['macro_f1']:>10.3f}")

    # Effective-memory-horizon analysis — the novel quantity.
    print("\n=== Effective memory horizon (SWFA) ===")
    for i, blk in enumerate(swfa.blocks):
        horizon = blk.attn.effective_memory_s().cpu().numpy()
        print(f"  layer {i}: per-head memory (s) = {np.array2string(horizon, precision=3)}")


if __name__ == "__main__":
    main()
