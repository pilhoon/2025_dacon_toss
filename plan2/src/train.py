from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold

from dataset import (
    DataConfig,
    CTRDatasetLazy,
    build_vocabs,
    prepare_data,
    make_collate_fn,
    compute_num_stats,
)
from metrics import compute_all
from utils import ensure_dir, load_yaml, save_json, set_seed, load_json
from modules.dcnv2 import DCNv2
from modules.xdeepfm import XDeepFM
from modules.ft_transformer import FTTransformer
from modules.din import DIN


def build_model(model_cfg: dict, cat_cardinalities: Dict[str, int], num_dim: int):
    t = model_cfg["type"].lower()
    if t == "dcnv2":
        return DCNv2(cat_cardinalities, num_dim, embed_dim=model_cfg.get("embed_dim", 32), cross_depth=model_cfg.get("cross_depth", 3), mlp_dims=model_cfg.get("mlp_dims"), dropout=model_cfg.get("dropout", 0.0))
    if t == "xdeepfm":
        return XDeepFM(cat_cardinalities, num_dim, embed_dim=model_cfg.get("embed_dim", 32), cin_layers=model_cfg.get("cin_layers"), dnn_layers=model_cfg.get("dnn_layers"), dropout=model_cfg.get("dropout", 0.0))
    if t == "ft_transformer":
        return FTTransformer(cat_cardinalities, num_dim, embed_dim=model_cfg.get("embed_dim", 64), n_layers=model_cfg.get("n_layers", 4), n_heads=model_cfg.get("n_heads", 8), ff_mult=model_cfg.get("ff_mult", 4), dropout=model_cfg.get("dropout", 0.1), stochastic_depth=model_cfg.get("stochastic_depth", 0.0))
    if t == "din":
        return DIN(cat_cardinalities, num_dim, target_item_col=model_cfg.get("target_item_col", "l_feat_14"), embed_dim=model_cfg.get("embed_dim", 32), attn_hidden=model_cfg.get("attn_hidden", 64), mlp_dims=model_cfg.get("mlp_dims"), dropout=model_cfg.get("dropout", 0.1))
    raise ValueError(f"Unknown model type: {t}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--out", default="plan2/experiments/tmp")
    parser.add_argument("--n-rows", type=int, default=None, help="Optional: limit number of rows for train/test load")
    parser.add_argument("--epochs", type=int, default=None, help="Optional: override epochs")
    parser.add_argument("--folds", type=int, default=None, help="Optional: override folds")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional: override batch size")
    parser.add_argument("--safe-mode", action="store_true", help="Use single-process DataLoader for sandboxed envs")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="Override DataLoader prefetch_factor (needs workers > 0)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for the model if available")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 matmul on CUDA (Ampere+) for speed")
    parser.add_argument("--precompute", action="store_true", help="Precompute encoded arrays on CPU to reduce collate overhead")
    args = parser.parse_args()

    print("[INIT] Loading configs...", flush=True)
    cfg = load_yaml(args.config)
    trn = load_yaml(args.trainer)["trainer"]

    # Apply optional overrides
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(args.epochs)
    if args.folds is not None:
        cfg.setdefault("train", {})["folds"] = int(args.folds)
    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = int(args.batch_size)

    set_seed(cfg.get("train", {}).get("seed", 42))
    ensure_dir(args.out)
    out_dir = Path(args.out)

    data_cfg = DataConfig(
        train_path=cfg["data"]["train_path"],
        test_path=cfg["data"]["test_path"],
        target=cfg["data"]["target"],
        cat_patterns=cfg["data"]["cat_patterns"],
        num_patterns=cfg["data"]["num_patterns"],
        min_freq=cfg["data"].get("min_freq", 10),
        max_seq_len=cfg["data"].get("max_seq_len", 0),
    )
    t0 = time.time()
    print(f"[DATA] Preparing data from {data_cfg.train_path} (this may take a while)...", flush=True)
    train_df, test_df, cat_cols, num_cols = prepare_data(data_cfg, n_rows=args.n_rows)
    if args.n_rows:
        print(f"[DATA] Loaded subset n_rows={args.n_rows}", flush=True)
    print(f"[DATA] Train shape: {train_df.shape}, Test shape: {test_df.shape}", flush=True)
    print(f"[DATA] Categorical cols: {len(cat_cols)}, Numerical cols: {len(num_cols)}", flush=True)

    # Cache vocabs
    voc_path = out_dir / "vocabs.json"
    if voc_path.exists():
        vocabs = load_json(voc_path)
        print(f"[VOCAB] Loaded cached vocabs from {voc_path}", flush=True)
    else:
        print("[VOCAB] Building categorical vocabularies...", flush=True)
        vocabs = build_vocabs(train_df, cat_cols, data_cfg.min_freq)
        save_json({k: len(v) for k, v in vocabs.items()}, out_dir / "vocab_sizes.json")
        save_json(vocabs, voc_path)
        print(f"[VOCAB] Saved {voc_path}", flush=True)

    # Cache numeric stats
    stats_path = out_dir / "num_stats.json"
    if stats_path.exists():
        num_stats = load_json(stats_path)
        print(f"[NUM ] Loaded cached stats from {stats_path}", flush=True)
    else:
        num_stats = compute_num_stats(train_df, num_cols)
        save_json(num_stats, stats_path)
        print(f"[NUM ] Saved {stats_path}", flush=True)

    y = train_df[data_cfg.target].to_numpy().astype(np.float32)
    print(f"[PIPE] Lazy encoding with DataLoader workers; caching vocabs/stats used.", flush=True)

    cat_cardinalities = {k: len(v) for k, v in vocabs.items()}
    num_dim = len(num_cols)

    print(f"[MODEL] Building model: {cfg['model']['type']}", flush=True)
    model = build_model(cfg["model"], cat_cardinalities, num_dim)
    req_device = str(trn.get("device", "cuda"))
    if req_device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available; falling back to CPU", flush=True)
        req_device = "cpu"
    device = torch.device(req_device)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Number of parameters: {n_params:,}", flush=True)

    if args.tf32 and req_device == "cuda":
        try:
            torch.set_float32_matmul_precision("high")  # enables TF32 on Ampere+
            print("[CUDA] TF32 enabled for matmul", flush=True)
        except Exception:
            pass

    if str(trn.get("amp", "")).lower() == "bf16":
        amp_dtype = torch.bfloat16
    elif str(trn.get("amp", "")).lower() in ("1", "true", "fp16", "mixed"):
        amp_dtype = torch.float16
    else:
        amp_dtype = None
    print(f"[TRAIN] Device={device.type} | AMP={str(amp_dtype)} | batch_size={cfg['train']['batch_size']} | epochs={cfg['train']['epochs']}", flush=True)

    ds = CTRDatasetLazy(size=len(train_df))
    collate = make_collate_fn(train_df.drop(columns=[data_cfg.target]), y, cat_cols, num_cols, vocabs, num_stats)

    # Optional precompute to reduce CPU overhead in collate
    precomputed = None
    if args.precompute:
        print("[PRE] Precomputing full encoded arrays on CPU...", flush=True)
        X_df = train_df.drop(columns=[data_cfg.target])
        enc_cat = {}
        for c in cat_cols:
            vocab = vocabs[c]
            enc_cat[c] = X_df[c].astype(str).map(lambda x: vocab.get(x, 1)).to_numpy(dtype=np.int64)
        from dataset import apply_num_stats  # relative import
        enc_num = apply_num_stats(X_df, num_cols, num_stats)
        precomputed = {"cat": enc_cat, "num": enc_num}
        def collate_pre(indices):
            cat_batch = {k: torch.from_numpy(v[indices]) for k, v in precomputed["cat"].items()}
            num_batch = torch.from_numpy(precomputed["num"][indices])
            out = {"cat": cat_batch, "num": num_batch}
            if y is not None:
                out["y"] = torch.as_tensor(y[indices], dtype=torch.float32)
            return out
        collate = collate_pre

    # Loss config (shared)
    opt_cfg = {"lr": cfg["train"]["lr"], "wd": cfg["train"].get("weight_decay", 0.0)}
    loss_cfg = trn.get("loss", {"type": "bce", "pos_weight": "auto", "focal_gamma": 2.0})
    pos_weight_tensor = None
    if loss_cfg.get("pos_weight", "auto") == "auto":
        pos = float(np.sum(y == 1))
        neg = float(np.sum(y == 0))
        w = min(20.0, max(1.0, neg / max(1.0, pos)))  # Cap at 20 for stability
        pos_weight_tensor = torch.tensor([w], device=device)
        print(f"[LOSS] Using BCE pos_weight auto={w:.2f} (capped at 20)", flush=True)
    elif isinstance(loss_cfg.get("pos_weight"), (int, float)):
        w = float(loss_cfg["pos_weight"])
        pos_weight_tensor = torch.tensor([w], device=device)
        print(f"[LOSS] Using BCE pos_weight={w:.2f}", flush=True)
    loss_type = str(loss_cfg.get("type", "bce")).lower()
    if loss_type == "bce":
        def make_loss():
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        use_focal = False
        focal_gamma = None
    else:
        def make_loss():
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        focal_gamma = float(loss_cfg.get("focal_gamma", 2.0))
        use_focal = True
        print(f"[LOSS] Using focal loss gamma={focal_gamma}", flush=True)

    folds = int(cfg["train"].get("folds", 1) or 1)
    if folds <= 1:
        # Single-run training (original behavior)
        num_workers = 0 if args.safe_mode else (args.num_workers if args.num_workers is not None else trn.get("num_workers", 8))
        prefetch_factor = None
        if num_workers and num_workers > 0:
            prefetch_factor = args.prefetch_factor if args.prefetch_factor is not None else 4
        dl = DataLoader(
            ds,
            batch_size=cfg["train"]["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(not args.safe_mode),
            persistent_workers=True if (not args.safe_mode and trn.get("num_workers", 0) > 0) else False,
            collate_fn=collate,
            prefetch_factor=prefetch_factor,
        )

        opt = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["wd"])
        steps = 0
        best_metric = -1.0
        use_scaler = (amp_dtype is not None and amp_dtype == torch.float16 and device.type == "cuda")
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

        # Early GPU sanity forward to fail fast
        print("[SANITY] Running early GPU forward pass...", flush=True)
        idx = np.arange(min(2048, len(y)))
        batch_eval = collate(idx.tolist())
        for k in batch_eval["cat"]:
            batch_eval["cat"][k] = batch_eval["cat"][k].to(device)
        batch_eval["num"] = batch_eval["num"].to(device)
        with torch.no_grad():
            if amp_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    _ = model(batch_eval)
            else:
                _ = model(batch_eval)
        print("[SANITY] OK. Starting full training.", flush=True)

        print(f"[READY] Data prepared in {time.time()-t0:.1f}s. Starting training...", flush=True)
        for epoch in range(cfg["train"]["epochs"]):
            model.train()
            print(f"[EPOCH {epoch+1}] START", flush=True)
            epoch_start = time.time()
            running_loss = 0.0
            seen = 0
            for bi, batch in enumerate(dl, start=1):
                steps += 1
                for k in batch["cat"]:
                    batch["cat"][k] = batch["cat"][k].to(device)
                batch["num"] = batch["num"].to(device)
                yb = batch["y"].to(device)
                opt.zero_grad(set_to_none=True)
                if amp_dtype is not None:
                    with torch.autocast(device_type=device.type, dtype=amp_dtype):
                        logit = model(batch)
                        loss_raw = make_loss()(logit, yb)
                        if use_focal:
                            p = torch.sigmoid(logit).detach()
                            pt = p * yb + (1 - p) * (1 - yb)
                            loss = ((1 - pt) ** focal_gamma) * loss_raw
                            loss = loss.mean()
                        else:
                            loss = loss_raw
                    if device.type == "cuda" and use_scaler:
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        # GradScaler is disabled on CPU; do normal backward
                        loss.backward()
                        # Gradient clipping
                        grad_clip = cfg.get("train", {}).get("grad_clip", None)
                        if grad_clip:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        opt.step()
                else:
                    logit = model(batch)
                    loss_raw = make_loss()(logit, yb)
                    if use_focal:
                        p = torch.sigmoid(logit).detach()
                        pt = p * yb + (1 - p) * (1 - yb)
                        loss = ((1 - pt) ** focal_gamma) * loss_raw
                        loss = loss.mean()
                    else:
                        loss = loss_raw
                    loss.backward()
                    # Gradient clipping
                    grad_clip = cfg.get("train", {}).get("grad_clip", None)
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()
                bs = yb.size(0)
                seen += bs
                running_loss += loss.item() * bs
                if bi % 50 == 0:
                    elapsed = time.time() - epoch_start
                    speed = seen / max(1e-6, elapsed)
                    avg_loss = running_loss / max(1, seen)
                    print(f"[EPOCH {epoch+1}] step {bi} | seen {seen} | avg_loss {avg_loss:.5f} | {speed:.0f} samples/s", flush=True)

            # quick eval each epoch on a held-out slice
            model.eval()
            with torch.no_grad():
                idx = np.random.choice(len(y), size=min(200000, len(y)), replace=False)
                batch_eval = collate(idx.tolist())
                for k in batch_eval["cat"]:
                    batch_eval["cat"][k] = batch_eval["cat"][k].to(device)
                batch_eval["num"] = batch_eval["num"].to(device)
                logit = model(batch_eval)
                prob = torch.sigmoid(logit).detach().float().cpu().numpy()
                metrics = compute_all(y[idx], prob)
                # distribution monitoring
                pred_mean = float(np.mean(prob))
                pred_std = float(np.std(prob))
                metrics_out = {"epoch": epoch, **metrics, "pred_mean": pred_mean, "pred_std": pred_std}
                save_json(metrics_out, Path(args.out) / "epoch_metrics.json")
                print(f"[EPOCH {epoch+1}] EVAL | AP {metrics.get('ap'):.6f} | WLL {metrics.get('wll'):.6f} | ROC {metrics.get('roc_auc'):.6f} | composite {metrics.get('composite'):.6f}", flush=True)
                print(f"[EPOCH {epoch+1}] DIST | mean {pred_mean:.6f} | std {pred_std:.6f}", flush=True)
                if metrics.get("composite", -1) > best_metric:
                    best_metric = metrics["composite"]
                    ckpt_path = Path(args.out) / "best.pt"
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"[EPOCH {epoch+1}] Saved new best checkpoint to {ckpt_path}", flush=True)
            print(f"[EPOCH {epoch+1}] END | duration {time.time()-epoch_start:.1f}s", flush=True)

        # final train-set eval snapshot
        model.eval()
        with torch.no_grad():
            probs = np.zeros(len(y), dtype=np.float32)
            for i in range(0, len(y), 262144):
                sl_idx = list(range(i, min(i + 262144, len(y))))
                batch_eval = collate(sl_idx)
                for k in batch_eval["cat"]:
                    batch_eval["cat"][k] = batch_eval["cat"][k].to(device)
                batch_eval["num"] = batch_eval["num"].to(device)
                logit = model(batch_eval)
                probs[i:i + len(sl_idx)] = torch.sigmoid(logit).float().cpu().numpy()
        metrics = compute_all(y, probs)
        save_json(metrics, Path(args.out) / "final_metrics.json")
        np.save(Path(args.out) / "train_probs.npy", probs)
        np.save(Path(args.out) / "labels.npy", y.astype(np.int8))
        print(f"[FINAL] Train snapshot | AP {metrics.get('ap'):.6f} | WLL {metrics.get('wll'):.6f} | ROC {metrics.get('roc_auc'):.6f} | composite {metrics.get('composite'):.6f}", flush=True)
    else:
        # K-fold OOF training
        print(f"[KFOLD] Starting {folds}-fold OOF training", flush=True)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(cfg.get("train", {}).get("seed", 42)))
        oof_probs = np.zeros(len(y), dtype=np.float32)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            fold_dir = Path(args.out) / f"fold{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            print(f"[FOLD {fold}] n_train={len(tr_idx)} n_val={len(va_idx)}", flush=True)

            # New model per fold
            model = build_model(cfg["model"], cat_cardinalities, num_dim).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["wd"])
            use_scaler = (amp_dtype is not None and amp_dtype == torch.float16 and device.type == "cuda")
            scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
            bce = make_loss()
            best_metric = -1.0

            num_workers = 0 if args.safe_mode else (args.num_workers if args.num_workers is not None else trn.get("num_workers", 8))
            prefetch_factor = None
            if num_workers and num_workers > 0:
                prefetch_factor = args.prefetch_factor if args.prefetch_factor is not None else 4
            dl = DataLoader(
                ds,
                batch_size=cfg["train"]["batch_size"],
                sampler=SubsetRandomSampler(tr_idx.tolist()),
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(not args.safe_mode),
                persistent_workers=True if (not args.safe_mode and trn.get("num_workers", 0) > 0) else False,
                collate_fn=collate,
                prefetch_factor=prefetch_factor,
            )

            for epoch in range(cfg["train"]["epochs"]):
                model.train()
                epoch_start = time.time()
                running_loss = 0.0
                seen = 0
                for bi, batch in enumerate(dl, start=1):
                    for k in batch["cat"]:
                        batch["cat"][k] = batch["cat"][k].to(device)
                    batch["num"] = batch["num"].to(device)
                    yb = batch["y"].to(device)
                    opt.zero_grad(set_to_none=True)
                    if amp_dtype is not None:
                        with torch.autocast(device_type=device.type, dtype=amp_dtype):
                            logit = model(batch)
                            loss_raw = bce(logit, yb)
                            if use_focal:
                                p = torch.sigmoid(logit).detach()
                                pt = p * yb + (1 - p) * (1 - yb)
                                loss = ((1 - pt) ** focal_gamma) * loss_raw
                                loss = loss.mean()
                            else:
                                loss = loss_raw
                        if device.type == "cuda":
                            scaler.scale(loss).backward()
                            scaler.step(opt)
                            scaler.update()
                        else:
                            loss.backward()
                            opt.step()
                    else:
                        logit = model(batch)
                        loss_raw = bce(logit, yb)
                        if use_focal:
                            p = torch.sigmoid(logit).detach()
                            pt = p * yb + (1 - p) * (1 - yb)
                            loss = ((1 - pt) ** focal_gamma) * loss_raw
                            loss = loss.mean()
                        else:
                            loss = loss_raw
                        loss.backward()
                        # Gradient clipping
                        grad_clip = cfg.get("train", {}).get("grad_clip", None)
                        if grad_clip:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        opt.step()
                    bs = yb.size(0)
                    seen += bs
                    running_loss += loss.item() * bs
                print(f"[FOLD {fold}] EPOCH {epoch+1} | seen {seen} | avg_loss {running_loss/max(1,seen):.5f} | {time.time()-epoch_start:.1f}s", flush=True)

                # eval on validation fold
                model.eval()
                with torch.no_grad():
                    batch_eval = collate(va_idx.tolist())
                    for k in batch_eval["cat"]:
                        batch_eval["cat"][k] = batch_eval["cat"][k].to(device)
                    batch_eval["num"] = batch_eval["num"].to(device)
                    logit = model(batch_eval)
                    prob = torch.sigmoid(logit).detach().float().cpu().numpy()
                    metrics = compute_all(y[va_idx], prob)
                    pred_mean = float(np.mean(prob))
                    pred_std = float(np.std(prob))
                    metrics_out = {"fold": fold, "epoch": epoch, **metrics, "pred_mean": pred_mean, "pred_std": pred_std}
                    save_json(metrics_out, fold_dir / "epoch_metrics.json")
                    print(f"[FOLD {fold}] EVAL | AP {metrics.get('ap'):.6f} | WLL {metrics.get('wll'):.6f} | ROC {metrics.get('roc_auc'):.6f} | composite {metrics.get('composite'):.6f}", flush=True)
                    if metrics.get("composite", -1) > best_metric:
                        best_metric = metrics["composite"]
                        ckpt_path = fold_dir / "best.pt"
                        torch.save(model.state_dict(), ckpt_path)
                        print(f"[FOLD {fold}] Saved new best checkpoint to {ckpt_path}", flush=True)

            # load best and generate final val probs
            best_ckpt = fold_dir / "best.pt"
            if best_ckpt.exists():
                model.load_state_dict(torch.load(best_ckpt, map_location=device))
            else:
                print(f"[FOLD {fold}] Warning: No best checkpoint found, using last model state", flush=True)
            model.eval()
            with torch.no_grad():
                batch_eval = collate(va_idx.tolist())
                for k in batch_eval["cat"]:
                    batch_eval["cat"][k] = batch_eval["cat"][k].to(device)
                batch_eval["num"] = batch_eval["num"].to(device)
                logit = model(batch_eval)
                prob = torch.sigmoid(logit).detach().float().cpu().numpy()
                oof_probs[va_idx] = prob

        # OOF metrics
        oof_metrics = compute_all(y, oof_probs)
        save_json(oof_metrics, Path(args.out) / "oof_metrics.json")
        # also save OOF predictions
        np.save(Path(args.out) / "oof_probs.npy", oof_probs)
        np.save(Path(args.out) / "labels.npy", y.astype(np.int8))
        print(f"[OOF] AP {oof_metrics.get('ap'):.6f} | WLL {oof_metrics.get('wll'):.6f} | ROC {oof_metrics.get('roc_auc'):.6f} | composite {oof_metrics.get('composite'):.6f}", flush=True)


if __name__ == "__main__":
    main()
