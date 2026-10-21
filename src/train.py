import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader

from config import CACHE_DIR, CHECKPOINT_DIR, CT_DIR, LABEL_DIR, SPLIT_DIR
from dataset import PanTSDataset
from model import CTFMClassifier


def parse_args():
    p = argparse.ArgumentParser(description="Train CT-FM Pancreas Classifier")
    p.add_argument("--data_dir", default=str(CT_DIR))
    p.add_argument("--label_dir", default=str(LABEL_DIR))
    p.add_argument("--cache_dir", default=str(CACHE_DIR))
    p.add_argument("--train_csv", default=str(SPLIT_DIR / "train_split.csv"))
    p.add_argument("--val_csv", default=str(SPLIT_DIR / "val_split.csv"))
    p.add_argument("--cohort_csv", default=str(SPLIT_DIR / "cohort_50.csv"))
    p.add_argument("--fold", type=int, default=None,
                   help="If set, use this fold of --cohort_csv as validation and the rest as training "
                        "(overrides --train_csv/--val_csv).")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for the reported metrics.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default=None, help="Suffix for output files (default: fold%%d or 'default').")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def metrics_from(targets, probs, threshold):
    preds = [1 if p >= threshold else 0 for p in probs]
    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(targets, probs)
    except ValueError:  # only one class present
        auc = float("nan")
    return {
        "acc": accuracy_score(targets, preds),
        "bal_acc": balanced_accuracy_score(targets, preds),
        "sens": recall_score(targets, preds, zero_division=0),
        "spec": tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
        "auc": auc,
        "tp": int(tp), "fn": int(fn), "tn": int(tn), "fp": int(fp),
    }


@torch.no_grad()
def predict(model, loader, device, criterion):
    model.eval()
    total_loss, probs, targets, cids = 0.0, [], [], []
    for imgs, labels, ids in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits, _ = model(imgs)
        total_loss += criterion(logits, labels).item() * len(labels)
        probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        targets.extend(labels.cpu().numpy())
        cids.extend(ids)
    return total_loss / len(loader.dataset), np.array(probs), np.array(targets), cids


def main():
    args = parse_args()
    set_seed(args.seed)
    tag = args.tag or (f"fold{args.fold}" if args.fold is not None else "default")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = CHECKPOINT_DIR / f"ctfm_head_{tag}.pth"
    report_path = SPLIT_DIR / f"val_predictions_{tag}.csv"
    history_path = CHECKPOINT_DIR / f"history_{tag}.csv"

    device = pick_device()
    print(f"Device: {device} | tag: {tag}")

    if args.fold is not None:
        cohort = pd.read_csv(args.cohort_csv)
        train_src, val_src = cohort[cohort["fold"] != args.fold], cohort[cohort["fold"] == args.fold]
    else:
        train_src, val_src = args.train_csv, args.val_csv

    common = dict(data_dir=args.data_dir, label_dir=args.label_dir, cache_dir=args.cache_dir)
    train_dataset = PanTSDataset(split_csv=train_src, augment=True, **common)
    val_dataset = PanTSDataset(split_csv=val_src, augment=False, **common)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = CTFMClassifier(num_classes=2, freeze_backbone=True).to(device)

    # Class weights derived from the training labels instead of a hardcoded 9.0
    n_pos = sum(s["label"] for s in train_dataset.samples)
    n_neg = len(train_dataset) - n_pos
    class_weights = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    print(f"Train pos/neg = {n_pos}/{n_neg} -> class weights {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_key, history = None, []
    for epoch in range(args.epochs):
        model.train()
        total_loss, tr_probs, tr_targets = 0.0, [], []
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            tr_probs.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
            tr_targets.extend(labels.cpu().numpy())
        scheduler.step()
        tr_loss = total_loss / len(train_dataset)
        tr_m = metrics_from(tr_targets, tr_probs, args.threshold)

        val_loss, val_probs, val_targets, _ = predict(model, val_loader, device, criterion)
        val_m = metrics_from(val_targets, val_probs, args.threshold)

        print(f"Epoch [{epoch+1:02d}/{args.epochs:02d}] Train Loss {tr_loss:.4f} BalAcc {tr_m['bal_acc']*100:.1f}% || "
              f"Val Loss {val_loss:.4f} Acc {val_m['acc']*100:.1f}% BalAcc {val_m['bal_acc']*100:.1f}% "
              f"Sens {val_m['sens']*100:.1f}% Spec {val_m['spec']*100:.1f}% AUC {val_m['auc']:.3f}")
        history.append({"epoch": epoch + 1, "train_loss": tr_loss, "val_loss": val_loss,
                        **{f"val_{k}": v for k, v in val_m.items()}})

        # Checkpoint selection: highest AUC, ties broken by lowest val loss.
        # NOTE: selecting on the validation fold makes its metrics optimistic; report CV across folds.
        auc = val_m["auc"] if not np.isnan(val_m["auc"]) else -1.0
        key = (auc, -val_loss)
        if best_key is None or key > best_key:
            best_key = key
            torch.save({"epoch": epoch + 1, "classifier_state_dict": model.head_state_dict(),
                        "val_metrics": history[-1], "args": vars(args)}, save_path)
            print(f"  new best checkpoint -> {save_path}")

    pd.DataFrame(history).to_csv(history_path, index=False)

    # Final report with the best checkpoint
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.classifier.load_state_dict(ckpt["classifier_state_dict"])
    _, probs, targets, cids = predict(model, val_loader, device, criterion)
    rows = [{"PanTS ID": c, "label": int(t), "tumor_probability": round(float(p), 4),
             "prediction": int(p >= args.threshold)} for c, t, p in zip(cids, targets, probs)]
    report = pd.DataFrame(rows).sort_values("tumor_probability", ascending=False)
    report.to_csv(report_path, index=False)
    final = metrics_from(targets, probs, args.threshold)
    print("\n" + report.to_string(index=False))
    print(f"\nBest epoch {ckpt['epoch']} @ threshold {args.threshold}: {final}")
    print(f"Saved report -> {report_path}")


if __name__ == "__main__":
    main()
