import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc

def plot_roc_pr_curves(metrics, out_dir, suffix=""):
    names = ['disorder', 'protein', 'rna', 'dna']
    roc_raw = metrics['roc_raw']

    best_f1_per_task = []

    for t_idx, name in enumerate(names):
        y_true = roc_raw['y_true'][t_idx]
        y_prob = roc_raw['y_prob'][t_idx]

        if len(y_true) == 0 or len(np.unique(y_true)) == 1:
            print(f"[WARN] Cannot plot ROC/PR for task {name} (no positive or negative examples)")
            best_f1_per_task.append(float('nan'))
            continue

        P, R, TH = precision_recall_curve(y_true, y_prob)

        if len(TH) > 0:
            F_score = 2 * P[1:] * R[1:] / (P[1:] + R[1:] + 1e-12)
            best_idx = np.argmax(F_score)
            best_f1 = F_score[best_idx]
            best_th = TH[best_idx]
            best_p = P[best_idx + 1]
            best_r = R[best_idx + 1]
        else:
            best_f1 = best_th = best_p = best_r = float('nan')

        pr_auc_val = auc(R, P)
        roc_auc_val = metrics['auc'][t_idx] if t_idx < len(metrics['auc']) else float('nan')

        plt.figure(figsize=(5, 5))
        plt.plot(
            R, P,
            label=f"AUC={roc_auc_val:.4f}, AUPR={pr_auc_val:.4f}, F1={best_f1:.4f}, Rec={best_r:.4f}"
        )

        if not np.isnan(best_f1):
            plt.scatter(best_r, best_p, color='red', s=40, label=f"Best-F1@th={best_th:.3f}")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve - {name}")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/pr_{name}{suffix}.png", dpi=200)
        plt.close()

        print(f"[PLOT] Saved PR curve for '{name}{suffix}' (Best F1={best_f1:.4f})")
        best_f1_per_task.append(best_f1)

    return best_f1_per_task

def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, pos_weight=None) -> torch.Tensor:
    logits = logits[mask]
    targets = targets[mask]
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

def focal_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, gamma: float = 2.0, alpha: float = 0.75) -> torch.Tensor:
    logits = logits[mask]
    targets = targets[mask]
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    pt = targets * p + (1 - targets) * (1 - p)
    alpha_added = targets * alpha + (1 - targets) * (1 - alpha)

    focal = (alpha_added * (1 - pt) ** gamma) * bce
    return focal.mean()

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs[mask]
    targets = targets[mask]
    inter = (probs * targets).sum()
    denom = probs.sum() + targets.sum() + eps
    dice = 1 - (2 * inter + eps) / denom
    return dice

@torch.no_grad()
def evaluate(model, loader, cfg):
    model.eval()
    total_loss = 0.0
    nsteps = 0

    names = ['disorder', 'protein', 'rna', 'dna']
    n_tasks = len(names)

    tp = torch.zeros(4, device=cfg.device)
    tn = torch.zeros(4, device=cfg.device)
    fp = torch.zeros(4, device=cfg.device)
    fn = torch.zeros(4, device=cfg.device)

    y_true_all = [[] for _ in range(n_tasks)]
    y_prob_all = [[] for _ in range(n_tasks)]

    for batch in loader:
        H = batch['emb'].to(cfg.device)
        Y = batch['label'].to(cfg.device)
        M = batch['mask'].to(cfg.device)

        logits = model(H, M)
        loss_sum = 0.0

        for j, name in enumerate(names):
            y = Y[:, :, j]
            valid = M & (y > -50)

            l = masked_bce_with_logits(logits[name], y, valid)
            if cfg.use_focal:
                l = l + focal_bce_with_logits(logits[name], y, valid, alpha=cfg.alphas[name])
            if cfg.use_dice:
                l = l + dice_loss(logits[name], y, valid)
            loss_sum = loss_sum + l

            if valid.any():
                prob = torch.sigmoid(logits[name])
                thr = cfg.thresholds.get(name, 0.5)
                pred = (prob >= thr).float()[valid]
                tgt = y[valid].float()

                tp[j] += (pred * tgt).sum()
                fp[j] += (pred * (1 - tgt)).sum()
                fn[j] += ((1 - pred) * tgt).sum()

                total_valid = valid.sum()
                tn[j] += total_valid - (pred * tgt).sum() - (pred * (1 - tgt)).sum() - ((1 - pred) * tgt).sum()

                y_true_all[j].append(tgt.detach().cpu().numpy())
                y_prob_all[j].append(prob[valid].detach().cpu().numpy())

        total_loss += loss_sum.item()
        nsteps += 1

    test_loss = total_loss / max(nsteps, 1)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    auc_list = []
    best_thresh_youden = []
    best_thresh_fbeta = []

    for j in range(n_tasks):
        if len(y_true_all[j]) == 0:
            auc_list.append(float('nan'))
            best_thresh_youden.append(float('nan'))
            best_thresh_fbeta.append(float('nan'))
            continue

        y_true = np.concatenate(y_true_all[j], axis=0)
        y_prob = np.concatenate(y_prob_all[j], axis=0)

        if y_true.max() == y_true.min():
            auc_list.append(float('nan'))
            best_thresh_youden.append(float('nan'))
            best_thresh_fbeta.append(float('nan'))
        else:
            try:
                auc_list.append(float(roc_auc_score(y_true, y_prob)))

                fpr_t, tpr_t, thr_t = roc_curve(y_true, y_prob)
                j_scores_t = tpr_t - fpr_t
                best_thresh_youden.append(float(thr_t[np.argmax(j_scores_t)]))
            except Exception:
                auc_list.append(float('nan'))
                best_thresh_youden.append(float('nan'))

            pos_cnt = int(y_true.sum())
            if pos_cnt == 0 or pos_cnt == y_true.size:
                best_thresh_fbeta.append(float('nan'))
            else:
                try:
                    P, R, TH = precision_recall_curve(y_true, y_prob)
                    if len(TH) == 0:
                        best_thresh_fbeta.append(0.5)
                    else:
                        F_score = (2.0 * P[1:] * R[1:]) / (P[1:] + R[1:] + 1e-12)
                        idx = int(np.nanargmax(F_score))
                        best_thresh_fbeta.append(float(TH[idx]))
                except Exception:
                    best_thresh_fbeta.append(float('nan'))

    idxs = [1, 2, 3]
    macro_sens = float(recall[idxs].mean().detach().cpu())
    macro_f1 = float(f1[idxs].mean().detach().cpu())

    TP = float(tp[idxs].sum().detach().cpu())
    FP = float(fp[idxs].sum().detach().cpu())
    FN = float(fn[idxs].sum().detach().cpu())
    micro_sens = TP / (TP + FN + 1e-8)
    micro_f1 = 2 * TP / (2 * TP + FP + FN + 1e-8)

    return {
        'loss': float(test_loss),
        'precision': precision.detach().cpu().tolist(),
        'recall': recall.detach().cpu().tolist(),
        'f1': f1.detach().cpu().tolist(),
        'auc': auc_list,
        'macro': {
            'sensitivity': macro_sens,
            'f1': macro_f1,
        },
        'micro': {
            'sensitivity': micro_sens,
            'f1': micro_f1,
        },
        'best_thr': best_thresh_youden,
        'best_thr_f1': best_thresh_fbeta,
        'confusion': {
            'tp': tp.detach().cpu().tolist(),
            'fp': fp.detach().cpu().tolist(),
            'fn': fn.detach().cpu().tolist(),
            'tn': tn.detach().cpu().tolist(),
        },
        'actual': {
            'pos': (tp + fn).detach().cpu().tolist(),
            'neg': (tn + fp).detach().cpu().tolist(),
        },
        'roc_raw': {
            'y_true': [np.concatenate(y_true_all[j]) if len(y_true_all[j]) > 0 else np.array([]) for j in range(4)],
            'y_prob': [np.concatenate(y_prob_all[j]) if len(y_prob_all[j]) > 0 else np.array([]) for j in range(4)],
        },
    }
