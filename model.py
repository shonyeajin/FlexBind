import os
import math
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from tqdm.auto import tqdm


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
            F = 2 * P[1:] * R[1:] / (P[1:] + R[1:] + 1e-12)
            best_idx = np.argmax(F)
            best_f1 = F[best_idx]
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


def log(s: str):
    print(f'[LOG] {s}')


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_ids(npz_ids):
    n = len(npz_ids)
    train_ids = [npz_ids[i] for i in range(n)]
    return train_ids


def split_ids(npz_ids, val_ratio=0.1, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(npz_ids))
    rng.shuffle(idx)

    n_val = int(len(npz_ids) * val_ratio)
    if n_val == 0 and len(npz_ids) > 1:
        n_val = 1

    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_ids = [npz_ids[i] for i in train_idx]
    val_ids = [npz_ids[i] for i in val_idx]

    return train_ids, val_ids


def parse_thresholds(s: str, default=0.5):
    names = ['disorder', 'protein', 'rna', 'dna']
    out = {k: default for k in names}

    if not s:
        return out
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        k, v = tok.split('=', 1)
        k = k.strip().lower()
        v = float(v.strip())
        if k in out:
            out[k] = v
    return out


class IDREmbDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        label_files: Dict[str, str],
        id_key: str = "ids",
        per_res_key: str = "per_residue",
        max_len: Optional[int] = None,
        include_ids: Optional[set] = None,
    ):
        super().__init__()
        self.npz = np.load(npz_path, allow_pickle=True)
        self.ids: List[str] = list(map(str, self.npz[id_key]))
        self.per_residue = list(self.npz[per_res_key])
        self.dim = int(self.npz.get('dim', self.per_residue[0].shape[-1]))
        self.max_len = max_len
        self.include_ids = set(include_ids) if include_ids is not None else None

        self.tasks = ['disorder', 'protein', 'rna', 'dna']
        for t in self.tasks:
            if t not in label_files:
                raise ValueError(f"Missing label file for task '{t}'. Provided keys: {list(label_files.keys())}")

        self.label_maps: Dict[str, Dict[str, np.ndarray]] = {
            t: self._load_label_txt(label_files[t]) for t in self.tasks
        }

        self.id_to_rawidx = {sid: i for i, sid in enumerate(self.ids)}

        self.valid_indices: List[int] = []
        dropped: List[Tuple[str, int, Dict[str, int]]] = []

        for i, sid in enumerate(self.ids):
            if self.include_ids is not None and sid not in self.include_ids:
                continue

            L = self.per_residue[i].shape[0]
            ok = True
            mis = {}

            for t in self.tasks:
                arr = self.label_maps[t].get(sid)
                if arr is None:
                    ok = False
                    mis[t] = -1
                elif len(arr) != L:
                    ok = False
                    mis[t] = len(arr)

            if ok:
                self.valid_indices.append(i)
            else:
                dropped.append((sid, L, mis))

        if dropped:
            log(f'Dropped {len(dropped)} sequences with missing/length-mismatched labels (showing up to 30): {dropped[:30]}')
        log(f'Dataset ready. kept={len(self.valid_indices)} / total={len(self.ids)}, dim={self.dim}')

    @staticmethod
    def _parse_label_line(line: str) -> Optional[List[int]]:
        line = line.strip()
        if not line:
            return None
        if set(line) <= set("01") and len(line) > 1:
            return [int(c) for c in line]
        else:
            return None

    def _load_label_txt(self, path: str) -> Dict[str, np.ndarray]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f'label file not found: {path}')
        mp: Dict[str, np.ndarray] = {}
        with open(path, 'r', encoding='utf-8') as f:
            lines = [ln.rstrip('\n') for ln in f]
        for ln in lines:
            if not ln.strip():
                continue
            sid, payload = ln.split('\t', 1)
            arr = self._parse_label_line(payload)
            if arr is None:
                raise ValueError(f'Could not parse labels in TSV line: {ln}')
            mp[sid] = np.asarray(arr, dtype=np.int64)
        return mp

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        i = self.valid_indices[idx]
        sid = self.ids[i]
        H = self.per_residue[i].astype(np.float32)
        L, d = H.shape

        y = np.zeros((L, 4), dtype=np.float32)
        for j, t in enumerate(self.tasks):
            y[:, j] = self.label_maps[t][sid].astype(np.float32)

        if self.max_len and L > self.max_len:
            start = (L - self.max_len) // 2
            H = H[start:start + self.max_len]
            y = y[start:start + self.max_len]
            L = self.max_len

        return {
            "id": sid,
            "emb": torch.from_numpy(H),
            "label": torch.from_numpy(y),
            "len": L,
        }


def collate_pad(batch: List[dict]):
    Ls = [b["len"] for b in batch]
    Lmax = max(Ls)
    d = batch[0]['emb'].shape[-1]
    B = len(batch)

    emb = torch.zeros(B, Lmax, d, dtype=torch.float32)
    lab = torch.full((B, Lmax, 4), fill_value=-100.0, dtype=torch.float32)
    mask = torch.zeros(B, Lmax, dtype=torch.bool)
    ids = []

    for i, b in enumerate(batch):
        L = b["len"]
        emb[i, :L] = b['emb']
        lab[i, :L] = b['label']
        mask[i, :L] = True
        ids.append(b['id'])

    return {'ids': ids, 'emb': emb, 'label': lab, 'mask': mask}


def build_loader(cfg, use_ids=None, test_flag=None):
    if test_flag is None:
        npz = np.load(cfg.npz_path, allow_pickle=True)
    else:
        npz = np.load(cfg.test_npz_path, allow_pickle=True)

    ids = [str(x) for x in npz['ids']]

    selected_ids = list(ids)
    if use_ids is not None:
        wanted = set(use_ids)
        selected_ids = [sid for sid in ids if sid in wanted]

    if test_flag is None:
        ds = IDREmbDataset(
            npz_path=cfg.npz_path,
            label_files={
                "disorder": cfg.disorder_txt,
                "protein": cfg.protein_txt,
                "rna": cfg.rna_txt,
                "dna": cfg.dna_txt,
            },
            max_len=cfg.max_len,
            include_ids=set(selected_ids)
        )
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=collate_pad,
            pin_memory=True
        )
    else:
        ds = IDREmbDataset(
            npz_path=cfg.test_npz_path,
            label_files={
                "disorder": cfg.test_disorder_txt,
                "protein": cfg.test_protein_txt,
                "rna": cfg.test_rna_txt,
                "dna": cfg.test_dna_txt,
            },
            max_len=cfg.max_len,
            include_ids=set(selected_ids)
        )
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_pad,
            pin_memory=True
        )


class ComplexCWT(nn.Module):
    def __init__(self, in_dim, scales, wavelet_dim, kernel_size=5):
        super().__init__()
        self.scales = scales
        self.D_w = wavelet_dim
        self.kernel_size = kernel_size

        self.dim_reduce = nn.Conv1d(in_dim, wavelet_dim, kernel_size=1)

        self.kernels_real = nn.Parameter(
            torch.randn(len(scales), wavelet_dim, wavelet_dim, kernel_size)
        )
        self.kernels_imag = nn.Parameter(
            torch.randn(len(scales), wavelet_dim, wavelet_dim, kernel_size)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dim_reduce(x)

        real_out = []
        imag_out = []

        for i, s in enumerate(self.scales):
            ker_r = self.kernels_real[i]
            ker_i = self.kernels_imag[i]

            r = F.conv1d(x, ker_r, padding=(self.kernel_size // 2) * s, dilation=s)
            m = F.conv1d(x, ker_i, padding=(self.kernel_size // 2) * s, dilation=s)

            real_out.append(r)
            imag_out.append(m)

        real_out = torch.stack(real_out, dim=2)
        imag_out = torch.stack(imag_out, dim=2)

        return real_out, imag_out


class ScaleAttention(nn.Module):
    def __init__(self, D_plm, D_att, D_w):
        super().__init__()
        self.W_Q = nn.Linear(D_plm, D_att)
        self.W_K = nn.Linear(D_w, D_att)
        self.W_V = nn.Linear(D_w, D_att)

    def forward(self, E, W_red):
        B, D_w, S, L = W_red.shape

        Q = self.W_Q(E)
        W_red_perm = W_red.permute(0, 3, 2, 1)
        W_red_flat = W_red_perm.reshape(B * L, S, D_w)

        K = self.W_K(W_red_flat)
        V = self.W_V(W_red_flat)

        Q_flat = Q.reshape(B * L, 1, -1)
        att = torch.softmax((Q_flat @ K.transpose(1, 2)) / math.sqrt(K.shape[-1]), dim=-1)
        ctx = att @ V

        ctx = ctx.reshape(B, L, -1)
        return ctx


class ComplexInteractionBlock(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.a = nn.Parameter(torch.ones(D))
        self.b = nn.Parameter(torch.zeros(D))
        self.fuse = nn.Linear(2 * D, D)

    def forward(self, R, I):
        R_ = self.a * R - self.b * I
        I_ = self.b * R + self.a * I
        out = torch.cat([R_, I_], dim=-1)
        return self.fuse(out)


class TinyMamba(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(D, 4 * D),
            nn.GELU(),
            nn.Linear(4 * D, D)
        )
        self.norm = nn.LayerNorm(D)

    def forward(self, x):
        return self.norm(x + self.ff(x))


class MambaLite(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, conv_kernel=4):
        super().__init__()

        hidden = d_model * expand

        self.gate_proj = nn.Linear(d_model, hidden)
        self.x_proj = nn.Linear(d_model, hidden)

        self.dwconv = nn.Conv1d(
            hidden, hidden, kernel_size=conv_kernel,
            groups=hidden, padding=conv_kernel // 2
        )

        self.ss_out = nn.Linear(hidden, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        gate = torch.sigmoid(self.gate_proj(x))
        u = self.x_proj(x)

        h = u.transpose(1, 2)
        h = self.dwconv(h)
        h = h.transpose(1, 2)

        h = h * gate
        out = self.ss_out(h)

        return self.norm(x + out)


class MambaPython(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, conv_kernel=3):
        super().__init__()

        self.d_model = d_model
        hidden = d_model * expand
        self.hidden = hidden
        self.d_state = d_state

        self.x_proj = nn.Linear(d_model, hidden)
        self.g_proj = nn.Linear(d_model, hidden)

        self.dwconv = nn.Conv1d(
            hidden, hidden,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=hidden
        )

        self.A_raw = nn.Parameter(torch.randn(hidden, d_state))
        self.B = nn.Parameter(torch.randn(hidden, d_state))
        self.C = nn.Parameter(torch.randn(hidden, d_state))

        self.dt_proj = nn.Linear(d_model, hidden)
        self.out_proj = nn.Linear(hidden, d_model)
        self.norm = nn.LayerNorm(d_model)

    def discretize(self, A, B, dt):
        dt = dt.unsqueeze(-1)
        A_dt = A * dt
        A_d = torch.exp(A_dt)

        A_safe = A + 1e-6
        B_d = ((A_d - 1.0) / A_safe) * B

        return A_d, B_d

    def selective_scan(self, A_d, B_d, C, u):
        B_, L, H = u.shape
        S = A_d.shape[1]

        x = torch.zeros(B_, H, S, device=u.device)
        outputs = []

        for t in range(L):
            u_t = u[:, t, :].unsqueeze(-1)
            x = A_d.unsqueeze(0) * x + B_d.unsqueeze(0) * u_t

            y_t = torch.sum(x * C.unsqueeze(0), dim=-1)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)

    def forward(self, x):
        B, L, D = x.shape
        u = self.x_proj(x)
        g = torch.sigmoid(self.g_proj(x))

        h = u.transpose(1, 2)
        h = self.dwconv(h)
        h = h.transpose(1, 2)

        u = u * g + h * (1 - g)

        dt_full = F.softplus(self.dt_proj(x))
        dt = dt_full.mean(dim=(0, 1))
        dt = torch.clamp(dt, 0.001, 2.0)

        A = -F.softplus(self.A_raw)
        A_d, B_d = self.discretize(A, self.B, dt)

        y = self.selective_scan(A_d, B_d, self.C, u)
        out = self.out_proj(y)

        return self.norm(x + out)


class IDRModel(nn.Module):
    def __init__(self, d, D_att=128, D_w=128, scales=[1, 3, 4, 5, 8, 16, 32]):
        super().__init__()
        self.d_plm = d
        self.D_w = D_w
        self.D_att = D_att

        self.heads = nn.ModuleDict({
            "disorder": nn.Linear(d, 1),
            "protein": nn.Linear(d, 1),
            "rna": nn.Linear(d, 1),
            "dna": nn.Linear(d, 1),
        })

        self.cwt = ComplexCWT(in_dim=d, scales=scales, wavelet_dim=D_w)
        self.real_att = ScaleAttention(d, D_att, D_w)
        self.imag_att = ScaleAttention(d, D_att, D_w)
        self.cib = ComplexInteractionBlock(D_att)
        self.out_proj = nn.Linear(D_att, d)

        self.mamba_blocks = nn.ModuleList([
            MambaPython(d_model=d, d_state=16, expand=2, conv_kernel=3)
            for _ in range(3)
        ])

    def forward(self, emb, mask=None):
        real, imag = self.cwt(emb)
        ctx_R = self.real_att(emb, real)
        ctx_I = self.imag_att(emb, imag)

        fused = self.cib(ctx_R, ctx_I)
        E_fused = emb + self.out_proj(fused)

        H = E_fused
        for blk in self.mamba_blocks:
            H = blk(H)

        out = {k: self.heads[k](H).squeeze(-1) for k in self.heads}
        return out


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


def train_one_epoch(model, loader, optimizer, cfg, epoch: int):
    model.train()
    total = 0.0

    pbar = tqdm(loader, desc=f'Train Epoch {epoch}', leave=False)
    for step, batch in enumerate(pbar):
        H = batch['emb'].to(cfg.device)
        Y = batch['label'].to(cfg.device)
        M = batch['mask'].to(cfg.device)

        logits = model(H, M)
        loss_sum = 0.0
        names = ['disorder', 'protein', 'rna', 'dna']

        task_weights = {
            "disorder": 1.0,
            "protein": 2.43,
            "rna": 27.61,
            "dna": 17.63,
        }

        pos_weight_map = {
            "disorder": torch.tensor(3.51, device=cfg.device),
            "protein": torch.tensor(9.98, device=cfg.device),
            "rna": torch.tensor(123.63, device=cfg.device),
            "dna": torch.tensor(78.59, device=cfg.device),
        }

        for j, name in enumerate(names):
            y = Y[:, :, j]
            mask = M & (y > -50)

            pw = pos_weight_map[name]
            l = masked_bce_with_logits(logits[name], y, mask, pos_weight=pw)

            if cfg.use_focal:
                l = l + focal_bce_with_logits(logits[name], y, mask, alpha=cfg.alphas[name])
            if cfg.use_dice:
                l = l + dice_loss(logits[name], y, mask)

            task_w = task_weights[name]
            loss_sum = loss_sum + l * task_w

        optimizer.zero_grad(set_to_none=True)
        loss_sum.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        total += loss_sum.item()
        pbar.set_postfix(loss=f"{(total / (step + 1)):.4f}")

    return total / max(len(loader), 1)


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
                        F = (2.0 * P[1:] * R[1:]) / (P[1:] + R[1:] + 1e-12)
                        idx = int(np.nanargmax(F))
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz_path', required=True)
    p.add_argument('--test_npz_path', required=True)
    p.add_argument('--disorder_txt', required=True)
    p.add_argument('--protein_txt', required=True)
    p.add_argument('--rna_txt', required=True)
    p.add_argument('--dna_txt', required=True)
    p.add_argument('--test_disorder_txt', required=True)
    p.add_argument('--test_protein_txt', required=True)
    p.add_argument('--test_rna_txt', required=True)
    p.add_argument('--test_dna_txt', required=True)
    p.add_argument('--out_dir', default='./checkpoints')
    p.add_argument('--batch_size', type=int, default=3)
    p.add_argument('--max_epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--max_len', type=int, default=0, help='optional crop length (0=no crop)')
    p.add_argument('--no_focal', action='store_true')
    p.add_argument('--no_dice', action='store_true')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--thresholds', type=str, default=None, help="e.g., 'disorder=0.4,protein=0.6,rna=0.35,dna=0.55'")
    p.add_argument('--alphas', type=str, default=None, help="e.g., 'disorder=0.4,protein=0.6,rna=0.35,dna=0.55'")
    args = p.parse_args()

    class c:
        pass

    cfg = c()
    cfg.npz_path = args.npz_path
    cfg.test_npz_path = args.test_npz_path
    cfg.disorder_txt = args.disorder_txt
    cfg.protein_txt = args.protein_txt
    cfg.rna_txt = args.rna_txt
    cfg.dna_txt = args.dna_txt
    cfg.test_disorder_txt = args.test_disorder_txt
    cfg.test_protein_txt = args.test_protein_txt
    cfg.test_rna_txt = args.test_rna_txt
    cfg.test_dna_txt = args.test_dna_txt
    cfg.out_dir = args.out_dir
    cfg.batch_size = args.batch_size
    cfg.max_epochs = args.max_epochs
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.grad_clip = args.grad_clip
    cfg.max_len = None if args.max_len in (0, None) else int(args.max_len)
    cfg.use_focal = not args.no_focal
    cfg.use_dice = not args.no_dice
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.thresholds = parse_thresholds(args.thresholds, default=0.5)
    cfg.alphas = parse_thresholds(args.alphas, default=0.5)
    cfg.seed = args.seed
    cfg.val_ratio = args.val_ratio

    log(f'Using thresholds: {cfg.thresholds}')
    log(f'Using alphas: {cfg.alphas}')
    log(f'Using device: {cfg.device}')

    set_seed(cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)
    best_ckpt_dir = os.path.join(cfg.out_dir, 'DP93')
    os.makedirs(best_ckpt_dir, exist_ok=True)

    npz = np.load(cfg.npz_path, allow_pickle=True)
    npz_ids = [str(x) for x in npz['ids']]

    test_npz = np.load(cfg.test_npz_path, allow_pickle=True)
    test_npz_ids = [str(x) for x in test_npz['ids']]

    train_ids, val_ids = split_ids(npz_ids, val_ratio=cfg.val_ratio, seed=cfg.seed)
    test_ids = make_ids(test_npz_ids)

    log(f"Train size: {len(train_ids)}")
    log(f"Val size:   {len(val_ids)}")
    log(f"Test size:  {len(test_ids)}")

    train_loader = build_loader(cfg, use_ids=train_ids)
    val_loader = build_loader(cfg, use_ids=val_ids)
    test_loader = build_loader(cfg, use_ids=test_ids, test_flag=1)

    sample = next(iter(train_loader))
    d = sample['emb'].shape[-1]

    model = IDRModel(d=d)
    model.to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_f1 = -1.0
    best_val_metrics = None
    best_model_path = os.path.join(best_ckpt_dir, 'disorder_best.pt')

    for ep in tqdm(range(1, cfg.max_epochs + 1), desc="Epochs"):
        train_loss = train_one_epoch(model, train_loader, opt, cfg, ep)

        val_metrics = evaluate(model, val_loader, cfg)
        best_f1_per_task = plot_roc_pr_curves(val_metrics, cfg.out_dir, suffix=f"_val_epoch{ep}")

        disorder_f1 = best_f1_per_task[0]

        if np.isfinite(disorder_f1) and disorder_f1 > best_val_f1:
            best_val_f1 = disorder_f1
            best_val_metrics = val_metrics
            torch.save(model.state_dict(), best_model_path)
            print(f"[BEST] Updated VAL disorder best-F1 = {best_val_f1:.4f} at epoch {ep}")

        val_loss = val_metrics['loss']
        val_f1_list = ['%.3f' % x for x in val_metrics['f1']]
        val_auc_list = ['%.3f' % (x if np.isfinite(x) else float('nan')) for x in val_metrics['auc']]
        val_rec_list = ['%.3f' % x for x in val_metrics['recall']]

        log(
            f"[epoch {ep}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
            f"\n| [VAL]\n"
            f"| F1(dis,pro,rna,dna)={val_f1_list} "
            f"| Sens(dis,pro,rna,dna)={val_rec_list} "
            f"| AUC(dis,pro,rna,dna)={val_auc_list} "
            f"| MACRO[pro,rna,dna]: Sens={val_metrics['macro']['sensitivity']:.3f}, F1={val_metrics['macro']['f1']:.3f} "
            f"| MICRO[pro,rna,dna]: Sens={val_metrics['micro']['sensitivity']:.3f}, F1={val_metrics['micro']['f1']:.3f} "
            f"| Best thres(dis,pro,rna,dna)={val_metrics['best_thr']} "
            f"| Best thres_f1(dis,pro,rna,dna)={val_metrics['best_thr_f1']}"
        )

        cm = val_metrics['confusion']
        act = val_metrics['actual']

        log(f"| Confusion Matrix (TP/FP/FN/TN) per task:")
        for name, tp_, fp_, fn_, tn_, pos_, neg_ in zip(
            ['disorder', 'protein', 'rna', 'dna'],
            cm['tp'], cm['fp'], cm['fn'], cm['tn'],
            act['pos'], act['neg']
        ):
            log(
                f"  - {name}: "
                f"TP={tp_:.0f}, FP={fp_:.0f}, FN={fn_:.0f}, TN={tn_:.0f} | "
                f"Actual+={pos_:.0f}, Actual-={neg_:.0f}"
            )

    if best_val_metrics is not None:
        plot_roc_pr_curves(best_val_metrics, cfg.out_dir, suffix='_val_best')
        print("Saved BEST-VAL ROC/PR plots!")

    if os.path.isfile(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=cfg.device))
        print(f"[LOAD] Loaded best model from: {best_model_path}")
    else:
        print("[WARN] Best model checkpoint not found. Using last epoch model for test evaluation.")

    test_metrics = evaluate(model, test_loader, cfg)
    plot_roc_pr_curves(test_metrics, cfg.out_dir, suffix='_test')

    test_f1_list = ['%.3f' % x for x in test_metrics['f1']]
    test_auc_list = ['%.3f' % (x if np.isfinite(x) else float('nan')) for x in test_metrics['auc']]
    test_rec_list = ['%.3f' % x for x in test_metrics['recall']]

    log(
        f"[FINAL TEST] loss={test_metrics['loss']:.4f} "
        f"\n| F1(dis,pro,rna,dna)={test_f1_list} "
        f"| Sens(dis,pro,rna,dna)={test_rec_list} "
        f"| AUC(dis,pro,rna,dna)={test_auc_list} "
        f"| MACRO[pro,rna,dna]: Sens={test_metrics['macro']['sensitivity']:.3f}, F1={test_metrics['macro']['f1']:.3f} "
        f"| MICRO[pro,rna,dna]: Sens={test_metrics['micro']['sensitivity']:.3f}, F1={test_metrics['micro']['f1']:.3f} "
        f"| Best thres(dis,pro,rna,dna)={test_metrics['best_thr']} "
        f"| Best thres_f1(dis,pro,rna,dna)={test_metrics['best_thr_f1']}"
    )

    cm = test_metrics['confusion']
    act = test_metrics['actual']

    log(f"| TEST Confusion Matrix (TP/FP/FN/TN) per task:")
    for name, tp_, fp_, fn_, tn_, pos_, neg_ in zip(
        ['disorder', 'protein', 'rna', 'dna'],
        cm['tp'], cm['fp'], cm['fn'], cm['tn'],
        act['pos'], act['neg']
    ):
        log(
            f"  - {name}: "
            f"TP={tp_:.0f}, FP={fp_:.0f}, FN={fn_:.0f}, TN={tn_:.0f} | "
            f"Actual+={pos_:.0f}, Actual-={neg_:.0f}"
        )


if __name__ == '__main__':
    main()
