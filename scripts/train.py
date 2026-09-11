import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# Add parent directory to python path to import models and scripts modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.flexbind import IDRModel
from scripts.evaluate import evaluate, plot_roc_pr_curves, masked_bce_with_logits, focal_bce_with_logits, dice_loss

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
    p.add_argument('--thresholds', type=str, default=None)
    p.add_argument('--alphas', type=str, default=None)
    args = p.parse_args()

    class c: pass
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

    model = IDRModel(d=d).to(cfg.device)
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

        log(f"[epoch {ep}] train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f}")

    if best_val_metrics is not None:
        plot_roc_pr_curves(best_val_metrics, cfg.out_dir, suffix='_val_best')

    if os.path.isfile(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=cfg.device))

    test_metrics = evaluate(model, test_loader, cfg)
    plot_roc_pr_curves(test_metrics, cfg.out_dir, suffix='_test')
    log(f"[FINAL TEST] loss={test_metrics['loss']:.4f}")

if __name__ == '__main__':
    main()
