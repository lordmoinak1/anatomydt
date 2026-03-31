from __future__ import annotations
import argparse, json, math, os, random, csv
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_erosion

from dataset_ucsf import make_metadata_dataset  # K=5: [tumor, anat1, anat2, anat3, background]

# ============================================================
# Utils
# ============================================================
def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass

def ensure_dir(p: str):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def ensure_finite(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

def get_anatomy_indices(K: int, tumor_idx: List[int]) -> List[int]:
    """All non-tumor, non-background class indices."""
    bg = K - 1
    return [i for i in range(K) if i not in tumor_idx and i != bg]

def save_binary_png(arr_bool: np.ndarray, path: str):
    ensure_dir(os.path.dirname(path))
    plt.imsave(path, arr_bool.astype(np.uint8), cmap="gray", vmin=0, vmax=1)

# ============================================================
# FD kernels & simplex projection
# ============================================================
def laplacian_kernel_2d(device: torch.device):
    k = torch.zeros((1, 1, 3, 3), device=device)
    k[0, 0, 1, 1] = -4.0
    k[0, 0, 1, 0] = 1.0; k[0, 0, 1, 2] = 1.0
    k[0, 0, 0, 1] = 1.0; k[0, 0, 2, 1] = 1.0
    return k

def grad_kernels_2d(device: torch.device):
    kx = torch.zeros((1, 1, 3, 3), device=device)
    ky = torch.zeros_like(kx)
    kx[0, 0, 1, 2] = 1.0; kx[0, 0, 1, 1] = -1.0
    ky[0, 0, 2, 1] = 1.0; ky[0, 0, 1, 1] = -1.0
    return kx, ky

def project_simplex(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    orig = p.shape
    K = p.shape[-1]
    x = p.reshape(-1, K)
    x = torch.nan_to_num(x)
    u, _ = torch.sort(x, dim=1, descending=True)
    cssv = torch.cumsum(u, dim=1) - 1
    ind = torch.arange(1, K+1, device=p.device).view(1, -1)
    cond = u - cssv / ind > 0
    rho = torch.argmax(cond.int() * ind, dim=1)
    rho_idx = torch.clamp(rho.view(-1, 1) - 1, min=0)
    tau = (cssv.gather(1, rho_idx) / (rho_idx + 1).clamp_min(1)).nan_to_num()
    w = torch.clamp(x - tau, min=0)
    denom = w.sum(dim=1, keepdim=True) + eps
    w = w / denom
    return w.reshape(orig)

# ============================================================
# GrowthCNN
# ============================================================
class GrowthCNN(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 16, k_max: float = 1.0):
        super().__init__()
        self.k_max = float(k_max)
        self.conv1 = nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(hidden, 1, kernel_size=1, padding=0, bias=True)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='linear')
        nn.init.zeros_(self.conv1.bias); nn.init.zeros_(self.conv2.bias); nn.init.zeros_(self.conv3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        raw = self.conv3(h)
        k_map = torch.sigmoid(raw) * self.k_max
        return k_map.clamp_min(1e-6)

# ============================================================
# IMEX cross-diffusion with learnable scalars + GrowthCNN
# ============================================================
@dataclass
class PDEParams:
    dt: float = 0.1
    steps: int = 12
    jacobi_iters: int = 8
    jacobi_omega: float = 0.9
    D: float = 0.5
    chi: float = 0.5
    k_tumor: float = 0.3
    cap_tumor: float = 0.7
    D_max: float = 2.0
    chi_max: float = 2.0

class CrossDiffusionIMEX2D(nn.Module):
    def __init__(self, K: int, params: PDEParams, use_growth_cnn: bool = True, k_max: float = 1.0,
                 k_mode: str = 'tumor_map', growth_in_ch: Optional[int] = None):
        super().__init__()
        self.K = K
        self.params = params
        self.use_growth_cnn = use_growth_cnn
        self.k_max = k_max
        self.k_mode = k_mode

        self.D_raw = nn.Parameter(torch.full((K,), float(params.D)))
        self.chi_raw = nn.Parameter(torch.tensor(float(params.chi)))
        self.k_tumor_raw = nn.Parameter(torch.tensor(float(params.k_tumor)))
        cap0 = params.cap_tumor if 0 < params.cap_tumor < 1 else 0.7
        cap_logit = math.log(cap0) - math.log(1.0 - cap0)
        self.cap_tumor_logit = nn.Parameter(torch.tensor(cap_logit, dtype=torch.float32))

        self.register_buffer('lap_k', torch.zeros(1))
        self.grad_kx = None; self.grad_ky = None

        if use_growth_cnn:
            gin = growth_in_ch if growth_in_ch is not None else K
            self.expected_in_ch = int(gin)
            self.growth_net = GrowthCNN(in_ch=gin, hidden=16, k_max=k_max)
        else:
            self.growth_net = None
            self.expected_in_ch = None

        self.last_kmap: Optional[torch.Tensor] = None

    def _ensure_kernels(self, device):
        if not torch.is_tensor(self.lap_k) or self.lap_k.ndim == 0 or self.lap_k.numel() == 1:
            self.lap_k = laplacian_kernel_2d(device)
            self.grad_kx, self.grad_ky = grad_kernels_2d(device)

    def laplacian(self, x):
        x = F.pad(x, (1,1,1,1), mode='replicate')
        return F.conv2d(x, self.lap_k.repeat(x.size(1),1,1,1), groups=x.size(1))

    def gradient(self, x):
        xpad = F.pad(x, (1,1,1,1), mode='replicate')
        gx = F.conv2d(xpad, self.grad_kx.repeat(x.size(1),1,1,1), groups=x.size(1))
        gy = F.conv2d(xpad, self.grad_ky.repeat(x.size(1),1,1,1), groups=x.size(1))
        return gx, gy

    def divergence(self, vx, vy):
        xpad = F.pad(vx, (1,1,1,1), mode='replicate')
        ypad = F.pad(vy, (1,1,1,1), mode='replicate')
        dvx = F.conv2d(xpad, -self.grad_kx.repeat(vx.size(1),1,1,1), groups=vx.size(1))
        dvy = F.conv2d(ypad, -self.grad_ky.repeat(vy.size(1),1,1,1), groups=vy.size(1))
        return dvx + dvy

    def forward(self, p0: torch.Tensor, tumor_idx: Optional[List[int]] = None,
                steps_override: Optional[int] = None,
                growth_feats: Optional[torch.Tensor] = None) -> torch.Tensor:

        self._ensure_kernels(p0.device)
        dt    = self.params.dt
        K     = self.K
        steps = int(steps_override) if (steps_override is not None and steps_override > 0) else int(self.params.steps)

        D_all = (F.softplus(self.D_raw) + 1e-8).clamp_max(self.params.D_max)
        chi   = (F.softplus(self.chi_raw) + 1e-8).clamp_max(self.params.chi_max)
        k_t   = (F.softplus(self.k_tumor_raw) + 1e-8).clamp_max(self.k_max)
        cap_t = torch.sigmoid(self.cap_tumor_logit).clamp(1e-3, 0.95)

        enc_in = p0
        if self.use_growth_cnn:
            if isinstance(growth_feats, torch.Tensor):
                enc_in = torch.cat([enc_in, growth_feats], dim=1)
            if self.expected_in_ch is not None and enc_in.size(1) != self.expected_in_ch:
                need = self.expected_in_ch - enc_in.size(1)
                if need > 0:
                    B, _, H, W = enc_in.shape
                    enc_in = torch.cat([enc_in, enc_in.new_zeros(B, need, H, W)], dim=1)
                elif need < 0:
                    enc_in = enc_in[:, :self.expected_in_ch]

        k_map = None
        if self.use_growth_cnn and self.growth_net is not None:
            k_map = self.growth_net(enc_in)  # [B,1,H,W]
        self.last_kmap = k_map

        p = torch.nan_to_num(p0).clamp_min(0)
        for _ in range(steps):
            gx, gy = self.gradient(p); gx = torch.nan_to_num(gx); gy = torch.nan_to_num(gy)
            sum_gx = gx.sum(1, keepdim=True); sum_gy = gy.sum(1, keepdim=True)
            fx = -chi * p * (sum_gx - gx); fy = -chi * p * (sum_gy - gy)
            fx, fy = torch.nan_to_num(fx), torch.nan_to_num(fy)
            cross_term = self.divergence(fx, fy)
            cross_term = torch.nan_to_num(cross_term)

            react = torch.zeros_like(p)
            if tumor_idx is not None and len(tumor_idx) > 0:
                if (self.k_mode == 'scalar') or (k_map is None):
                    for ti in tumor_idx:
                        pt = p[:, ti:ti+1].clamp(0, 1)
                        rt = k_t * pt * (1.0 - pt / cap_t)
                        react[:, ti:ti+1] = torch.nan_to_num(rt)
                else:
                    kt = k_map[:, :1]
                    for ti in tumor_idx:
                        pt = p[:, ti:ti+1].clamp(0, 1)
                        rt = kt * pt * (1.0 - pt / cap_t)
                        react[:, ti:ti+1] = torch.nan_to_num(rt)

            rhs = torch.nan_to_num(p + dt * (cross_term + react))

            p_new = torch.empty_like(p)
            for c in range(K):
                x  = p[:, c:c+1]; b = rhs[:, c:c+1]
                Dc = D_all[c]; diag = 1.0 + dt * Dc * 4.0
                xk = x
                for _it in range(self.params.jacobi_iters):
                    Lx = self.laplacian(xk)
                    r  = b - (xk - dt * Dc * Lx)
                    xk = xk + (self.params.jacobi_omega / diag) * r
                    xk = torch.nan_to_num(xk)
                p_new[:, c:c+1] = xk

            p = project_simplex(p_new.movedim(1, -1)).movedim(-1, 1)
            p = torch.nan_to_num(p).clamp_min(0)
        return p

# ============================================================
# Metrics 
# ============================================================
def dice_per_class_from_probs(probs: torch.Tensor, target_oh: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.nan_to_num(probs).clamp(0, 1); t = torch.nan_to_num(target_oh).clamp(0, 1)
    inter = (p * t).sum(dim=(0,2,3)); denom = p.sum(dim=(0,2,3)) + t.sum(dim=(0,2,3))
    return (2 * inter + eps) / (denom + eps)

def dice_subset_from_probs(probs: torch.Tensor, target_oh: torch.Tensor, subset: List[int], eps: float = 1e-6) -> float:
    p = torch.nan_to_num(probs)[:, subset].clamp(0, 1)
    t = torch.nan_to_num(target_oh)[:, subset].clamp(0, 1)
    inter = (p * t).sum(dim=(2,3))
    denom = p.sum(dim=(2,3)) + t.sum(dim=(2,3))
    dice  = (2 * inter + eps) / (denom + eps)
    return float(dice.mean().item())

def _surface(mask: np.ndarray) -> np.ndarray:
    er = binary_erosion(mask, border_value=0)
    return np.logical_xor(mask, er)

def hd95_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.sum() == 0 and gt.sum() == 0: return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        h, w = pred.shape[-2], pred.shape[-1]
        return float(np.hypot(h, w))
    sp = _surface(pred); sg = _surface(gt)
    if not sp.any() and not sg.any(): return 0.0
    dt_p = distance_transform_edt(~sp); dt_g = distance_transform_edt(~sg)
    d_pg = dt_g[sp].ravel() if sp.any() else np.array([])
    d_gp = dt_p[sg].ravel() if sg.any() else np.array([])
    if d_pg.size == 0 and d_gp.size == 0: return 0.0
    all_d = np.concatenate([d_pg, d_gp]) if (d_pg.size and d_gp.size) else (d_pg if d_pg.size else d_gp)
    return float(np.percentile(all_d, 95))

def hd95_macro_from_hard(pred_hard: torch.Tensor, gt_hard: torch.Tensor, K: int) -> float:
    pred_np = pred_hard.detach().cpu().numpy(); gt_np = gt_hard.detach().cpu().numpy()
    B, H, W = pred_np.shape; vals = []
    for k in range(K):
        ck = []
        for b in range(B):
            ck.append(hd95_binary(pred_np[b] == k, gt_np[b] == k))
        vals.append(float(np.mean(ck)))
    return float(np.mean(vals))

def hd95_subset_from_hard(pred_hard: torch.Tensor, gt_hard: torch.Tensor, subset: List[int]) -> float:
    pred_np = pred_hard.detach().cpu().numpy(); gt_np = gt_hard.detach().cpu().numpy()
    B, H, W = pred_np.shape; vals = []
    for k in subset:
        ck = []
        for b in range(B):
            ck.append(hd95_binary(pred_np[b] == k, gt_np[b] == k))
        vals.append(float(np.mean(ck)))
    return float(np.mean(vals)) if vals else 0.0

def to_onehot_from_softmax(sm: torch.Tensor) -> torch.Tensor:
    hard = sm.argmax(dim=1)
    K = sm.size(1)
    oh = torch.zeros_like(sm)
    oh.scatter_(1, hard.unsqueeze(1), 1.0)
    return oh

def save_pred_gt_panels(ph_bool: np.ndarray, gh_bool: np.ndarray, out_path: str, show=False):
    b = 0; K = ph_bool.shape[1]
    fig, axs = plt.subplots(2, K, figsize=(3*K, 6))
    for k in range(K):
        axs[0, k].imshow(gh_bool[b, k], cmap="gray"); axs[0, k].set_title(f"GT {k}");   axs[0, k].axis("off")
        axs[1, k].imshow(ph_bool[b, k], cmap="gray"); axs[1, k].set_title(f"Pred {k}"); axs[1, k].axis("off")
    plt.tight_layout(); ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)

def save_overlay_labelmaps(pred_lbl: np.ndarray, gt_lbl: np.ndarray, out_path: str, show=False):
    from matplotlib.colors import ListedColormap
    base_cmap = plt.get_cmap("tab20")
    K = int(max(np.max(pred_lbl), np.max(gt_lbl))) + 1
    colors = base_cmap(np.linspace(0, 1, max(3, K)))
    lbl_cmap = ListedColormap(colors)
    err = np.zeros((*gt_lbl.shape, 3), dtype=np.float32)
    fp = (pred_lbl != gt_lbl) & (pred_lbl != 0)
    fn = (pred_lbl != gt_lbl) & (gt_lbl != 0)
    err[..., :] = 0.7 * np.stack([~fp & ~fn, ~fp & ~fn, ~fp & ~fn], axis=-1)
    err[fp] = np.array([1.0, 0.2, 0.2]); err[fn] = np.array([0.2, 1.0, 0.2])
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(gt_lbl, cmap=lbl_cmap, vmin=0, vmax=K-1); axs[0].set_title("GT");   axs[0].axis("off")
    axs[1].imshow(pred_lbl, cmap=lbl_cmap, vmin=0, vmax=K-1); axs[1].set_title("Pred"); axs[1].axis("off")
    axs[2].imshow(err); axs[2].set_title("FP/FN"); axs[2].axis("off")
    plt.tight_layout(); ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)

# ============================================================
# Losses 
# ============================================================
def dice_loss_multiclass(pred_prob: torch.Tensor, target_prob: torch.Tensor,
                         class_idxs: Optional[List[int]] = None,
                         eps: float = 1e-6, reduction: str = "mean") -> torch.Tensor:
    pred = torch.nan_to_num(pred_prob).clamp(0, 1)
    targ = torch.nan_to_num(target_prob).clamp(0, 1)
    if class_idxs is not None and len(class_idxs) > 0:
        pred = pred[:, class_idxs]
        targ = targ[:, class_idxs]
    inter = (pred * targ).sum(dim=(2,3))
    denom = pred.sum(dim=(2,3)) + targ.sum(dim=(2,3))
    dice  = (2 * inter + eps) / (denom + eps)
    dice_b = dice.mean(dim=1)
    loss = 1.0 - dice_b
    return loss.mean() if reduction == "mean" else loss.sum() if reduction == "sum" else loss

def tv_loss_map(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()

def anatomy_consistency(pred_prob: torch.Tensor, src_prob: torch.Tensor,
                        target_prob: torch.Tensor, anatomy_idxs: List[int]) -> torch.Tensor:
    if len(anatomy_idxs) == 0:
        return pred_prob.new_zeros(())
    pred = pred_prob[:, anatomy_idxs]
    src  = src_prob[:, anatomy_idxs]
    targ = target_prob[:, anatomy_idxs]
    wt = 1.0 - targ
    return (wt * (pred - src)**2).mean()

# ============================================================
# Data utils 
# ============================================================
def collate_pairs(batch: List[Dict]) -> Dict:
    out: Dict = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out

@torch.no_grad()
def evaluate_and_save(model: nn.Module, loader: DataLoader, K: int, device: torch.device,
                      out_fold_dir: Optional[str], show: bool, viz_n: int, steps_per_day: float,
                      default_steps: int, tumor_idx: List[int], report_macro: bool = True,
                      anatomy_idx: Optional[List[int]] = None, save_all_masks_dir: Optional[str] = None) -> Dict[str, float]:
    model.eval()
    tum_dice_list, tum_hd95_list = [], []
    ana_dice_list, ana_hd95_list = [], []
    mac_dice_list, mac_hd95_list = [], []
    per_samples = []; saved = 0

    if anatomy_idx is None:
        anatomy_idx = get_anatomy_indices(K, tumor_idx)

    for bidx, batch in enumerate(loader):
        p1_list = batch['p_t1'].to(device)
        p2_list = batch['p_t2'].to(device)
        growth_feats = batch.get('growth_feats', None)
        if isinstance(growth_feats, torch.Tensor): growth_feats = growth_feats.to(device)

        B = p1_list.size(0); preds = []
        for bb in range(B):
            dt_days = batch["delta_t"][bb]
            steps_here = default_steps if (dt_days is None) else max(1, int(np.ceil(float(dt_days) * steps_per_day)))
            pred_bb = model(p1_list[bb:bb+1], tumor_idx=tumor_idx, steps_override=steps_here,
                            growth_feats=(growth_feats[bb:bb+1] if isinstance(growth_feats, torch.Tensor) else None))
            preds.append(pred_bb)
        pred = torch.cat(preds, dim=0)

        pred_hard = pred.argmax(dim=1)
        gt_hard   = p2_list.argmax(dim=1)

        tum_dice = dice_subset_from_probs(pred, p2_list, tumor_idx)
        tum_hd95 = hd95_subset_from_hard(pred_hard, gt_hard, tumor_idx)
        tum_dice_list.extend([tum_dice]*B)
        tum_hd95_list.extend([tum_hd95]*B)

        if len(anatomy_idx) > 0:
            ana_dice = dice_subset_from_probs(pred, p2_list, anatomy_idx)
            ana_hd95 = hd95_subset_from_hard(pred_hard, gt_hard, anatomy_idx)
            ana_dice_list.extend([ana_dice]*B)
            ana_hd95_list.extend([ana_hd95]*B)
        else:
            ana_dice = ana_hd95 = 0.0

        if report_macro:
            dice_k = dice_per_class_from_probs(pred, p2_list).detach().cpu().numpy()
            mac_dice = float(dice_k.mean())
            mac_hd95 = hd95_macro_from_hard(pred_hard, gt_hard, K)
            mac_dice_list.extend([mac_dice]*B)
            mac_hd95_list.extend([mac_hd95]*B)
        else:
            mac_dice = mac_hd95 = None

        for bb in range(B):
            per_samples.append({
                "subject": batch["subject"][bb],
                "tp1": batch["tp1"][bb],
                "tp2": batch["tp2"][bb],
                "delta_t": (None if batch["delta_t"][bb] is None else float(batch["delta_t"][bb])),
                "dice_tumor": tum_dice,
                "hd95_tumor": tum_hd95,
                "dice_anatomy": (ana_dice if len(anatomy_idx) > 0 else None),
                "hd95_anatomy": (ana_hd95 if len(anatomy_idx) > 0 else None),
                **({ "dice_macro": mac_dice, "hd95_macro": mac_hd95 } if report_macro else {})
            })

        if out_fold_dir and saved < viz_n:
            viz_dir = os.path.join(out_fold_dir, "viz"); ensure_dir(viz_dir)
            ph = to_onehot_from_softmax(pred).cpu().numpy().astype(bool)
            gh = to_onehot_from_softmax(p2_list).cpu().numpy().astype(bool)
            panels_path = os.path.join(viz_dir, f"batch{bidx:03d}_panels.png")
            save_pred_gt_panels(ph, gh, panels_path, show=show)
            pred_lbl = np.argmax(ph[0].astype(np.uint8), axis=0)
            gt_lbl   = np.argmax(gh[0].astype(np.uint8), axis=0)
            overlay_path = os.path.join(viz_dir, f"batch{bidx:03d}_overlay.png")
            save_overlay_labelmaps(pred_lbl, gt_lbl, overlay_path, show=show)
            saved += 1

        if save_all_masks_dir is not None:
            ph = to_onehot_from_softmax(pred).cpu().numpy().astype(bool)
            gh = to_onehot_from_softmax(p2_list).cpu().numpy().astype(bool)
            for bb in range(B):
                subj = str(batch["subject"][bb])
                tp2  = str(batch["tp2"][bb])
                base_dir = os.path.join(save_all_masks_dir, subj, tp2)
                for k in range(K):
                    save_binary_png(ph[bb, k], os.path.join(base_dir, f"pred_c{k}.png"))
                    save_binary_png(gh[bb, k], os.path.join(base_dir, f"gt_c{k}.png"))
                if tumor_idx:
                    pred_tum = np.any(ph[bb, tumor_idx], axis=0)
                    gt_tum   = np.any(gh[bb, tumor_idx], axis=0)
                    save_binary_png(pred_tum, os.path.join(base_dir, f"pred_tumor_union.png"))
                    save_binary_png(gt_tum,   os.path.join(base_dir, f"gt_tumor_union.png"))
                ana_idx = get_anatomy_indices(K, tumor_idx)
                if ana_idx:
                    pred_ana = np.any(ph[bb, ana_idx], axis=0)
                    gt_ana   = np.any(gh[bb, ana_idx], axis=0)
                    save_binary_png(pred_ana, os.path.join(base_dir, f"pred_anatomy_union.png"))
                    save_binary_png(gt_ana,   os.path.join(base_dir, f"gt_anatomy_union.png"))

    out = {
        "dice_tumor_mean": float(np.mean(tum_dice_list)) if tum_dice_list else 0.0,
        "dice_tumor_std": float(np.std(tum_dice_list) + 1e-12) if tum_dice_list else 0.0,
        "hd95_tumor_mean": float(np.mean(tum_hd95_list)) if tum_hd95_list else 0.0,
        "hd95_tumor_std": float(np.std(tum_hd95_list) + 1e-12) if tum_hd95_list else 0.0,
        "dice_anatomy_mean": float(np.mean(ana_dice_list)) if ana_dice_list else 0.0,
        "dice_anatomy_std": float(np.std(ana_dice_list) + 1e-12) if ana_dice_list else 0.0,
        "hd95_anatomy_mean": float(np.mean(ana_hd95_list)) if ana_hd95_list else 0.0,
        "hd95_anatomy_std": float(np.std(ana_hd95_list) + 1e-12) if ana_hd95_list else 0.0,
        "per_samples": per_samples
    }
    if report_macro:
        out.update({
            "dice_macro_mean": float(np.mean(mac_dice_list)) if mac_dice_list else 0.0,
            "dice_macro_std": float(np.std(mac_dice_list) + 1e-12) if mac_dice_list else 0.0,
            "hd95_macro_mean": float(np.mean(mac_hd95_list)) if mac_hd95_list else 0.0,
            "hd95_macro_std": float(np.std(mac_hd95_list) + 1e-12) if mac_hd95_list else 0.0,
        })
    return out

# ============================================================
# Train one fold
# ============================================================
def train_one_fold(train_ds: Dataset, val_ds: Dataset, K: int, tumor_idx: List[int],
                   device: torch.device, args) -> Dict[str, float]:

    if args.use_growth_cnn and args.growth_includes_image and args.growth_image_ch == 0:
        try:
            ex = train_ds[0] if isinstance(train_ds, Dataset) else train_ds.dataset[train_ds.indices[0]]
            gf = ex.get("growth_feats", None)
            if isinstance(gf, torch.Tensor):
                args.growth_image_ch = int(gf.shape[0])
                print(f"[Info] Auto-detected growth_image_ch={args.growth_image_ch}")
        except Exception:
            pass

    model = CrossDiffusionIMEX2D(
        K,
        PDEParams(
            dt=args.dt, steps=args.steps, jacobi_iters=args.jacobi_iters,
            jacobi_omega=args.jacobi_omega, D=args.D, chi=args.chi,
            k_tumor=args.k_tumor, cap_tumor=args.cap_tumor,
            D_max=args.D_max, chi_max=args.chi_max
        ),
        use_growth_cnn=args.use_growth_cnn,
        k_max=args.k_max,
        k_mode=args.k_mode,
        growth_in_ch=(K if not args.growth_includes_image else K + args.growth_image_ch)
    ).to(device)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=max(args.lr*0.1, 1e-5))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    tr_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True, drop_last=False,
                           collate_fn=collate_pairs)
    va_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True, drop_last=False,
                           collate_fn=collate_pairs)

    best_val = None
    running_losses, running_dice_tum, running_dice_ana = [], [], []
    ana_indices = get_anatomy_indices(K, tumor_idx)

    for epoch in range(1, args.epochs+1):
        model.train()
        running_losses.clear(); running_dice_tum.clear(); running_dice_ana.clear()

        for batch in tr_loader:
            p1 = batch['p_t1'].to(device, non_blocking=True)
            p2 = batch['p_t2'].to(device, non_blocking=True)
            growth_feats = batch.get('growth_feats', None)
            if isinstance(growth_feats, torch.Tensor):
                growth_feats = growth_feats.to(device, non_blocking=True)

            B  = p1.size(0)
            loss_total = torch.zeros((), device=device)

            for bb in range(B):
                dt_days = batch["delta_t"][bb]
                base_steps = args.steps if epoch > args.curriculum_epochs else args.steps_warmup
                steps_here = base_steps if (dt_days is None) else max(
                    1, int(np.ceil(float(dt_days) * args.steps_per_day * (base_steps / max(1, args.steps))))
                )

                with torch.cuda.amp.autocast(enabled=args.amp):
                    pred_bb = model(p1[bb:bb+1], tumor_idx=tumor_idx, steps_override=steps_here,
                                    growth_feats=(growth_feats[bb:bb+1] if isinstance(growth_feats, torch.Tensor) else None))
                    pred_bb = ensure_finite(pred_bb)
                    targ_bb = torch.nan_to_num(p2[bb:bb+1]).clamp_min(0)

                    loss_bb = dice_loss_multiclass(
                        torch.softmax(torch.log(pred_bb + 1e-8), dim=1),
                        targ_bb, class_idxs=tumor_idx
                    )

                    if args.use_growth_cnn and model.last_kmap is not None:
                        tv = tv_loss_map(model.last_kmap)
                        l2 = (model.last_kmap**2).mean()
                        loss_bb = loss_bb + args.lambda_tv * tv + args.lambda_k_l2 * l2

                    if args.lambda_anatomy > 0 and len(ana_indices) > 0:
                        loss_bb = loss_bb + args.lambda_anatomy * anatomy_consistency(
                            pred_bb, p1[bb:bb+1], targ_bb, ana_indices
                        )

                loss_total = loss_total + loss_bb

                with torch.no_grad():
                    tum_dice = dice_subset_from_probs(pred_bb, targ_bb, tumor_idx)
                    running_dice_tum.append(tum_dice)
                    if len(ana_indices) > 0:
                        ana_dice = dice_subset_from_probs(pred_bb, targ_bb, ana_indices)
                        running_dice_ana.append(ana_dice)

            loss = loss_total / max(1, B)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(opt); scaler.update()
            running_losses.append(loss.item())

        sched.step()

        val_res = evaluate_and_save(
            model, va_loader, K, device,
            out_fold_dir=None, show=args.show, viz_n=args.viz_n,
            steps_per_day=args.steps_per_day, default_steps=args.steps,
            tumor_idx=tumor_idx, report_macro=args.report_macro,
            anatomy_idx=ana_indices, save_all_masks_dir=None
        )
        tr_loss = float(np.mean(running_losses)) if running_losses else 0.0
        tr_dice_tum = float(np.mean(running_dice_tum)) if running_dice_tum else 0.0
        tr_dice_ana = float(np.mean(running_dice_ana)) if running_dice_ana else 0.0

        with torch.no_grad():
            D_all = (F.softplus(model.D_raw) + 1e-8).clamp_max(model.params.D_max).detach().cpu().numpy()
            chi   = float((F.softplus(model.chi_raw) + 1e-8).clamp_max(model.params.chi_max).item())
            k_t   = float((F.softplus(model.k_tumor_raw) + 1e-8).clamp_max(model.k_max).item())
            cap_t = float(torch.sigmoid(model.cap_tumor_logit).clamp(1e-3, 0.95).item())
            km_mu = km_sd = None
            if model.last_kmap is not None:
                km_mu = float(model.last_kmap.mean().item())
                km_sd = float(model.last_kmap.std().item())

        if args.report_macro:
            print(
              f"  Epoch {epoch:03d} | "
              f"train_loss={tr_loss:.4f} | "
              f"train_DSC_tum={tr_dice_tum:.4f} | train_DSC_ana={tr_dice_ana:.4f} | "
              f"val_DSC_tum={val_res['dice_tumor_mean']:.4f}±{val_res['dice_tumor_std']:.4f} | "
              f"val_DSC_ana={val_res['dice_anatomy_mean']:.4f}±{val_res['dice_anatomy_std']:.4f} | "
              f"val_HD95_tum={val_res['hd95_tumor_mean']:.2f}±{val_res['hd95_tumor_std']:.2f} | "
              f"val_HD95_ana={val_res['hd95_anatomy_mean']:.2f}±{val_res['hd95_anatomy_std']:.2f} | "
              f"val_DSC_mac={val_res['dice_macro_mean']:.4f}±{val_res['dice_macro_std']:.4f} | "
              f"val_HD95_mac={val_res['hd95_macro_mean']:.2f}±{val_res['hd95_macro_std']:.2f}"
            )
        else:
            print(
              f"  Epoch {epoch:03d} | "
              f"train_loss={tr_loss:.4f} | "
              f"train_DSC_tum={tr_dice_tum:.4f} | train_DSC_ana={tr_dice_ana:.4f} | "
              f"val_DSC_tum={val_res['dice_tumor_mean']:.4f}±{val_res['dice_tumor_std']:.4f} | "
              f"val_DSC_ana={val_res['dice_anatomy_mean']:.4f}±{val_res['dice_anatomy_std']:.4f} | "
              f"val_HD95_tum={val_res['hd95_tumor_mean']:.2f}±{val_res['hd95_tumor_std']:.2f} | "
              f"val_HD95_ana={val_res['hd95_anatomy_mean']:.2f}±{val_res['hd95_anatomy_std']:.2f}"
            )

        print(f"    [diag] D_per_class={np.round(D_all,4)} | chi={chi:.4f} | k={k_t:.4f} | cap={cap_t:.3f} "
              f"| k_map μ/σ={km_mu:.4f}/{km_sd:.4f}" if km_mu is not None else "")

        key = val_res['dice_tumor_mean']
        if best_val is None or key > best_val['dice_tumor_mean']:
            best_val = {"dice_tumor_mean": key, "epoch": epoch}

    res = evaluate_and_save(
        model, va_loader, K, device,
        out_fold_dir=args._fold_dir, show=args.show, viz_n=args.viz_n,
        steps_per_day=args.steps_per_day, default_steps=args.steps,
        tumor_idx=tumor_idx, report_macro=args.report_macro,
        anatomy_idx=ana_indices,
        save_all_masks_dir=os.path.join(args._fold_dir, "masks")
    )

    with torch.no_grad():
        D_all = (F.softplus(model.D_raw) + 1e-8).clamp_max(model.params.D_max).detach().cpu().numpy()
        chi   = float((F.softplus(model.chi_raw) + 1e-8).clamp_max(model.params.chi_max).item())
        k_t   = float((F.softplus(model.k_tumor_raw) + 1e-8).clamp_max(model.k_max).item())
        cap_t = float(torch.sigmoid(model.cap_tumor_logit).clamp(1e-3, 0.95).item())

    res["learned_params"] = {
        "D_per_class": D_all.tolist(),
        "chi": chi,
        "k_tumor": k_t,
        "cap_tumor": cap_t,
        "best_epoch": best_val["epoch"] if best_val else None
    }
    return res

def parse_args():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument('--data_root', type=str, default='.')
    ap.add_argument('--images_csv', type=str, default='metadata_images.csv')
    ap.add_argument('--patients_csv', type=str, default='metadata_patients.csv')
    ap.add_argument('--pair_mode', type=str, choices=['consecutive','pre_to_last'], default='consecutive')
    ap.add_argument('--K', type=int, default=5, help='Fixed K=5: [tumor, anat1, anat2, anat3, background]')
    ap.add_argument('--grid', type=int, nargs=2, default=[96,96])
    ap.add_argument('--num_workers', type=int, default=4)

    # Tumor indices
    ap.add_argument('--tumor_idx', type=int, nargs='+', default=None,
                    help='Defaults to [0] (tumor channel).')

    # Model / Growth encoder
    ap.add_argument('--use_growth_cnn', action='store_true', default=True)
    ap.add_argument('--k_mode', type=str, choices=['scalar','tumor_map'], default='tumor_map')
    ap.add_argument('--k_max', type=float, default=3.0)
    ap.add_argument('--growth_includes_image', action='store_true',
                   help='Concat dataset["growth_feats"] (if present) to p0 for GrowthCNN.')
    ap.add_argument('--growth_image_ch', type=int, default=0,
                   help='If 0, will try to auto-detect from first sample when using growth_includes_image.')

    # PDE
    ap.add_argument('--dt', type=float, default=0.1)
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--steps_per_day', type=float, default=1.0)
    ap.add_argument('--jacobi_iters', type=int, default=8)
    ap.add_argument('--jacobi_omega', type=float, default=0.9)
    ap.add_argument('--D', type=float, default=0.5)
    ap.add_argument('--chi', type=float, default=0.5)
    ap.add_argument('--k_tumor', type=float, default=0.5)
    ap.add_argument('--cap_tumor', type=float, default=0.85)
    ap.add_argument('--D_max', type=float, default=2.0)
    ap.add_argument('--chi_max', type=float, default=2.0)

    # Train
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch_size', type=int, default=6)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--amp', action='store_true')

    # Curriculum
    ap.add_argument('--curriculum_epochs', type=int, default=10)
    ap.add_argument('--steps_warmup', type=int, default=4)

    # Regularization
    ap.add_argument('--lambda_tv', type=float, default=1e-4)
    ap.add_argument('--lambda_k_l2', type=float, default=1e-5)
    ap.add_argument('--lambda_anatomy', type=float, default=0.0)

    # Eval/logging
    ap.add_argument('--report_macro', action='store_true')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--out_dir', type=str, default='runs/cv_out')

    # Viz
    ap.add_argument('--viz_n', type=int, default=6)
    ap.add_argument('--show', action='store_true')

    return ap.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)
    ensure_dir(args.out_dir)
    save_json(vars(args), os.path.join(args.out_dir, "run_config.json"))

    # Build dataset with fixed K=5 (dataset already outputs 5 channels)
    ds = make_metadata_dataset(
        data_root=args.data_root,
        images_csv=args.images_csv,
        patients_csv=args.patients_csv if args.patients_csv else None,
        K=None,                               # <-- fixed
        grid=tuple(args.grid),
        pair_mode=args.pair_mode,
        strict=True
    )
    args.K = 5
    print(f"[Info] Using fixed K={args.K} = [tumor, anat1, anat2, anat3, background]")

    # Tumor index default -> channel 0
    if args.tumor_idx is None:
        args.tumor_idx = [0]
    print(f"[Info] tumor_idx={args.tumor_idx}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Samples: {len(ds)} | K={args.K}")

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    idxs = np.arange(len(ds))
    fold_summaries = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(idxs), 1):
        args._fold_dir = os.path.join(args.out_dir, f"fold{fold:02d}")
        ensure_dir(args._fold_dir)

        tr_ds, va_ds = Subset(ds, tr_idx.tolist()), Subset(ds, va_idx.tolist())
        print(f"\n=== Fold {fold}/{args.folds} ===  (train={len(tr_ds)}, val={len(va_ds)})")

        res = train_one_fold(tr_ds, va_ds, K=args.K, tumor_idx=args.tumor_idx,
                             device=device, args=args)

        per_csv = os.path.join(args._fold_dir, "metrics_per_sample.csv")
        with open(per_csv, "w", newline="") as f:
            fields = ["subject","tp1","tp2","delta_t","dice_tumor","hd95_tumor","dice_anatomy","hd95_anatomy"]
            if args.report_macro:
                fields += ["dice_macro","hd95_macro"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(res["per_samples"])

        save_json(res["learned_params"], os.path.join(args._fold_dir, "params_learned.json"))

        if args.report_macro:
            print(f"Fold {fold}: "
                  f"DSC_tum={res['dice_tumor_mean']:.4f}±{res['dice_tumor_std']:.4f} | "
                  f"HD95_tum={res['hd95_tumor_mean']:.2f}±{res['hd95_tumor_std']:.2f} | "
                  f"DSC_ana={res['dice_anatomy_mean']:.4f}±{res['dice_anatomy_std']:.4f} | "
                  f"HD95_ana={res['hd95_anatomy_mean']:.2f}±{res['hd95_anatomy_std']:.2f} | "
                  f"DSC_mac={res['dice_macro_mean']:.4f}±{res['dice_macro_std']:.4f} | "
                  f"HD95_mac={res['hd95_macro_mean']:.2f}±{res['hd95_macro_std']:.2f}")
        else:
            print(f"Fold {fold}: "
                  f"DSC_tum={res['dice_tumor_mean']:.4f}±{res['dice_tumor_std']:.4f} | "
                  f"HD95_tum={res['hd95_tumor_mean']:.2f}±{res['hd95_tumor_std']:.2f} | "
                  f"DSC_ana={res['dice_anatomy_mean']:.4f}±{res['dice_anatomy_std']:.4f} | "
                  f"HD95_ana={res['hd95_anatomy_mean']:.2f}±{res['hd95_anatomy_std']:.2f}")

        fold_summaries.append({
            "fold": fold,
            "dice_tumor_mean": res["dice_tumor_mean"], "dice_tumor_std": res["dice_tumor_std"],
            "hd95_tumor_mean": res["hd95_tumor_mean"], "hd95_tumor_std": res["hd95_tumor_std"],
            "dice_anatomy_mean": res["dice_anatomy_mean"], "dice_anatomy_std": res["dice_anatomy_std"],
            "hd95_anatomy_mean": res["hd95_anatomy_mean"], "hd95_anatomy_std": res["hd95_anatomy_std"],
            **({
                "dice_macro_mean": res["dice_macro_mean"], "dice_macro_std": res["dice_macro_std"],
                "hd95_macro_mean": res["hd95_macro_mean"], "hd95_macro_std": res["hd95_macro_std"],
            } if args.report_macro else {})
        })

    cv_csv = os.path.join(args.out_dir, "cv_summary.csv")
    fields = ["fold","dice_tumor_mean","dice_tumor_std","hd95_tumor_mean","hd95_tumor_std",
              "dice_anatomy_mean","dice_anatomy_std","hd95_anatomy_mean","hd95_anatomy_std"]
    if args.report_macro:
        fields += ["dice_macro_mean","dice_macro_std","hd95_macro_mean","hd95_macro_std"]
    with open(cv_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(fold_summaries)

    d_means = [s["dice_tumor_mean"] for s in fold_summaries]
    h_means = [s["hd95_tumor_mean"] for s in fold_summaries]
    a_means = [s["dice_anatomy_mean"] for s in fold_summaries]
    ah_means = [s["hd95_anatomy_mean"] for s in fold_summaries]
    print("\n=== CV Summary (tumor) ===")
    print(f"DSC:  mean={np.mean(d_means):.4f}, std={np.std(d_means):.4f}")
    print(f"HD95: mean={np.mean(h_means):.2f}, std={np.std(h_means):.2f}")
    print("\n=== CV Summary (anatomy) ===")
    print(f"DSC:  mean={np.mean(a_means):.4f}, std={np.std(a_means):.4f}")
    print(f"HD95: mean={np.mean(ah_means):.2f}, std={np.std(ah_means):.2f}")
    if args.report_macro:
        md = [s["dice_macro_mean"] for s in fold_summaries]
        mh = [s["hd95_macro_mean"] for s in fold_summaries]
        print("\n=== CV Summary (macro) ===")
        print(f"DSC:  mean={np.mean(md):.4f}, std={np.std(md):.4f}")
        print(f"HD95: mean={np.mean(mh):.2f}, std={np.std(mh):.2f}")
    print(f"\nOutputs saved under: {os.path.abspath(args.out_dir)}")

if __name__ == "__main__":
    main()

# python3 train.py --data_root /path/to/UCSF_DT --images_csv metadata_images.csv --patients_csv metadata_patients.csv --grid 96 96 --epochs 50 --batch_size 8 --lr 1e-2 --K 5 --tumor_idx 0 --k_mode scalar --k_max 0.08 --D_max 0.08 --chi_max 0.05 --D 0.02 --chi 0.01 --dt 0.03 --steps 6 --steps_per_day 0.1 --jacobi_iters 12 --lambda_tv 0 --lambda_k_l2 0 --lambda_anatomy 0 --curriculum_epochs 100 --steps_warmup 2 --report_macro --out_dir runs/ucsf_multi_test1
