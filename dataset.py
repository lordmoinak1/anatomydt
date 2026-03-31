from __future__ import annotations
import csv, os, re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import nibabel as nib
except Exception:
    nib = None

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


_TIME_ORDER = {
    "pre": 0, "baseline": 0, "t0": 0,
    "post1": 1, "t1": 1,
    "post2": 2, "t2": 2,
    "post3": 3, "t3": 3, "post4": 4, "t4": 4, "post5": 5, "t5": 5,
}

COL_PRE_TO_POST1 = "Days from 1st surgery/DX  to 1st scan"  # pre -> post1
COL_POST1_TO_POST2 = "Days from 1st scan to 2nd scan"       # post1 -> post2

def _read_csv(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({(k or "").strip(): (v or "").strip() for k, v in row.items()})
    return rows

def _parse_listish(s: str) -> List[str]:
    """Parse list-like strings: JSON-ish, parentheses, or delimited with | , ;"""
    s = (s or "").strip()
    if not s:
        return []
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            arr = eval(s.replace("(", "[").replace(")", "]"))
            if isinstance(arr, (list, tuple)):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    # split by common delimiters
    for sep in ("|", ",", ";"):
        if sep in s:
            return [t.strip() for t in s.split(sep) if t.strip()]
    return [s]

def _resolve_path(root: str, p: str) -> str:
    """Resolve possibly messy relative path against data_root."""
    if not p:
        return p
    if os.path.isabs(p) and os.path.exists(p):
        return p
    cand = os.path.join(root, p)
    if os.path.exists(cand):
        return cand
    basename = os.path.basename(p)
    for dirpath, _, files in os.walk(root):
        if basename in files:
            return os.path.join(dirpath, basename)
    return cand  # may not exist; caller decides strict handling

def _binarize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    if arr.max() <= 3:
        return (arr > 0).astype(np.float32)
    return (arr >= 1).astype(np.float32)

def _load_mask_tumor(path: str, to_size: Tuple[int, int]) -> torch.Tensor:
    """
    Load mask from image/npy/nii into FloatTensor[1,H,W] in {0,1}.
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Mask not found: {path}")

    lower = path.lower()
    if lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        img = Image.open(path)
        if img.size != (to_size[1], to_size[0]):  # PIL is (W,H)
            img = img.resize((to_size[1], to_size[0]), Image.NEAREST)
        arr = np.array(img)
        arr = _binarize(arr)
        return torch.from_numpy(arr)[None, ...]  # [1,H,W]

    if lower.endswith(".npy"):
        arr = np.load(path)
        if arr.ndim == 3:  # [H,W,Z] -> mid-slice
            z = arr.shape[-1] // 2
            arr = arr[..., z]
        arr = _binarize(arr)
        img = Image.fromarray((arr > 0.5).astype(np.uint8) * 255)
        if img.size != (to_size[1], to_size[0]):
            img = img.resize((to_size[1], to_size[0]), Image.NEAREST)
        return torch.from_numpy((np.array(img) > 127).astype(np.float32))[None, ...]

    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        if nib is None:
            raise RuntimeError("nibabel is required to read NIfTI files.")
        vol = np.asanyarray(nib.load(path).get_fdata())
        if vol.ndim == 3:
            z = vol.shape[-1] // 2
            arr = vol[..., z]
        else:
            arr = vol
        arr = _binarize(arr)
        img = Image.fromarray((arr > 0.5).astype(np.uint8) * 255)
        if img.size != (to_size[1], to_size[0]):
            img = img.resize((to_size[1], to_size[0]), Image.NEAREST)
        return torch.from_numpy((np.array(img) > 127).astype(np.float32))[None, ...]

    raise ValueError(f"Unsupported mask format: {path}")

def _load_mask_anatomy(path: str, to_size: Tuple[int, int]) -> torch.Tensor:
    """
    Load mask from image/npy/nii into FloatTensor[1,H,W] in {0,1}.
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Mask not found: {path}")

    lower = path.lower()
    if lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        img = Image.open(path)
        if img.size != (to_size[1], to_size[0]):  # PIL is (W,H)
            img = img.resize((to_size[1], to_size[0]), Image.NEAREST)
        
        arr = np.array(img)

        mask = np.zeros_like(arr, dtype=np.uint8)
        mask[np.isin(arr, [215, 254])] = 1
        mask[np.isin(arr, [214, 253])] = 2
        mask[np.isin(arr, [213, 252])] = 3

        # arr = _binarize(arr)
        return torch.from_numpy(mask)[None, ...]  # [1,H,W]

    if lower.endswith(".npy"):
        arr = np.load(path)
        if arr.ndim == 3:  # [H,W,Z] -> mid-slice
            z = arr.shape[-1] // 2
            arr = arr[..., z]
        arr = _binarize(arr)
        img = Image.fromarray((arr > 0.5).astype(np.uint8) * 255)
        if img.size != (to_size[1], to_size[0]):
            img = img.resize((to_size[1], to_size[0]), Image.NEAREST)
        return torch.from_numpy((np.array(img) > 127).astype(np.float32))[None, ...]

    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        if nib is None:
            raise RuntimeError("nibabel is required to read NIfTI files.")
        vol = np.asanyarray(nib.load(path).get_fdata())
        if vol.ndim == 3:
            z = vol.shape[-1] // 2
            arr = vol[..., z]
        else:
            arr = vol
        arr = _binarize(arr)
        img = Image.fromarray((arr > 0.5).astype(np.uint8) * 255)
        if img.size != (to_size[1], to_size[0]):
            img = img.resize((to_size[1], to_size[0]), Image.NEAREST)
        return torch.from_numpy((np.array(img) > 127).astype(np.float32))[None, ...]

    raise ValueError(f"Unsupported mask format: {path}")

def _load_multi(paths: List[str], to_size: Tuple[int, int]) -> torch.Tensor:
    """
    Load a list of binary masks -> FloatTensor [C,H,W] (C=len(paths)), stacked.
    Missing/empty list returns zero-channel tensor.
    """
    if not paths:
        return torch.zeros((0, to_size[0], to_size[1]), dtype=torch.float32)
    chans = []
    for p in paths:
        if 'synthseg' in p:
            t = _load_mask_anatomy(p, to_size)  # [1,H,W]
        else:
            t = _load_mask_tumor(p, to_size)  # [1,H,W]
        chans.append(t)
    return torch.cat(chans, dim=0)  # [C,H,W]

def _normalize_simplex(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    x: [C,H,W] non-negative. Normalize to simplex across C per-pixel.
    """
    x = x.clamp_min(0)
    s = x.sum(dim=0, keepdim=True).clamp_min(eps)
    return x / s

def _order_key(tp: str) -> int:
    tp_l = (tp or "").lower().strip()
    if tp_l in _TIME_ORDER:
        return _TIME_ORDER[tp_l]
    m = re.match(r"t(\d+)", tp_l)
    if m:
        return int(m.group(1))
    return 9999

def _to_int_safe(x: str | float | int | None) -> Optional[int]:
    if x is None:
        return None
    try:
        v = int(round(float(x)))
        return v
    except Exception:
        return None

@dataclass
class PairRow:
    subject: str
    tp1: str
    tp2: str
    tumor1: List[str]        # list of tumor mask paths @ tp1
    tumor2: List[str]        # list of tumor mask paths @ tp2
    anat1: List[str]         # list of anatomy mask paths @ tp1
    anat2: List[str]         # list of anatomy mask paths @ tp2
    delta_t: Optional[float]  # days
    day1: Optional[int]
    day2: Optional[int]

# Helpers to gather multi-channel columns with many naming styles
_TUMOR_KEYS = ["path_tumor", "tumor", "seg", "path_seg"]
_ANAT_KEYS  = ["path_anat", "anat", "path_t1c_synth", "path_t1c_anat"]

def _collect_multi_paths(r: Dict[str, str], keys: List[str]) -> List[str]:
    acc: List[str] = []
    # base keys and with numeric suffixes
    for k in list(keys):
        if k in r and r[k].strip():
            acc += _parse_listish(r[k])
        # numeric suffixes like key1, key_1, key02
        for suf in ["", "_"]:
            for i in range(1, 16):  # up to 15 channels
                kk = f"{k}{suf}{i}"
                if kk in r and r[kk].strip():
                    acc += _parse_listish(r[kk])
    # clean & unique while preserving order
    out = []
    seen = set()
    for p in acc:
        if p and p not in seen:
            out.append(p); seen.add(p)
    return out

def _build_pairs(
    img_rows: List[Dict[str, str]],
    pat_rows_by_id: Dict[str, Dict[str, str]],
    data_root: str,
    pair_mode: str = "consecutive"
) -> List[PairRow]:

    by_id: Dict[str, List[Dict[str, str]]] = {}
    for r in img_rows:
        sid = r.get("SubjectID", "").strip()
        if sid:
            by_id.setdefault(sid, []).append(r)

    pairs: List[PairRow] = []
    for sid, rows in by_id.items():
        def day_or_order(r):
            d = _to_int_safe(r.get("day"))
            return (0, d) if d is not None else (1, _order_key(r.get("timepoint", "")))
        rows_sorted = sorted(rows, key=day_or_order)
        if len(rows_sorted) < 2:
            continue

        if pair_mode == "pre_to_last":
            idx_pairs = [(0, len(rows_sorted) - 1)]
        else:
            idx_pairs = [(i, i + 1) for i in range(len(rows_sorted) - 1)]

        pinf = pat_rows_by_id.get(sid, {})

        def estimate_delta_and_days(r1: Dict[str, str], r2: Dict[str, str]) -> Tuple[Optional[float], Optional[int], Optional[int]]:
            d1 = _to_int_safe(r1.get("day"))
            d2 = _to_int_safe(r2.get("day"))
            if (d1 is not None) and (d2 is not None):
                return float(max(0, d2 - d1)), d1, d2

            o1, o2 = _order_key(r1.get("timepoint", "")), _order_key(r2.get("timepoint", ""))

            if o1 == 0 and o2 == 1:
                v = pinf.get(COL_PRE_TO_POST1, "")
                try: return (max(0.0, float(v)), None, None) if v != "" else (None, None, None)
                except Exception: return (None, None, None)
            if o1 == 1 and o2 == 2:
                v = pinf.get(COL_POST1_TO_POST2, "")
                try: return (max(0.0, float(v)), None, None) if v != "" else (None, None, None)
                except Exception: return (None, None, None)
            return (None, None, None)

        for i, j in idx_pairs:
            r1, r2 = rows_sorted[i], rows_sorted[j]

            # multi-channel tumor/anat paths
            tumor1_raw = _collect_multi_paths(r1, _TUMOR_KEYS)
            tumor2_raw = _collect_multi_paths(r2, _TUMOR_KEYS)
            anat1_raw  = _collect_multi_paths(r1, _ANAT_KEYS)
            anat2_raw  = _collect_multi_paths(r2, _ANAT_KEYS)

            # resolve on disk
            def _resolve_list(lst: List[str]) -> List[str]:
                return [_resolve_path(data_root, p) for p in lst]

            tumor1 = _resolve_list(tumor1_raw)
            tumor2 = _resolve_list(tumor2_raw)
            anat1  = _resolve_list(anat1_raw)
            anat2  = _resolve_list(anat2_raw)

            dt, d1, d2 = estimate_delta_and_days(r1, r2)

            pairs.append(PairRow(
                subject=sid,
                tp1=r1.get("timepoint", ""),
                tp2=r2.get("timepoint", ""),
                tumor1=tumor1, tumor2=tumor2,
                anat1=anat1,   anat2=anat2,
                delta_t=dt, day1=d1, day2=d2
            ))
    return pairs

class GliomaMetadataDataset(Dataset):
    """
    Yields paired timepoints with multi-channel tumor + multi-channel anatomy + background.

    Each sample:
      p_t1: [K,H,W] simplex
      p_t2: [K,H,W] simplex
      Optional: growth_feats [Cg,H,W] (NOT set here; add in your dataset if desired)
    """
    def __init__(
        self,
        pairs: List[PairRow],
        K: Optional[int] = None,                 # if None, inferred as Ct+Ca+1
        grid: Tuple[int, int] = (96, 96),
        strict: bool = True
    ):
        self.grid = grid
        self.strict = strict
        self.pairs: List[PairRow] = []

        # Filter pairs that have at least one mask at both timepoints
        for pr in pairs:
            ok1 = (len(pr.tumor1) + len(pr.anat1)) > 0
            ok2 = (len(pr.tumor2) + len(pr.anat2)) > 0
            if not (ok1 and ok2):
                continue
            if strict:
                # Ensure all listed paths exist
                allp = pr.tumor1 + pr.anat1 + pr.tumor2 + pr.anat2
                if not all(os.path.isfile(x) for x in allp):
                    continue
            self.pairs.append(pr)

        if len(self.pairs) == 0:
            raise RuntimeError("No valid subject pairs found. Check CSV paths and data_root.")

        # Infer K from the first pair (Ct+Ca+1)
        pr0 = self.pairs[0]
        Ct0 = max(1, len(pr0.tumor1)) if (len(pr0.tumor1) > 0) else max(1, len(pr0.tumor2))
        Ca0 = max(0, len(pr0.anat1)) if (len(pr0.anat1) > 0) else max(0, len(pr0.anat2))
        self.Ct = Ct0
        self.Ca = Ca0
        self.K = (K if K is not None else (self.Ct + self.Ca + 1))
        if self.K != (self.Ct + self.Ca + 1):
            raise ValueError(f"K mismatch: requested K={self.K} but inferred Ct+Ca+1={self.Ct+self.Ca+1}")

    def __len__(self) -> int:
        return len(self.pairs)

    def _build_simplex(self, tumor: torch.Tensor, anat: torch.Tensor) -> torch.Tensor:
        """
        tumor: [Ct,H,W], anat: [Ca,H,W]
        returns: [Ct+Ca+1, H, W] with background as last channel, simplex-normalized.
        """
        H, W = self.grid
        if tumor.ndim != 3 or (tumor.shape[1:] != (H, W)):
            raise ValueError("tumor must be [Ct,H,W]")
        if anat.ndim != 3 or (anat.shape[1:] != (H, W)):
            raise ValueError("anat must be [Ca,H,W]")

        # clamp to {0,1}
        tumor = tumor.clamp(0, 1)
        anat_x  = anat.clamp(0, 1)

        tumor = (tumor > 0.5)
        anat_x = (anat_x > 0.5)
        anat1  = (anat == 1)
        anat2  = (anat == 2)
        anat3  = (anat == 3)

        brain_wo_tumor = anat & (~tumor)
        anat1 = anat1 & (~tumor)
        anat2 = anat2 & (~tumor)
        anat3 = anat3 & (~tumor)
        background = ~(anat_x | tumor)

        x = torch.cat([tumor, anat1, anat2, anat3, background], dim=0)  # [K,H,W]
        x = x.clamp_min(0)
        # normalize to simplex to be safe (in case overlaps existed)
        x = _normalize_simplex(x)
        return x

    def __getitem__(self, idx: int):
        pr = self.pairs[idx]
        H, W = self.grid

        # load tumor/anat stacks for both t1 and t2
        tumor1 = _load_multi(pr.tumor1, (H, W))  # [Ct1,H,W]
        tumor2 = _load_multi(pr.tumor2, (H, W))  # [Ct2,H,W]
        anat1  = _load_multi(pr.anat1,  (H, W))  # [Ca1,H,W]
        anat2  = _load_multi(pr.anat2,  (H, W))  # [Ca2,H,W]

        # If channel counts differ across timepoints, pad with zeros to match Ct, Ca
        def _pad_to(x: torch.Tensor, C: int) -> torch.Tensor:
            c, h, w = x.shape
            if c == C: return x
            if c > C:  return x[:C]
            pad = x.new_zeros((C - c, h, w))
            return torch.cat([x, pad], dim=0)

        tumor1 = _pad_to(tumor1, self.Ct)
        tumor2 = _pad_to(tumor2, self.Ct)
        anat1  = _pad_to(anat1,  self.Ca)
        anat2  = _pad_to(anat2,  self.Ca)

        p1 = self._build_simplex(tumor1, anat1)  # [K,H,W]
        p2 = self._build_simplex(tumor2, anat2)  # [K,H,W]

        sample = {
            "p_t1": p1,
            "p_t2": p2,
            "subject": pr.subject,
            "tp1": pr.tp1,
            "tp2": pr.tp2,
            "delta_t": pr.delta_t,   # may be None if unknown
            "day1": pr.day1,
            "day2": pr.day2,
        }
        return sample

def make_metadata_dataset(
    data_root: str,
    images_csv: str = "metadata_images.csv",
    patients_csv: Optional[str] = "metadata_patients.csv",
    K: Optional[int] = None,                    # if None, infer as Ct+Ca+1
    grid: Tuple[int, int] = (96, 96),
    pair_mode: str = "consecutive",
    strict: bool = True
) -> GliomaMetadataDataset:
    img_csv_path = os.path.join(data_root, images_csv)
    if not os.path.isfile(img_csv_path):
        raise FileNotFoundError(f"Could not find {img_csv_path}")

    img_rows = _read_csv(img_csv_path)

    pat_by_id: Dict[str, Dict[str, str]] = {}
    if patients_csv:
        pat_csv_path = os.path.join(data_root, patients_csv)
        if os.path.isfile(pat_csv_path):
            for r in _read_csv(pat_csv_path):
                sid = r.get("SubjectID", "").strip()
                if sid:
                    pat_by_id[sid] = r

    pairs = _build_pairs(img_rows, pat_by_id, data_root, pair_mode=pair_mode)
    return GliomaMetadataDataset(pairs, K=K, grid=grid, strict=strict)

# -------------------------
# Smoke test
# -------------------------
if __name__ == "__main__":
    ds = make_metadata_dataset(
        data_root="/path/to/UCSF_DT",
        images_csv="metadata_images.csv",
        patients_csv="metadata_patients.csv",
        K=None, grid=(96, 96),
        pair_mode="consecutive"
    )
