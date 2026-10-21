import glob
import hashlib
import json
import os
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.data import MetaTensor
import monai.transforms as mt
from torch.utils.data import Dataset

# CT-FM was pretrained with a wide HU window (see the project-lighter/ct_fm_feature_extractor model card).
PRETRAIN_HU_WINDOW = (-1024.0, 2048.0)
CACHE_VERSION = "v2"


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def compute_roi_slices(shape, zooms, masks, margin_mm):
    """Bounding box (native voxel space) of the union of `masks`, padded by `margin_mm` on every axis.

    Margins are specified in millimetres so the physical context is identical across scans with
    different voxel spacing / axis orientation. Returns None if all masks are empty.
    """
    union = np.zeros(shape, dtype=bool)
    for m in masks:
        union |= m
    coords = np.argwhere(union)
    if len(coords) == 0:
        return None
    margin = np.ceil(margin_mm / np.asarray(zooms[:3], dtype=float)).astype(int)
    lo = np.maximum(coords.min(axis=0) - margin, 0)
    hi = np.minimum(coords.max(axis=0) + margin + 1, shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def load_roi_volume(image_path, mask_path, lesion_path=None, margin_mm=20.0):
    """Load only the ROI of a CT as a MetaTensor (1, X, Y, Z) that keeps the correct affine.

    The ROI is the union of the pancreas mask and (if given) the lesion mask: in PanTS the lesion
    is not always contained in `pancreas.nii.gz`, so the pancreas mask alone can cut tumours off.
    """
    ct = nib.load(image_path)
    pan = nib.load(mask_path)
    if ct.shape != pan.shape:
        raise ValueError(f"CT/mask shape mismatch: {image_path} {ct.shape} vs {pan.shape}")
    masks = [np.asanyarray(pan.dataobj) > 0]
    if lesion_path and os.path.exists(lesion_path):
        masks.append(np.asanyarray(nib.load(lesion_path).dataobj) > 0)
    zooms = nib.affines.voxel_sizes(ct.affine)
    slices = compute_roi_slices(ct.shape, zooms, masks, margin_mm)
    if slices is None:
        raise ValueError(f"Empty pancreas mask, cannot define ROI: {mask_path}")
    sub = ct.slicer[slices]  # nibabel updates the affine for the crop
    arr = sub.get_fdata(dtype=np.float32)
    return MetaTensor(torch.from_numpy(arr)[None], affine=torch.as_tensor(sub.affine, dtype=torch.float64))


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def get_preprocess_transforms(spacing=2.5, spatial_size=(80, 96, 96), hu_window=PRETRAIN_HU_WINDOW):
    """Deterministic preprocessing (cacheable). Must be applied to a MetaTensor carrying the affine.

    Returns (geometry, finish): geometry = Orientation(SPL) + isotropic resampling,
    finish = HU windowing + pad/crop to `spatial_size` (padding value 0 == air after windowing).
    """
    geometry = mt.Compose([
        mt.Orientation(axcodes="SPL"),
        mt.Spacing(pixdim=(spacing,) * 3, mode="bilinear", padding_mode="border", dtype=np.float32),
    ])
    finish = mt.Compose([
        mt.ScaleIntensityRange(a_min=hu_window[0], a_max=hu_window[1], b_min=0.0, b_max=1.0, clip=True),
        mt.ResizeWithPadOrCrop(spatial_size=list(spatial_size)),
    ])
    return geometry, finish


def get_augment_transforms():
    """Random augmentation, applied AFTER the cache so that it is different every epoch."""
    return mt.Compose([
        mt.RandRotate(range_x=0.1, range_y=0.1, range_z=0.1, mode="bilinear", prob=0.3),
    ])


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class PanTSDataset(Dataset):
    def __init__(self, data_dir, label_dir, split_csv=None, metadata_path=None, cache_dir=None,
                 augment=False, margin_mm=20.0, spacing=2.5, spatial_size=(80, 96, 96),
                 hu_window=PRETRAIN_HU_WINDOW):
        self.data_dir = str(data_dir)
        self.label_dir = str(label_dir)
        self.augment = get_augment_transforms() if augment else None
        self.margin_mm = margin_mm
        self.spatial_size = tuple(spatial_size)
        self.geometry, self.finish = get_preprocess_transforms(spacing, spatial_size, hu_window)

        params = dict(version=CACHE_VERSION, margin_mm=margin_mm, spacing=spacing,
                      spatial_size=list(spatial_size), hu_window=list(hu_window))
        self.cache_key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
        self.cache_dir = os.path.join(str(cache_dir), self.cache_key) if cache_dir else None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        df = self._read_table(split_csv if split_csv is not None else metadata_path)
        df = df[df["tumor?"].notna()]

        self.samples, missing = [], []
        for case_id, label in zip(df["PanTS ID"], df["tumor?"]):
            nii_files = glob.glob(os.path.join(self.data_dir, case_id, "ct.nii*"))
            seg_dir = os.path.join(self.label_dir, case_id, "segmentations")
            mask_path = os.path.join(seg_dir, "pancreas.nii.gz")
            if not nii_files or not os.path.exists(mask_path):
                missing.append(case_id)
                continue
            self.samples.append({
                "id": case_id,
                "image_path": nii_files[0],
                "mask_path": mask_path,
                "lesion_path": os.path.join(seg_dir, "pancreatic_lesion.nii.gz"),
                "label": int(label),
            })

        if missing:
            warnings.warn(f"{len(missing)} cases missing CT or pancreas mask and were skipped: {missing}")
        num_pos = sum(s["label"] for s in self.samples)
        print(f"Loaded {len(self.samples)}/{len(df)} cases | tumor: {num_pos}, normal: {len(self.samples) - num_pos}"
              f" | cache key {self.cache_key}")

    @staticmethod
    def _read_table(source):
        if source is None:
            raise ValueError("Either split_csv or metadata_path must be provided.")
        if isinstance(source, pd.DataFrame):
            return source
        source = str(source)
        if source.endswith(".csv"):
            return pd.read_csv(source)
        if source.endswith(".xlsx"):
            return pd.read_excel(source)
        raise ValueError(f"Unsupported table format: {source}")

    def __len__(self):
        return len(self.samples)

    def _preprocess(self, item):
        vol = load_roi_volume(item["image_path"], item["mask_path"], item["lesion_path"], self.margin_mm)
        vol = self.geometry(vol)
        too_big = [i for i, (s, t) in enumerate(zip(vol.shape[1:], self.spatial_size)) if s > t]
        if too_big:
            warnings.warn(f"{item['id']}: ROI {tuple(vol.shape[1:])} exceeds spatial_size {self.spatial_size} "
                          f"on axes {too_big}; ROI will be centre-cropped (possible loss of tumour).")
        return self.finish(vol).as_tensor().clone().float()

    def __getitem__(self, idx):
        item = self.samples[idx]
        case_id = item["id"]

        cache_file = os.path.join(self.cache_dir, f"{case_id}.pt") if self.cache_dir else None
        if cache_file and os.path.exists(cache_file):
            tensor = torch.load(cache_file, weights_only=True)
        else:
            tensor = self._preprocess(item)
            if cache_file:
                torch.save(tensor, cache_file)

        if self.augment is not None:
            tensor = torch.as_tensor(self.augment(tensor), dtype=torch.float32).clone()
        return tensor, torch.tensor(item["label"], dtype=torch.long), case_id


if __name__ == "__main__":
    from config import CT_DIR, LABEL_DIR, SPLIT_DIR, CACHE_DIR

    dataset = PanTSDataset(CT_DIR, LABEL_DIR, split_csv=SPLIT_DIR / "train_split.csv", cache_dir=CACHE_DIR)
    img, lbl, cid = dataset[0]
    print(f"Sample: ID={cid}, Label={lbl}, Tensor Shape={tuple(img.shape)}")
