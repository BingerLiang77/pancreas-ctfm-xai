"""Synthetic-data checks for the ROI / resampling pipeline. Run: python tests/test_preprocessing.py"""
import os
import sys
import tempfile

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import compute_roi_slices, get_preprocess_transforms, load_roi_volume  # noqa: E402


def make_case(tmp, affine, shape=(120, 100, 60)):
    ct = np.full(shape, -1000.0, dtype=np.float32)
    ct[10:110, 10:90, 5:55] = 40.0                      # "body"
    pan = np.zeros(shape, np.uint8); pan[40:60, 40:60, 20:30] = 1
    les = np.zeros(shape, np.uint8); les[70:76, 40:46, 25:28] = 1   # lesion OUTSIDE the pancreas mask
    ct[les > 0] = 2000.0                                  # marker visible only if lesion kept
    paths = {}
    for name, arr, dt in [("ct", ct, np.float32), ("pan", pan, np.uint8), ("les", les, np.uint8)]:
        p = os.path.join(tmp, f"{name}.nii.gz")
        nib.save(nib.Nifti1Image(arr.astype(dt), affine), p)
        paths[name] = p
    return paths


def test_union_roi_keeps_lesion_and_orientation_invariance():
    results = []
    # RAS 0.8/0.8/2.5 mm  and  flipped/permuted LPS-like 1.0 mm
    affines = [np.diag([0.8, 0.8, 2.5, 1.0]),
               np.array([[-1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])]
    geometry, finish = get_preprocess_transforms(spacing=2.5, spatial_size=(48, 64, 64))
    for aff in affines:
        with tempfile.TemporaryDirectory() as tmp:
            p = make_case(tmp, aff)
            vol = load_roi_volume(p["ct"], p["pan"], p["les"], margin_mm=10)
            vol = geometry(vol)
            out = finish(vol).as_tensor()
            assert out.shape == (1, 48, 64, 64), out.shape
            assert out.max() > 0.9, "lesion marker lost (ROI must be union of pancreas and lesion)"
            results.append(vol.shape[1:])
    print("ROI sizes after resampling per orientation:", results)


def test_spacing_is_physical():
    aff = np.diag([0.8, 0.8, 2.5, 1.0])
    geometry, _ = get_preprocess_transforms(spacing=2.5, spatial_size=(48, 64, 64))
    with tempfile.TemporaryDirectory() as tmp:
        p = make_case(tmp, aff)
        vol = load_roi_volume(p["ct"], p["pan"], p["les"], margin_mm=10)
        n_native = vol.shape[1:]
        out = geometry(vol)
        # axis 0 native (0.8mm) -> shrinks ~3.1x; z axis (2.5mm) unchanged
        assert abs(out.shape[-1] - n_native[0] * 0.8 / 2.5) <= 2, (out.shape, n_native)  # SPL: last axis is L (x)


def test_margin_in_mm():
    m = np.zeros((100, 100, 100), bool); m[50, 50, 50] = True
    sl = compute_roi_slices(m.shape, (1.0, 2.0, 5.0), [m], margin_mm=10)
    assert (sl[0].stop - sl[0].start, sl[1].stop - sl[1].start, sl[2].stop - sl[2].start) == (21, 11, 5)


if __name__ == "__main__":
    test_margin_in_mm()
    test_union_roi_keeps_lesion_and_orientation_invariance()
    test_spacing_is_physical()
    print("all preprocessing checks passed")
