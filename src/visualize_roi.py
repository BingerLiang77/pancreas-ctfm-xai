import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from config import CT_DIR, LABEL_DIR
from dataset import PRETRAIN_HU_WINDOW, compute_roi_slices


def window_hu(img, w_min, w_max):
    return np.clip((img - w_min) / (w_max - w_min), 0.0, 1.0)


def load_canonical(path):
    """Load in RAS orientation so the axial plane is always [:, :, z], regardless of how the scan was stored."""
    return nib.as_closest_canonical(nib.load(path))


def visualize_comparison(data_dir, label_dir, pos_id, neg_id, out_path, margin_mm=20.0,
                         hu_window=(-135.0, 215.0)):
    """QC figure. Uses the SAME ROI logic as the dataset (mm margins, pancreas ∪ lesion).
    A soft-tissue window is used for display only (the model input window is PRETRAIN_HU_WINDOW)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor="white")
    cases = [(neg_id, "Normal (0)", False, 0), (pos_id, "Tumor (1)", True, 1)]

    for cid, _, is_pos, row in cases:
        ct_path = glob.glob(os.path.join(data_dir, cid, "ct.nii*"))[0]
        seg = os.path.join(label_dir, cid, "segmentations")
        ct_nii = load_canonical(ct_path)
        ct = np.asanyarray(ct_nii.dataobj).astype(np.float32)

        def read(name):
            p = os.path.join(seg, name)
            return (np.asanyarray(load_canonical(p).dataobj) > 0) if os.path.exists(p) else np.zeros(ct.shape, bool)

        pan, lesion = read("pancreas.nii.gz"), read("pancreatic_lesion.nii.gz")
        zooms = nib.affines.voxel_sizes(ct_nii.affine)
        sl = compute_roi_slices(ct.shape, zooms, [pan, lesion], margin_mm)
        if sl is None:
            raise ValueError(f"{cid}: empty pancreas and lesion masks")

        ref = lesion if (is_pos and lesion.any()) else pan
        best_z = int(np.argmax(ref.sum(axis=(0, 1))))
        z_crop = best_z - sl[2].start

        full = np.rot90(window_hu(ct[:, :, best_z], *hu_window))
        crop = ct[sl]
        roi = np.rot90(window_hu(crop[:, :, z_crop], *hu_window))
        m_pan = np.rot90(pan[sl][:, :, z_crop])
        m_les = np.rot90(lesion[sl][:, :, z_crop])
        kept = 100.0 * lesion[sl].sum() / lesion.sum() if lesion.any() else float("nan")
        frac = 100.0 * crop.size / ct.size

        axes[row, 0].imshow(full, cmap="gray")
        axes[row, 0].set_title(f"[{cid}] Full CT (z={best_z})", fontsize=13, fontweight="bold")
        axes[row, 1].imshow(roi, cmap="gray")
        axes[row, 1].set_title(f"ROI crop: {crop.shape} vox = {frac:.1f}% of volume", fontsize=13,
                               fontweight="bold", color="navy")
        axes[row, 2].imshow(roi, cmap="gray")
        if m_pan.any():
            axes[row, 2].imshow(np.ma.masked_where(~m_pan, m_pan), cmap="Greens", alpha=0.55, vmin=0, vmax=1)
        if m_les.any():
            axes[row, 2].imshow(np.ma.masked_where(~m_les, m_les), cmap="Reds", alpha=0.75, vmin=0, vmax=1)
        title = "Green: pancreas"
        if lesion.any():
            title += f" | Red: lesion (volume retained in ROI: {kept:.0f}%)"
        axes[row, 2].set_title(title, fontsize=13, fontweight="bold")
        for c in range(3):
            axes[row, c].axis("off")

    plt.suptitle("Pancreas-centred ROI extraction (RAS display)", fontsize=17, fontweight="heavy", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Visualization comparison saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(CT_DIR))
    parser.add_argument("--label_dir", default=str(LABEL_DIR))
    parser.add_argument("--pos_id", default="PanTS_00000029")
    parser.add_argument("--neg_id", default="PanTS_00000023")
    parser.add_argument("--margin_mm", type=float, default=20.0)
    parser.add_argument("--out_path", default="pancreas_roi_comparison.png")
    a = parser.parse_args()
    visualize_comparison(a.data_dir, a.label_dir, a.pos_id, a.neg_id, a.out_path, a.margin_mm)
