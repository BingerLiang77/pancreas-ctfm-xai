import argparse
import glob
import os

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import CT_DIR, LABEL_DIR, METADATA_PATH, SPLIT_DIR

# Columns that are not needed downstream. Free-text radiology reports in particular can leak the label.
DROP_COLUMNS = ["structured report", "site detail"]


def mask_nonempty(label_dir, case_id, name):
    path = os.path.join(label_dir, case_id, "segmentations", name)
    return os.path.exists(path) and bool(np.asanyarray(nib.load(path).dataobj).any())


def ct_readable(path):
    """Detect truncated/corrupt volumes (e.g. incomplete iCloud sync) by reading the last slice."""
    try:
        np.asanyarray(nib.load(path).dataobj[..., -1:])
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  unreadable CT {path}: {type(e).__name__}")
        return False


def usable(case_id, tumor, data_dir, label_dir):
    """A case can be used if it has a readable CT and a non-empty ROI (pancreas mask; lesion counts for tumour cases)."""
    cts = glob.glob(os.path.join(data_dir, case_id, "ct.nii*"))
    if not cts or not ct_readable(cts[0]):
        return False
    if mask_nonempty(label_dir, case_id, "pancreas.nii.gz"):
        return True
    return bool(tumor) and mask_nonempty(label_dir, case_id, "pancreatic_lesion.nii.gz")


def generate_cohort(args):
    data_dir, label_dir, out_dir = str(args.data_dir), str(args.label_dir), str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_excel(args.metadata)
    df = df[df["tumor?"].notna()]
    valid_cases = {d for d in os.listdir(data_dir) if glob.glob(os.path.join(data_dir, d, "ct.nii*"))}
    df = df[df["PanTS ID"].isin(valid_cases)].copy()
    print(f"Total valid CT cases: {len(df)}")

    pos_df = df[df["tumor?"] == 1]
    neg_df = df[df["tumor?"] == 0]
    print(f"Total positive: {len(pos_df)}, Total negative: {len(neg_df)}")

    bad_pos = [i for i in pos_df["PanTS ID"] if not usable(i, 1, data_dir, label_dir)]
    if bad_pos:
        print(f"WARNING: positives without usable ROI dropped: {bad_pos}")
        pos_df = pos_df[~pos_df["PanTS ID"].isin(bad_pos)]

    # Stratified negative sampling by CT phase (proportional to the phase mix of all negatives).
    # Candidates with an empty pancreas mask are skipped and replaced by the next candidate of the same phase.
    n_neg = args.n_neg
    phase_share = neg_df["ct phase"].value_counts(normalize=True)
    quota = (phase_share * n_neg).round().astype(int)
    quota.iloc[0] += n_neg - quota.sum()
    picked = []
    for phase, n in quota.items():
        cands = neg_df[neg_df["ct phase"] == phase].sample(frac=1.0, random_state=args.seed)
        got = []
        for _, row in cands.iterrows():
            if len(got) == n:
                break
            if usable(row["PanTS ID"], 0, data_dir, label_dir):
                got.append(row)
            else:
                print(f"  skip {row['PanTS ID']} ({phase}): unusable (empty pancreas mask or unreadable CT)")
        if len(got) < n:
            print(f"WARNING: only {len(got)}/{n} usable negatives for phase {phase}")
        picked.extend(got)
    neg_sampled = pd.DataFrame(picked)

    cohort = pd.concat([pos_df, neg_sampled]).sort_values("PanTS ID").reset_index(drop=True)
    cohort = cohort.drop(columns=[c for c in DROP_COLUMNS if c in cohort.columns])

    # Stratified K-fold assignment (with very few positives, use all folds / CV rather than one hold-out split).
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    cohort["fold"] = -1
    for k, (_, val_idx) in enumerate(skf.split(cohort, cohort["tumor?"])):
        cohort.loc[val_idx, "fold"] = k

    cohort.to_csv(os.path.join(out_dir, "cohort_50.csv"), index=False)
    # Default single split == fold 0 as validation (kept for convenience / backward compatibility).
    val_df = cohort[cohort["fold"] == 0].reset_index(drop=True)
    train_df = cohort[cohort["fold"] != 0].reset_index(drop=True)
    train_df.to_csv(os.path.join(out_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val_split.csv"), index=False)

    print(f"\nCohort: {len(cohort)} cases -> {out_dir}/cohort_50.csv  (train {len(train_df)} / val {len(val_df)})")
    print("\nCT phase x tumor (check for phase/label confounding):")
    print(pd.crosstab(cohort["ct phase"], cohort["tumor?"]))
    print("\nFold x tumor:")
    print(pd.crosstab(cohort["fold"], cohort["tumor?"]))


def parse_args():
    p = argparse.ArgumentParser(description="Stratified pilot cohort + K-fold assignment")
    p.add_argument("--data_dir", default=str(CT_DIR))
    p.add_argument("--label_dir", default=str(LABEL_DIR))
    p.add_argument("--metadata", default=str(METADATA_PATH))
    p.add_argument("--out_dir", default=str(SPLIT_DIR))
    p.add_argument("--n_neg", type=int, default=45)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    generate_cohort(parse_args())
