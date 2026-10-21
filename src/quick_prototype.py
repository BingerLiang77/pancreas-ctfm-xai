import torch
from torch.utils.data import DataLoader, Subset

from config import CACHE_DIR, CT_DIR, LABEL_DIR, SPLIT_DIR
from dataset import PanTSDataset
from model import CTFMClassifier


def run_prototype():
    """Smoke test: a few cases through the real preprocessing + one optimisation step."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = PanTSDataset(CT_DIR, LABEL_DIR, split_csv=SPLIT_DIR / "train_split.csv", cache_dir=CACHE_DIR, augment=True)
    # include at least one positive so the smoke test exercises both classes
    pos = [i for i, s in enumerate(dataset.samples) if s["label"] == 1][:1]
    neg = [i for i, s in enumerate(dataset.samples) if s["label"] == 0][:3]
    loader = DataLoader(Subset(dataset, pos + neg), batch_size=2, shuffle=True)

    model = CTFMClassifier(num_classes=2, freeze_backbone=True).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)

    model.train()
    for step, (imgs, labels, cids) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        print(f"Batch {step+1}: IDs={list(cids)}, shape={tuple(imgs.shape)}, Loss={loss.item():.4f}")
    print("Quick prototype OK")


if __name__ == "__main__":
    run_prototype()
