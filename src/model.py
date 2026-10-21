import torch
import torch.nn as nn
from lighter_zoo import SegResEncoder


class CTFMClassifier(nn.Module):
    """
    3D Pancreatic Tumor Classifier based on pre-trained CT-FM (SegResEncoder).
    The foundation model backbone is frozen to prevent overfitting on 3D CT,
    and a lightweight MLP classification head is attached.
    """
    def __init__(self, num_classes=2, hidden_dim=128, freeze_backbone=True):
        super().__init__()
        # 1. Pre-trained CT-FM feature extractor (~77M parameters)
        self.backbone = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        self.freeze_backbone = freeze_backbone

        # 2. Lightweight classification head
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def train(self, mode: bool = True):
        # model.train() would otherwise flip the frozen backbone back to train mode
        # (changing normalisation / dropout behaviour of the pretrained encoder).
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def head_state_dict(self):
        """Only the trainable head: avoids saving the 77M frozen backbone in every checkpoint."""
        return {k: v.detach().cpu() for k, v in self.classifier.state_dict().items()}

    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                feats = self.backbone(x)[-1]
        else:
            feats = self.backbone(x)[-1]

        pooled = self.avgpool(feats)
        logits = self.classifier(pooled)
        return logits, feats


if __name__ == "__main__":
    print("Testing CTFMClassifier architecture...")
    model = CTFMClassifier(num_classes=2, freeze_backbone=True)
    model.train()
    assert not model.backbone.training and model.classifier.training
    x = torch.randn(1, 1, 80, 96, 96)
    logits, feats = model(x)
    print("Forward pass successful!")
    print("Logits shape:", logits.shape)
    print("Feature map shape:", feats.shape)
