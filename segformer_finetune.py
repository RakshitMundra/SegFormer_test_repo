"""Minimal SegFormer fine-tuning helpers (segmentation_models_pytorch).

Call train_model(...) and test_model(...) from a Kaggle notebook.

Binary segmentation only (num_classes = 1).

Data layout: a single directory holding paired files where the RGB image ends
with "_sat" and its mask ends with "_mask", e.g.
    region_1_sat.png   ->  region_1_mask.png
The extensions may differ; only the "_sat"/"_mask" stems are matched.
"""

import glob
import os

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset

# Best weights are always written under this directory (created if missing).
SAVE_DIR = "/kaggle/working/checkpoints"


def _find_pairs(data_dir):
    """Return [(sat_path, mask_path), ...] for every *_sat file with a mask."""
    pairs = []
    for sat_path in sorted(glob.glob(os.path.join(data_dir, "*_sat.*"))):
        stem, ext = os.path.splitext(sat_path)
        mask_path = stem[: -len("_sat")] + "_mask" + ext
        if not os.path.exists(mask_path):
            # allow the mask to use a different extension
            candidates = glob.glob(stem[: -len("_sat")] + "_mask.*")
            if not candidates:
                continue
            mask_path = candidates[0]
        pairs.append((sat_path, mask_path))
    if not pairs:
        raise FileNotFoundError(f"No *_sat / *_mask pairs found in {data_dir}")
    return pairs


class SegDataset(Dataset):
    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sat_path, mask_path = self.pairs[idx]
        image = cv2.cvtColor(cv2.imread(sat_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        augmented = self.transform(image=image, mask=mask)
        image, mask = augmented["image"], augmented["mask"]

        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy((mask > 0).astype("float32")).unsqueeze(0)
        return image, mask


def _transforms(img_size, mean, std, train):
    aug = []
    if train:
        aug += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]
    aug += [A.Resize(img_size, img_size), A.Normalize(mean=mean, std=std)]
    return A.Compose(aug)


def get_loaders(data_dir, encoder_name, img_size, batch_size, val_split):
    params = smp.encoders.get_preprocessing_params(encoder_name)
    mean, std = params["mean"], params["std"]

    pairs = _find_pairs(data_dir)
    n_val = int(len(pairs) * val_split)
    train_pairs, val_pairs = pairs[n_val:], pairs[:n_val]

    train_ds = SegDataset(train_pairs, _transforms(img_size, mean, std, True))
    val_ds = SegDataset(val_pairs, _transforms(img_size, mean, std, False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


def build_model(encoder_name):
    # SegFormer (MiT) encoder with a U-Net decoder head.
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )


def _make_loss(
    focal_alpha, focal_gamma, tversky_alpha, tversky_beta, tversky_gamma,
    focal_weight, tversky_weight,
):
    """Weighted sum of binary Focal loss and (Focal-)Tversky loss."""
    focal = smp.losses.FocalLoss(mode="binary", alpha=focal_alpha, gamma=focal_gamma)
    tversky = smp.losses.TverskyLoss(
        mode="binary",
        alpha=tversky_alpha,
        beta=tversky_beta,
        gamma=tversky_gamma,
        from_logits=True,
    )

    def loss_fn(logits, masks):
        return focal_weight * focal(logits, masks) + tversky_weight * tversky(logits, masks)

    return loss_fn


@torch.inference_mode()
def _evaluate(model, loader, device):
    """Mean (micro) IoU over a loader."""
    if len(loader.dataset) == 0:
        return float("nan")
    model.eval()
    scores = []
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        preds = (model(images).sigmoid() > 0.5).long()
        tp, fp, fn, tn = smp.metrics.get_stats(preds, masks.long(), mode="binary")
        scores.append(smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item())
    return float(np.mean(scores))


def train_model(
    data_dir,
    weights_filename="best_model.pth",
    encoder_name="mit_b0",
    epochs=10,
    batch_size=8,
    lr=1e-4,
    img_size=512,
    focal_alpha=0.25,
    focal_gamma=2.0,
    tversky_alpha=0.5,
    tversky_beta=0.5,
    tversky_gamma=1.0,
    focal_weight=1.0,
    tversky_weight=1.0,
    device=None,
):
    """Fine-tune SegFormer (binary) with a Focal + Tversky loss.

    Uses an 80/20 train/validation split. The best weights (highest validation
    IoU) are saved to SAVE_DIR/weights_filename; SAVE_DIR is created if needed.
    Focal/Tversky hyperparameters are exposed as arguments.
    Returns (model, history, best_path).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)
    best_path = os.path.join(SAVE_DIR, weights_filename)

    train_loader, val_loader = get_loaders(
        data_dir, encoder_name, img_size, batch_size, val_split=0.2
    )
    model = build_model(encoder_name).to(device)

    # Freeze the encoder: only the decoder + segmentation head are trained.
    for p in model.encoder.parameters():
        p.requires_grad = False

    loss_fn = _make_loss(
        focal_alpha, focal_gamma, tversky_alpha, tversky_beta, tversky_gamma,
        focal_weight, tversky_weight,
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    history = []
    best_iou = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        model.encoder.eval()  # encoder is frozen; keep it in eval mode
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        val_iou = _evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_iou": val_iou})
        print(f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_iou={val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), best_path)
            print(f"  saved best -> {best_path} (val_iou={val_iou:.4f})")

    return model, history, best_path


def test_model(
    data_dir,
    checkpoint_path,
    encoder_name="mit_b0",
    batch_size=8,
    img_size=512,
    device=None,
):
    """Load saved weights and report mean IoU over every pair in data_dir."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    params = smp.encoders.get_preprocessing_params(encoder_name)
    pairs = _find_pairs(data_dir)
    ds = SegDataset(pairs, _transforms(img_size, params["mean"], params["std"], False))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_model(encoder_name).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    iou = _evaluate(model, loader, device)
    print(f"test mean IoU = {iou:.4f}  ({len(pairs)} images)")
    return {"iou": iou, "num_images": len(pairs)}
