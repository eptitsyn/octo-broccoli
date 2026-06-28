from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:
    raise ImportError(
        "scipy is required for Hungarian matching: pip install scipy"
    ) from exc

from dataset import TableDatasetConfig, TableSeparatorDataset, table_collate_fn
from model import TableLineDetector, TableLineModelConfig


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _stack_target(targets: Sequence[Mapping[str, Any]], *keys: str) -> Tensor:
    values: List[Tensor] = []
    for target in targets:
        value: Any = target
        for key in keys:
            value = value[key]
        values.append(value)
    return torch.stack(values, dim=0)


def soft_dice_loss(logits: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    prob = logits.sigmoid()
    dims = tuple(range(2, prob.ndim))
    intersection = (prob * target).sum(dim=dims)
    denominator = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def binary_focal_loss_with_logits(
    logits: Tensor,
    target: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = logits.sigmoid()
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()


def line_dice_score(logits: Tensor, target: Tensor, threshold: float = 0.5) -> Tensor:
    prediction = (logits.sigmoid() >= threshold).float()
    dims = tuple(range(2, prediction.ndim))
    intersection = (prediction * target).sum(dim=dims)
    denominator = prediction.sum(dim=dims) + target.sum(dim=dims)
    return ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def homography_normalize(h: Tensor) -> Tensor:
    scale = h[..., 2:3, 2:3].clamp_min(1e-6)
    return h / scale


# -----------------------------------------------------------------------------
# Hungarian matcher
# -----------------------------------------------------------------------------


class LineHungarianMatcher(nn.Module):
    """Matches line queries to ground-truth polylines independently per image.

    Matching is intentionally performed on detached CPU costs. The selected pairs
    are then used to compute differentiable losses on the original tensors.
    """

    def __init__(
        self,
        class_cost: float = 1.0,
        point_cost: float = 5.0,
        visibility_cost: float = 1.0,
    ) -> None:
        super().__init__()
        self.class_cost = class_cost
        self.point_cost = point_cost
        self.visibility_cost = visibility_cost

    @torch.no_grad()
    def forward(
        self,
        predictions: Mapping[str, Tensor],
        targets: Sequence[Mapping[str, Tensor]],
    ) -> List[Tuple[Tensor, Tensor]]:
        class_prob = predictions["class_logits"].softmax(dim=-1)[..., 1]
        points = predictions["control_points"]
        visibility = predictions["visibility_logits"].sigmoid()

        matches: List[Tuple[Tensor, Tensor]] = []
        for batch_index, target in enumerate(targets):
            gt_points = target["control_points"].to(points.device)
            gt_visibility = target["visibility"].to(points.device)
            num_targets = gt_points.shape[0]

            if num_targets == 0:
                empty = torch.empty(0, dtype=torch.long, device=points.device)
                matches.append((empty, empty))
                continue

            pred_points = points[batch_index]       # [Q, K, 2]
            pred_vis = visibility[batch_index]      # [Q, K]
            pred_prob = class_prob[batch_index]     # [Q]

            # [Q, T, K]
            point_delta = (pred_points[:, None] - gt_points[None]).abs().sum(dim=-1)
            visible_weight = gt_visibility[None]
            point_cost = (point_delta * visible_weight).sum(dim=-1) / visible_weight.sum(dim=-1).clamp_min(1.0)

            visibility_cost = (
                pred_vis[:, None] - gt_visibility[None]
            ).abs().mean(dim=-1)
            classification_cost = -pred_prob[:, None].expand(-1, num_targets)

            total_cost = (
                self.class_cost * classification_cost
                + self.point_cost * point_cost
                + self.visibility_cost * visibility_cost
            )
            pred_indices, target_indices = linear_sum_assignment(total_cost.cpu().numpy())
            matches.append(
                (
                    torch.as_tensor(pred_indices, dtype=torch.long, device=points.device),
                    torch.as_tensor(target_indices, dtype=torch.long, device=points.device),
                )
            )
        return matches


# -----------------------------------------------------------------------------
# Criterion
# -----------------------------------------------------------------------------


class TableLineCriterion(nn.Module):
    def __init__(
        self,
        matcher: LineHungarianMatcher,
        no_object_weight: float = 0.15,
        mask_focal_alpha: float = 0.25,
        mask_focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.no_object_weight = no_object_weight
        self.mask_focal_alpha = mask_focal_alpha
        self.mask_focal_gamma = mask_focal_gamma

    def _line_losses(
        self,
        predictions: Mapping[str, Tensor],
        targets: Sequence[Mapping[str, Tensor]],
        prefix: str,
    ) -> Tuple[Dict[str, Tensor], List[Tuple[Tensor, Tensor]]]:
        matches = self.matcher(predictions, targets)
        logits = predictions["class_logits"]
        device = logits.device
        batch_size, num_queries = logits.shape[:2]

        target_classes = torch.zeros((batch_size, num_queries), dtype=torch.long, device=device)
        for batch_index, (pred_indices, _) in enumerate(matches):
            target_classes[batch_index, pred_indices] = 1

        class_weights = torch.tensor([self.no_object_weight, 1.0], device=device)
        loss_class = F.cross_entropy(
            logits.transpose(1, 2), target_classes, weight=class_weights
        )

        matched_pred_points: List[Tensor] = []
        matched_gt_points: List[Tensor] = []
        matched_pred_visibility: List[Tensor] = []
        matched_gt_visibility: List[Tensor] = []
        matched_pred_widths: List[Tensor] = []
        matched_gt_widths: List[Tensor] = []

        for batch_index, (pred_indices, target_indices) in enumerate(matches):
            if pred_indices.numel() == 0:
                continue
            target = targets[batch_index]
            matched_pred_points.append(predictions["control_points"][batch_index, pred_indices])
            matched_gt_points.append(target["control_points"][target_indices].to(device))
            matched_pred_visibility.append(predictions["visibility_logits"][batch_index, pred_indices])
            matched_gt_visibility.append(target["visibility"][target_indices].to(device))
            matched_pred_widths.append(predictions["widths"][batch_index, pred_indices])
            matched_gt_widths.append(target["widths"][target_indices].to(device))

        if matched_pred_points:
            pred_points = torch.cat(matched_pred_points, dim=0)
            gt_points = torch.cat(matched_gt_points, dim=0)
            pred_visibility = torch.cat(matched_pred_visibility, dim=0)
            gt_visibility = torch.cat(matched_gt_visibility, dim=0)
            pred_widths = torch.cat(matched_pred_widths, dim=0)
            gt_widths = torch.cat(matched_gt_widths, dim=0)

            visible = gt_visibility.unsqueeze(-1)
            loss_points = (
                (pred_points - gt_points).abs() * visible
            ).sum() / visible.sum().clamp_min(1.0) / 2.0
            loss_visibility = F.binary_cross_entropy_with_logits(
                pred_visibility, gt_visibility
            )
            loss_width = F.smooth_l1_loss(
                pred_widths * gt_visibility,
                gt_widths * gt_visibility,
                reduction="sum",
            ) / gt_visibility.sum().clamp_min(1.0)
        else:
            zero = logits.sum() * 0.0
            loss_points = zero
            loss_visibility = zero
            loss_width = zero

        predicted_positive = logits.argmax(dim=-1).eq(1).sum().float()
        target_positive = sum(t["control_points"].shape[0] for t in targets)
        line_count_error = (
            predicted_positive - float(target_positive)
        ).abs() / max(float(target_positive), 1.0)

        return {
            f"{prefix}_class": loss_class,
            f"{prefix}_points": loss_points,
            f"{prefix}_visibility": loss_visibility,
            f"{prefix}_width": loss_width,
            f"{prefix}_count_error": line_count_error,
        }, matches

    def forward(
        self,
        outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
    ) -> Tuple[Dict[str, Tensor], Dict[str, List[Tuple[Tensor, Tensor]]]]:
        horizontal_targets = [target["horizontal"] for target in targets]
        vertical_targets = [target["vertical"] for target in targets]

        horizontal_losses, horizontal_matches = self._line_losses(
            outputs["horizontal"], horizontal_targets, "horizontal"
        )
        vertical_losses, vertical_matches = self._line_losses(
            outputs["vertical"], vertical_targets, "vertical"
        )

        line_masks = _stack_target(targets, "line_masks").to(outputs["line_mask_logits"].device)
        junction_masks = _stack_target(targets, "junction_masks").to(outputs["junction_logits"].device)
        distance_maps = _stack_target(targets, "distance_maps").to(outputs["distance_maps"].device)

        loss_mask_focal = binary_focal_loss_with_logits(
            outputs["line_mask_logits"], line_masks,
            alpha=self.mask_focal_alpha, gamma=self.mask_focal_gamma,
        )
        loss_mask_dice = soft_dice_loss(outputs["line_mask_logits"], line_masks)
        loss_junction = binary_focal_loss_with_logits(
            outputs["junction_logits"], junction_masks, alpha=0.75, gamma=2.0
        )
        loss_distance = F.smooth_l1_loss(outputs["distance_maps"], distance_maps)

        gt_angle = _stack_target(targets, "geometry", "angle_vector").to(
            outputs["geometry"]["angle_vector"].device
        )
        gt_corners = _stack_target(targets, "geometry", "corners").to(
            outputs["geometry"]["corners"].device
        )
        gt_homography = _stack_target(targets, "geometry", "homography").to(
            outputs["geometry"]["homography"].device
        )

        loss_angle = (1.0 - F.cosine_similarity(
            outputs["geometry"]["angle_vector"], gt_angle, dim=-1
        )).mean()
        loss_corners = F.smooth_l1_loss(outputs["geometry"]["corners"], gt_corners)
        loss_homography = F.smooth_l1_loss(
            homography_normalize(outputs["geometry"]["homography"]),
            homography_normalize(gt_homography),
        )

        losses: Dict[str, Tensor] = {
            **horizontal_losses,
            **vertical_losses,
            "mask_focal": loss_mask_focal,
            "mask_dice": loss_mask_dice,
            "junction": loss_junction,
            "distance": loss_distance,
            "angle": loss_angle,
            "corners": loss_corners,
            "homography": loss_homography,
        }
        matches = {
            "horizontal": horizontal_matches,
            "vertical": vertical_matches,
        }
        return losses, matches


# -----------------------------------------------------------------------------
# Lightning DataModule
# -----------------------------------------------------------------------------


class TableDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_dir: str | Path,
        annotation_dir: str | Path,
        dataset_config: TableDatasetConfig,
        batch_size: int = 4,
        num_workers: int = 8,
        val_fraction: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.seed = seed
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None:
            return
        full_dataset = TableSeparatorDataset(
            image_dir=self.image_dir,
            annotation_dir=self.annotation_dir,
            config=self.dataset_config,
        )
        if len(full_dataset) < 2:
            raise RuntimeError("At least two image/annotation pairs are required")

        val_size = max(1, int(round(len(full_dataset) * self.val_fraction)))
        val_size = min(val_size, len(full_dataset) - 1)
        train_size = len(full_dataset) - val_size
        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=shuffle and len(dataset) >= self.batch_size,
            collate_fn=table_collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return self._loader(self.val_dataset, shuffle=False)


# -----------------------------------------------------------------------------
# Lightning module
# -----------------------------------------------------------------------------


DEFAULT_LOSS_WEIGHTS: Dict[str, float] = {
    "horizontal_class": 1.0,
    "horizontal_points": 5.0,
    "horizontal_visibility": 1.0,
    "horizontal_width": 0.25,
    "horizontal_count_error": 0.0,
    "vertical_class": 1.0,
    "vertical_points": 5.0,
    "vertical_visibility": 1.0,
    "vertical_width": 0.25,
    "vertical_count_error": 0.0,
    "mask_focal": 2.0,
    "mask_dice": 2.0,
    "junction": 1.0,
    "distance": 0.5,
    "angle": 0.5,
    "corners": 1.0,
    "homography": 0.25,
}


class TableLineLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_config: Optional[TableLineModelConfig] = None,
        learning_rate: float = 2e-4,
        backbone_learning_rate: float = 2e-5,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 2,
        max_epochs: int = 50,
        loss_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        super().__init__()
        self.model_config = model_config or TableLineModelConfig()
        self.model = TableLineDetector(self.model_config)
        self.matcher = LineHungarianMatcher()
        self.criterion = TableLineCriterion(self.matcher)
        self.learning_rate = learning_rate
        self.backbone_learning_rate = backbone_learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.loss_weights = dict(DEFAULT_LOSS_WEIGHTS)
        if loss_weights is not None:
            self.loss_weights.update(loss_weights)

        self.save_hyperparameters(
            {
                "model_config": asdict(self.model_config),
                "learning_rate": learning_rate,
                "backbone_learning_rate": backbone_learning_rate,
                "weight_decay": weight_decay,
                "warmup_epochs": warmup_epochs,
                "max_epochs": max_epochs,
                "loss_weights": self.loss_weights,
            }
        )

    def forward(self, images: Tensor) -> Dict[str, Any]:
        return self.model(images)

    def _shared_step(
        self,
        batch: Tuple[Tensor, List[Dict[str, Any]]],
        stage: str,
    ) -> Tensor:
        images, targets = batch
        outputs = self(images)
        losses, _ = self.criterion(outputs, targets)

        total_loss = sum(
            self.loss_weights.get(name, 1.0) * value
            for name, value in losses.items()
        )

        batch_size = images.shape[0]
        self.log(
            f"{stage}/loss",
            total_loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        for name, value in losses.items():
            self.log(
                f"{stage}/{name}", value,
                on_step=False, on_epoch=True,
                prog_bar=False, sync_dist=True,
                batch_size=batch_size,
            )

        with torch.no_grad():
            gt_line_masks = _stack_target(targets, "line_masks").to(images.device)
            dice = line_dice_score(outputs["line_mask_logits"], gt_line_masks)
            self.log(
                f"{stage}/line_dice", dice,
                on_step=False, on_epoch=True,
                prog_bar=stage == "val", sync_dist=True,
                batch_size=batch_size,
            )

            gt_angle = _stack_target(targets, "geometry", "angle_vector").to(images.device)
            angle_cosine = F.cosine_similarity(
                outputs["geometry"]["angle_vector"], gt_angle, dim=-1
            ).clamp(-1.0, 1.0)
            angle_error_deg = torch.rad2deg(torch.acos(angle_cosine)).mean()
            self.log(
                f"{stage}/angle_error_deg", angle_error_deg,
                on_step=False, on_epoch=True,
                sync_dist=True, batch_size=batch_size,
            )

        return total_loss

    def training_step(
        self,
        batch: Tuple[Tensor, List[Dict[str, Any]]],
        batch_idx: int,
    ) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(
        self,
        batch: Tuple[Tensor, List[Dict[str, Any]]],
        batch_idx: int,
    ) -> Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self) -> Dict[str, Any]:
        backbone_parameters: List[nn.Parameter] = []
        other_parameters: List[nn.Parameter] = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("backbone."):
                backbone_parameters.append(parameter)
            else:
                other_parameters.append(parameter)

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_parameters, "lr": self.backbone_learning_rate},
                {"params": other_parameters, "lr": self.learning_rate},
            ],
            weight_decay=self.weight_decay,
        )

        def lr_lambda(epoch: int) -> float:
            if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
                return float(epoch + 1) / float(self.warmup_epochs)
            progress = (epoch - self.warmup_epochs) / max(
                self.max_epochs - self.warmup_epochs, 1
            )
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


class ImageLoggerCallback(pl.Callback):
    """Logs a small line-mask preview to TensorBoard once per validation epoch."""

    def __init__(self, max_images: int = 2) -> None:
        super().__init__()
        self.max_images = max_images

    @torch.no_grad()
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: TableLineLightningModule,
        outputs: Any,
        batch: Tuple[Tensor, List[Dict[str, Any]]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx != 0 or trainer.sanity_checking:
            return
        logger = trainer.logger
        experiment = getattr(logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "add_images"):
            return

        images, targets = batch
        predictions = pl_module(images)
        count = min(self.max_images, images.shape[0])
        pred = predictions["line_mask_logits"][:count].sigmoid()
        gt = _stack_target(targets[:count], "line_masks").to(pred.device)

        # RGB overlay: red=horizontal, green=vertical.
        pred_rgb = torch.zeros((count, 3, *pred.shape[-2:]), device=pred.device)
        gt_rgb = torch.zeros_like(pred_rgb)
        pred_rgb[:, 0] = pred[:, 0]
        pred_rgb[:, 1] = pred[:, 1]
        gt_rgb[:, 0] = gt[:, 0]
        gt_rgb[:, 1] = gt[:, 1]
        experiment.add_images("val/predicted_lines", pred_rgb.cpu(), trainer.global_step)
        experiment.add_images("val/ground_truth_lines", gt_rgb.cpu(), trainer.global_step)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the table separator detector with PyTorch Lightning"
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/table_lines"))
    parser.add_argument("--resume", type=Path, default=None)

    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-control-points", type=int, default=16)
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--val-fraction", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num-horizontal-queries", type=int, default=100)
    parser.add_argument("--num-vertical-queries", type=int, default=100)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--fpn-channels", type=int, default=256)
    parser.add_argument("--no-pretrained-backbone", action="store_true")

    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--log-every-n-steps", type=int, default=20)
    parser.add_argument("--compile", action="store_true")
    return parser


def parse_devices(value: str) -> Any:
    if value == "auto":
        return "auto"
    if "," in value:
        return [int(item.strip()) for item in value.split(",")]
    try:
        return int(value)
    except ValueError:
        return value


def main() -> None:
    args = build_parser().parse_args()
    pl.seed_everything(args.seed, workers=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = TableDatasetConfig(
        image_size=(args.height, args.width),
        num_control_points=args.num_control_points,
        line_width=args.line_width,
    )
    model_config = TableLineModelConfig(
        pretrained_backbone=not args.no_pretrained_backbone,
        fpn_channels=args.fpn_channels,
        hidden_dim=args.hidden_dim,
        num_decoder_layers=args.decoder_layers,
        num_horizontal_queries=args.num_horizontal_queries,
        num_vertical_queries=args.num_vertical_queries,
        num_control_points=args.num_control_points,
    )

    datamodule = TableDataModule(
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        dataset_config=dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    module = TableLineLightningModule(
        model_config=model_config,
        learning_rate=args.lr,
        backbone_learning_rate=args.backbone_lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
    )
    if args.compile:
        module.model = torch.compile(module.model)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir / "checkpoints",
        filename="table-lines-{epoch:03d}-{step}",
        monitor="val/line_dice",
        mode="max",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="epoch"),
        ImageLoggerCallback(max_images=2),
    ]
    logger = TensorBoardLogger(
        save_dir=str(args.output_dir), name="tensorboard"
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=parse_devices(args.devices),
        strategy=args.strategy,
        precision=args.precision,
        max_epochs=args.epochs,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        deterministic=False,
    )
    trainer.fit(
        module,
        datamodule=datamodule,
        ckpt_path=str(args.resume) if args.resume is not None else None,
    )


if __name__ == "__main__":
    main()
