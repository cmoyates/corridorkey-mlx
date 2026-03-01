#!/usr/bin/env python3
"""Dump intermediate PyTorch tensors for MLX parity testing.

Loads original CorridorKey checkpoint, runs a forward pass,
and saves intermediate activations to reference/fixtures/.

Usage:
    uv run --group reference python scripts/dump_pytorch_reference.py \
        --checkpoint checkpoints/CorridorKey_v1.0.pth

Requires: torch, timm, rich (install via `uv sync --group reference`)
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
IMG_SIZE = 512
BACKBONE_CHANNELS = [112, 224, 448, 896]
EMBED_DIM = 256
REFINER_CHANNELS = 64
REFINER_GROUPS = 8
REFINER_SCALE = 10.0
DROPOUT_RATE = 0.1
HIERA_MODEL_NAME = "hiera_base_plus_224.mae_in1k_ft_in1k"
INPUT_CHANNELS = 4  # RGB + alpha hint
OUTPUT_DIR = Path("reference/fixtures")
FIXTURE_FILENAME = "golden.npz"

console = Console()


# ---------------------------------------------------------------------------
# Model components (mirror of nikopueringer/CorridorKey GreenFormer)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Single linear projection: input_dim -> embed_dim."""

    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DecoderHead(nn.Module):
    """SegFormer-style multiscale feature fusion head."""

    def __init__(
        self,
        in_channels: list[int],
        embed_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.linear_c1 = MLP(in_channels[0], embed_dim)
        self.linear_c2 = MLP(in_channels[1], embed_dim)
        self.linear_c3 = MLP(in_channels[2], embed_dim)
        self.linear_c4 = MLP(in_channels[3], embed_dim)

        fused_channels = embed_dim * len(in_channels)
        self.linear_fuse = nn.Conv2d(fused_channels, embed_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.classifier = nn.Conv2d(embed_dim, output_dim, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = features
        target_size = c1.shape[2:]  # H/4, W/4

        projected = []
        for feat, linear in zip(
            [c1, c2, c3, c4],
            [self.linear_c1, self.linear_c2, self.linear_c3, self.linear_c4],
            strict=True,
        ):
            b, c, h, w = feat.shape
            # Flatten spatial -> project -> reshape
            x = feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
            x = linear(x)  # [B, H*W, embed_dim]
            x = x.transpose(1, 2).reshape(b, -1, h, w)  # [B, embed_dim, H, W]
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            projected.append(x)

        fused = torch.cat(projected, dim=1)  # [B, embed_dim*4, H/4, W/4]
        fused = self.linear_fuse(fused)
        fused = self.bn(fused)
        fused = F.relu(fused)
        fused = F.dropout(fused, p=DROPOUT_RATE, training=self.training)
        return self.classifier(fused)


class RefinerBlock(nn.Module):
    """Dilated residual block with GroupNorm."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.gn1 = nn.GroupNorm(REFINER_GROUPS, channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.gn2 = nn.GroupNorm(REFINER_GROUPS, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.relu(out + residual)


class CNNRefinerModule(nn.Module):
    """CNN refiner: stem + 4 dilated ResBlocks + 1x1 projection."""

    def __init__(self) -> None:
        super().__init__()
        # 7 input channels: RGB (3) + coarse_pred (4: alpha + fg)
        refiner_input_channels = 7
        self.stem = nn.Sequential(
            nn.Conv2d(refiner_input_channels, REFINER_CHANNELS, kernel_size=3, padding=1),
            nn.GroupNorm(REFINER_GROUPS, REFINER_CHANNELS),
            nn.ReLU(),
        )
        self.res1 = RefinerBlock(REFINER_CHANNELS, dilation=1)
        self.res2 = RefinerBlock(REFINER_CHANNELS, dilation=2)
        self.res3 = RefinerBlock(REFINER_CHANNELS, dilation=4)
        self.res4 = RefinerBlock(REFINER_CHANNELS, dilation=8)
        # 4 output channels: delta for alpha (1) + delta for fg (3)
        refiner_output_channels = 4
        self.final = nn.Conv2d(REFINER_CHANNELS, refiner_output_channels, kernel_size=1)

    def forward(self, rgb: torch.Tensor, coarse_pred: torch.Tensor) -> torch.Tensor:
        x = torch.cat([rgb, coarse_pred], dim=1)  # [B, 7, H, W]
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        return self.final(x) * REFINER_SCALE


class GreenFormer(nn.Module):
    """Top-level CorridorKey model: Hiera encoder + dual decoder heads + CNN refiner."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            HIERA_MODEL_NAME,
            pretrained=False,
            features_only=True,
            img_size=IMG_SIZE,
        )
        self._patch_first_conv()

        self.alpha_decoder = DecoderHead(BACKBONE_CHANNELS, EMBED_DIM, output_dim=1)
        self.fg_decoder = DecoderHead(BACKBONE_CHANNELS, EMBED_DIM, output_dim=3)
        self.refiner = CNNRefinerModule()

    def _patch_first_conv(self) -> None:
        """Replace 3-channel patch embed conv with 4-channel version."""
        old_conv = self.encoder.model.patch_embed.proj
        new_conv = nn.Conv2d(
            INPUT_CHANNELS,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        # Zero-init; checkpoint weights will overwrite
        nn.init.zeros_(new_conv.weight)
        if old_conv.bias is not None:
            new_conv.bias = old_conv.bias
        new_conv.weight.data[:, :3] = old_conv.weight.data
        self.encoder.model.patch_embed.proj = new_conv

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[2:]  # (H, W)

        # Backbone
        features = self.encoder(x)  # list of 4 feature maps

        # Decoder heads (at H/4 resolution)
        alpha_logits = self.alpha_decoder(features)
        fg_logits = self.fg_decoder(features)

        # Upsample to input resolution
        alpha_logits_up = F.interpolate(
            alpha_logits, size=input_size, mode="bilinear", align_corners=False
        )
        fg_logits_up = F.interpolate(
            fg_logits, size=input_size, mode="bilinear", align_corners=False
        )

        # Coarse predictions
        alpha_coarse = torch.sigmoid(alpha_logits_up)
        fg_coarse = torch.sigmoid(fg_logits_up)

        # Refiner
        rgb = x[:, :3]
        coarse_pred = torch.cat([alpha_coarse, fg_coarse], dim=1)  # [B, 4, H, W]
        delta_logits = self.refiner(rgb, coarse_pred)

        # Final predictions (residual in logit space)
        alpha_final = torch.sigmoid(alpha_logits_up + delta_logits[:, 0:1])
        fg_final = torch.sigmoid(fg_logits_up + delta_logits[:, 1:4])

        return {
            "encoder_features": features,
            "alpha_logits": alpha_logits,
            "fg_logits": fg_logits,
            "alpha_logits_up": alpha_logits_up,
            "fg_logits_up": fg_logits_up,
            "alpha_coarse": alpha_coarse,
            "fg_coarse": fg_coarse,
            "delta_logits": delta_logits,
            "alpha_final": alpha_final,
            "fg_final": fg_final,
        }


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def _interpolate_pos_embed(
    ckpt_embed: torch.Tensor,
    model_embed: torch.Tensor,
    embed_dim: int,
) -> torch.Tensor:
    """Bicubic interpolation of pos_embed from checkpoint resolution to model resolution.

    Both tensors are shape (1, N, C) where N = H*W spatial tokens.
    """
    ckpt_n = ckpt_embed.shape[1]
    model_n = model_embed.shape[1]
    if ckpt_n == model_n:
        return ckpt_embed

    import math

    ckpt_side = int(math.sqrt(ckpt_n))
    model_side = int(math.sqrt(model_n))
    assert ckpt_side * ckpt_side == ckpt_n, f"pos_embed not square: {ckpt_n}"
    assert model_side * model_side == model_n, f"pos_embed not square: {model_n}"

    console.print(
        f"[cyan]Interpolating pos_embed: {ckpt_side}x{ckpt_side} -> "
        f"{model_side}x{model_side}[/cyan]"
    )

    # (1, N, C) -> (1, C, H, W) for interpolation
    embed = ckpt_embed.reshape(1, ckpt_side, ckpt_side, embed_dim).permute(0, 3, 1, 2)
    embed = F.interpolate(
        embed, size=(model_side, model_side), mode="bicubic", align_corners=False
    )
    # Back to (1, N, C)
    return embed.permute(0, 2, 3, 1).reshape(1, model_n, embed_dim)


def load_checkpoint(model: GreenFormer, checkpoint_path: Path) -> None:
    """Load state_dict from checkpoint, stripping torch.compile prefix if present."""
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = raw.get("state_dict", raw)

    # Strip _orig_mod. prefix from torch.compile
    compile_prefix = "_orig_mod."
    cleaned: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        clean_key = key[len(compile_prefix) :] if key.startswith(compile_prefix) else key
        cleaned[clean_key] = value

    # Interpolate pos_embed if spatial dimensions differ
    pos_embed_key = "encoder.model.pos_embed"
    if pos_embed_key in cleaned:
        model_embed = model.state_dict()[pos_embed_key]
        if cleaned[pos_embed_key].shape != model_embed.shape:
            cleaned[pos_embed_key] = _interpolate_pos_embed(
                cleaned[pos_embed_key],
                model_embed,
                embed_dim=BACKBONE_CHANNELS[0],
            )

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        console.print(f"[yellow]Missing keys ({len(missing)}):[/yellow]")
        for k in missing:
            console.print(f"  {k}")
    if unexpected:
        console.print(f"[yellow]Unexpected keys ({len(unexpected)}):[/yellow]")
        for k in unexpected:
            console.print(f"  {k}")
    if not missing and not unexpected:
        console.print("[green]All keys matched.[/green]")


# ---------------------------------------------------------------------------
# Tensor dumping
# ---------------------------------------------------------------------------
def dump_fixtures(outputs: dict[str, torch.Tensor | list[torch.Tensor]], output_dir: Path) -> None:
    """Save all intermediate tensors to a single .npz file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}

    # Flatten encoder features into separate keys
    features = outputs.pop("encoder_features")
    for i, feat in enumerate(features):
        arrays[f"encoder_feature_{i}"] = feat.cpu().numpy()

    # Remaining tensors
    for name, tensor in outputs.items():
        assert isinstance(tensor, torch.Tensor), f"Expected Tensor for {name}, got {type(tensor)}"
        arrays[name] = tensor.cpu().numpy()

    out_path = output_dir / FIXTURE_FILENAME
    np.savez(out_path, **arrays)
    console.print(f"\n[green]Saved fixtures to {out_path}[/green]")


WEIGHTS_FILENAME = "golden_weights.npz"


def dump_weights(model: GreenFormer, output_dir: Path) -> None:
    """Save decoder and refiner weights as numpy arrays for Phase 2 parity tests.

    Keys are the PyTorch state_dict keys (e.g. 'alpha_decoder.linear_c1.proj.weight').
    Conv weights remain in PyTorch format (O,I,H,W) — tests handle conversion.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}

    # Extract decoder and refiner state dicts
    prefixes = ("alpha_decoder.", "fg_decoder.", "refiner.")
    for key, param in model.state_dict().items():
        if any(key.startswith(p) for p in prefixes):
            arrays[key] = param.cpu().numpy()

    out_path = output_dir / WEIGHTS_FILENAME
    np.savez(out_path, **arrays)
    console.print(f"[green]Saved {len(arrays)} weight tensors to {out_path}[/green]")


def print_shape_report(outputs: dict[str, torch.Tensor | list[torch.Tensor]]) -> None:
    """Print a rich table of tensor names, shapes, and dtypes."""
    table = Table(title="Reference Fixture Shapes")
    table.add_column("Tensor", style="cyan")
    table.add_column("Shape", style="green")
    table.add_column("Dtype", style="yellow")
    table.add_column("Min", style="dim")
    table.add_column("Max", style="dim")

    for name, value in outputs.items():
        if isinstance(value, list):
            for i, v in enumerate(value):
                table.add_row(
                    f"{name}[{i}]",
                    str(tuple(v.shape)),
                    str(v.dtype),
                    f"{v.min().item():.4f}",
                    f"{v.max().item():.4f}",
                )
        else:
            table.add_row(
                name,
                str(tuple(value.shape)),
                str(value.dtype),
                f"{value.min().item():.4f}",
                f"{value.max().item():.4f}",
            )

    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Dump PyTorch reference fixtures")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/CorridorKey_v1.0.pth"),
        help="Path to CorridorKey checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for fixture output",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for deterministic input",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        console.print(f"[red]Checkpoint not found: {args.checkpoint}[/red]")
        console.print("Download CorridorKey_v1.0.pth and place it in checkpoints/")
        raise SystemExit(1)

    # Deterministic setup
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    console.print("[bold]Building GreenFormer model...[/bold]")
    model = GreenFormer()
    # Set to evaluation mode (disables dropout, uses running stats for BatchNorm)
    model.train(False)

    console.print(f"[bold]Loading checkpoint: {args.checkpoint}[/bold]")
    load_checkpoint(model, args.checkpoint)

    # Deterministic input
    torch.manual_seed(args.seed)
    sample_input = torch.randn(1, INPUT_CHANNELS, IMG_SIZE, IMG_SIZE)

    shape_str = str(tuple(sample_input.shape))
    console.print(f"[bold]Running forward pass (input shape: {shape_str})...[/bold]")
    outputs = model(sample_input)

    # Add input to outputs for completeness
    outputs = {"input": sample_input, **outputs}

    print_shape_report(outputs)
    dump_fixtures(outputs, args.output_dir)
    dump_weights(model, args.output_dir)


if __name__ == "__main__":
    main()
