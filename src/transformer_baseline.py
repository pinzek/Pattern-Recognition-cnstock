"""
Transformer baseline for A-share cross-sectional return prediction.

This script provides the reusable model components and project path configuration
for the V5 Transformer baseline experiment. The complete experimental workflow,
including data loading, walk-forward training, evaluation, and visualization, is
documented in:

    notebooks/v5_transformer_baseline.ipynb

Raw market data and large intermediate files are not included in this repository.
Users should provide their own local data path through the A_SHARE_DAILY_DIR
environment variable.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn


@dataclass
class ProjectPaths:
    """Repository-relative path configuration."""

    project_root: Path
    data_dir: Path
    artifact_dir: Path
    figure_dir: Path
    report_dir: Path


def get_project_paths() -> ProjectPaths:
    """
    Build repository-relative paths.

    Raw market data are not included in this repository. To run the full
    experiment, set the local data directory with:

        export A_SHARE_DAILY_DIR=/path/to/local/daily_data
    """

    project_root = Path(__file__).resolve().parents[1]

    data_dir = Path(
        os.environ.get(
            "A_SHARE_DAILY_DIR",
            str(project_root / "data" / "daily_temp3"),
        )
    )

    artifact_dir = project_root / "artifacts"
    figure_dir = project_root / "figures"
    report_dir = project_root / "reports"

    artifact_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    return ProjectPaths(
        project_root=project_root,
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        figure_dir=figure_dir,
        report_dir=report_dir,
    )


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer sequence inputs.
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding.

        Parameters
        ----------
        x:
            Tensor with shape ``(batch_size, sequence_length, d_model)``.
        """

        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerReturnRegressor(nn.Module):
    """
    Transformer Encoder baseline for next-day stock return prediction.

    Input shape:
        (batch_size, sequence_length, feature_dim)

    Output shape:
        (batch_size,)
    """

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Linear(feature_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.prediction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        encoded = self.encoder(x)

        # Use the final sequence token as the stock-level representation.
        last_token = encoded[:, -1, :]
        return self.prediction_head(last_token).squeeze(-1)


def rank_ic_loss_proxy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Simple MSE objective used as the training loss proxy.

    Final evaluation should be based on out-of-sample Rank IC, ICIR,
    long-short return, turnover, Sharpe, and drawdown.
    """

    return nn.functional.mse_loss(pred, target)


def build_model(
    feature_dim: int,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.10,
) -> TransformerReturnRegressor:
    """Create the Transformer return prediction model."""

    return TransformerReturnRegressor(
        feature_dim=feature_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    )


def main() -> None:
    """
    Lightweight entry point.

    The full V5 experiment is notebook-based. This script only provides the
    reusable Transformer model and path configuration.
    """

    paths = get_project_paths()

    print("Project root:", paths.project_root)
    print("Data directory:", paths.data_dir)
    print("Artifact directory:", paths.artifact_dir)
    print("Figure directory:", paths.figure_dir)
    print("Report directory:", paths.report_dir)

    model = build_model(feature_dim=5)
    print(model)


if __name__ == "__main__":
    main()
