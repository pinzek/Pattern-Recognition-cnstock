"""
Configuration utilities for the V5 Transformer baseline experiment.

Raw market data are not included in this repository. To run the full workflow,
users should provide their own local A-share data path through the
A_SHARE_DAILY_DIR environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List


FEATURE_COLUMNS: List[str] = [
    "return_5d",
    "return_10d",
    "return_20d",
    "turnover_20d_mean",
    "log_neg_market_value",
]

LABEL_COLUMN: str = "label_1d_raw"

SEQUENCE_LENGTH: int = 20
TRAIN_DAYS: int = 504
EMBARGO_DAYS: int = 5
MAX_TRAIN_SEQUENCES: int = 120_000

TRANSACTION_COST_BPS: float = 10.0


@dataclass(frozen=True)
class ProjectConfig:
    """Repository-relative project configuration."""

    project_root: Path
    data_dir: Path
    artifact_dir: Path
    figure_dir: Path
    report_dir: Path

    feature_columns: List[str]
    label_column: str
    sequence_length: int
    train_days: int
    embargo_days: int
    max_train_sequences: int
    transaction_cost_bps: float


def get_project_root() -> Path:
    """Return the repository root directory."""

    return Path(__file__).resolve().parents[1]


def get_config() -> ProjectConfig:
    """
    Build the project configuration.

    The default data path points to ``data/daily_temp3`` under the repository,
    but raw data are intentionally excluded from GitHub. For a real run, set:

        export A_SHARE_DAILY_DIR=/path/to/local/daily_temp3
    """

    project_root = get_project_root()

    data_dir = Path(
        os.environ.get(
            "A_SHARE_DAILY_DIR",
            str(project_root / "data" / "daily_temp3"),
        )
    )

    artifact_dir = project_root / "artifacts"
    figure_dir = project_root / "figures"
    report_dir = project_root / "reports"

    return ProjectConfig(
        project_root=project_root,
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        figure_dir=figure_dir,
        report_dir=report_dir,
        feature_columns=FEATURE_COLUMNS,
        label_column=LABEL_COLUMN,
        sequence_length=SEQUENCE_LENGTH,
        train_days=TRAIN_DAYS,
        embargo_days=EMBARGO_DAYS,
        max_train_sequences=MAX_TRAIN_SEQUENCES,
        transaction_cost_bps=TRANSACTION_COST_BPS,
    )


def ensure_output_dirs(config: ProjectConfig) -> None:
    """Create output folders if they do not already exist."""

    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    config.figure_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    cfg = get_config()
    ensure_output_dirs(cfg)

    print("Project root:", cfg.project_root)
    print("Data directory:", cfg.data_dir)
    print("Artifact directory:", cfg.artifact_dir)
    print("Figure directory:", cfg.figure_dir)
    print("Report directory:", cfg.report_dir)
    print("Features:", cfg.feature_columns)
