from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelArtifacts:
    root: Path
    model_path: Path
    feature_names_path: Path
    metrics_path: Path
    config_path: Path
    feature_importance_path: Path


def create_artifact_paths(root: Path) -> ModelArtifacts:
    return ModelArtifacts(
        root=root,
        model_path=root / "model.bin",
        feature_names_path=root / "feature_names.json",
        metrics_path=root / "metrics.json",
        config_path=root / "config.json",
        feature_importance_path=root / "feature_importance.csv",
    )


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
