"""Generate scored tracked-video CSV reports and frame-maximum plots."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from dashguard.analysis.risk_features import (
    RiskFeatureConfig,
    compute_video_risk_features,
)
from dashguard.analysis.risk_score import (
    RiskScoreConfig,
    compute_video_risk_scores,
)
from dashguard.video.metadata import VideoMetadata, probe_video_metadata


@dataclass(frozen=True)
class RiskReportSummary:
    """Serializable summary of a generated video risk proxy report."""

    input_csv_path: str
    input_video_path: str
    output_csv_path: str
    output_plot_path: str
    input_rows: int
    output_rows: int
    unique_track_ids: int
    frame_width: int
    frame_height: int
    growth_valid_rows: int
    max_score: float | None
    max_score_track_id: int | None
    max_score_frame_index: int | None
    max_score_timestamp_sec: float | None


def _require_input_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} does not exist or is not a file: {path}"
        )


def _check_output_paths(
    output_csv: Path,
    output_plot: Path,
    *,
    overwrite: bool,
) -> None:
    if output_csv.resolve() == output_plot.resolve():
        raise ValueError("Output CSV and output plot paths must be different")

    for label, path in (
        ("Output CSV", output_csv),
        ("Output plot", output_plot),
    ):
        if path.exists() and not path.is_file():
            raise IsADirectoryError(f"{label} path is not a file: {path}")
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"{label} already exists; pass overwrite=True to replace it: "
                f"{path}"
            )


def _new_temporary_path(target: Path, *, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.",
        suffix=suffix,
        dir=target.parent,
        delete=False,
    ) as temporary:
        return Path(temporary.name)


def _write_frame_maximum_plot(scores: pd.DataFrame, path: Path) -> None:
    frame_maximum = (
        scores.groupby("timestamp_sec", as_index=False, sort=True)[
            "total_score"
        ].max()
        if not scores.empty
        else scores.loc[:, ["timestamp_sec", "total_score"]]
    )

    figure = plt.figure(figsize=(9.0, 5.0))
    try:
        if frame_maximum.empty:
            plt.text(
                0.5,
                0.5,
                "No tracked objects",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
        else:
            plt.plot(
                frame_maximum["timestamp_sec"],
                frame_maximum["total_score"],
                linewidth=1.2,
            )
        plt.xlabel("Time (seconds)")
        plt.ylabel("Risk proxy score")
        plt.ylim(0.0, 1.0)
        plt.title("Video-based approach risk proxy — frame maximum")
        plt.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(path, format="png", dpi=150)
    finally:
        plt.close(figure)


def _replace_output_pair(
    temporary_csv: Path,
    output_csv: Path,
    temporary_plot: Path,
    output_plot: Path,
    *,
    overwrite: bool,
) -> None:
    pairs = (
        (temporary_csv, output_csv),
        (temporary_plot, output_plot),
    )
    backups: dict[Path, Path] = {}
    installed: list[Path] = []

    try:
        for _, target in pairs:
            if target.exists():
                if not overwrite:
                    raise FileExistsError(
                        "Output appeared while generating the report: "
                        f"{target}"
                    )
                backup = _new_temporary_path(target, suffix=".backup")
                try:
                    target.replace(backup)
                except Exception:
                    if backup.is_file():
                        backup.unlink()
                    raise
                backups[target] = backup

        for temporary, target in pairs:
            temporary.replace(target)
            installed.append(target)
    except Exception:
        for target in reversed(installed):
            if target.is_file():
                target.unlink()
        for target, backup in reversed(tuple(backups.items())):
            if backup.is_file():
                backup.replace(target)
        raise
    else:
        for backup in backups.values():
            if backup.is_file():
                backup.unlink()


def _maximum_score_values(
    scores: pd.DataFrame,
) -> tuple[float | None, int | None, int | None, float | None]:
    if scores.empty:
        return None, None, None, None

    ranked = scores.assign(_original_position=range(len(scores))).sort_values(
        [
            "total_score",
            "timestamp_sec",
            "frame_index",
            "track_id",
            "_original_position",
        ],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    )
    maximum = ranked.iloc[0]
    return (
        float(maximum["total_score"]),
        int(maximum["track_id"]),
        int(maximum["frame_index"]),
        float(maximum["timestamp_sec"]),
    )


def generate_video_risk_report(
    *,
    input_csv: str | Path,
    input_video: str | Path,
    output_csv: str | Path,
    output_plot: str | Path,
    feature_config: RiskFeatureConfig | None = None,
    score_config: RiskScoreConfig = RiskScoreConfig(),
    metadata_loader: Callable[[str | Path], VideoMetadata] = (
        probe_video_metadata
    ),
    overwrite: bool = False,
) -> RiskReportSummary:
    """Create feature/score CSV and frame-maximum risk proxy PNG outputs."""

    input_csv_path = Path(input_csv)
    input_video_path = Path(input_video)
    output_csv_path = Path(output_csv)
    output_plot_path = Path(output_plot)

    _require_input_file(input_csv_path, label="Input CSV")
    _require_input_file(input_video_path, label="Input video")
    _check_output_paths(
        output_csv_path,
        output_plot_path,
        overwrite=overwrite,
    )

    metadata = metadata_loader(input_video_path)
    if metadata.width <= 0 or metadata.height <= 0:
        raise ValueError(
            "Video metadata width and height must be greater than zero: "
            f"width={metadata.width}, height={metadata.height}"
        )
    effective_feature_config = feature_config or RiskFeatureConfig(
        frame_width=metadata.width,
        frame_height=metadata.height,
    )

    tracks = pd.read_csv(input_csv_path)
    features = compute_video_risk_features(
        tracks,
        config=effective_feature_config,
    )
    scores = compute_video_risk_scores(features, config=score_config)
    if len(scores) != len(tracks):
        raise RuntimeError(
            "Risk pipeline changed the row count: "
            f"input={len(tracks)}, output={len(scores)}"
        )

    maximum = _maximum_score_values(scores)
    summary = RiskReportSummary(
        input_csv_path=str(input_csv_path),
        input_video_path=str(input_video_path),
        output_csv_path=str(output_csv_path),
        output_plot_path=str(output_plot_path),
        input_rows=len(tracks),
        output_rows=len(scores),
        unique_track_ids=int(scores["track_id"].nunique(dropna=True)),
        frame_width=int(metadata.width),
        frame_height=int(metadata.height),
        growth_valid_rows=int(scores["growth_valid"].sum()),
        max_score=maximum[0],
        max_score_track_id=maximum[1],
        max_score_frame_index=maximum[2],
        max_score_timestamp_sec=maximum[3],
    )

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv: Path | None = None
    temporary_plot: Path | None = None
    try:
        temporary_csv = _new_temporary_path(output_csv_path, suffix=".csv")
        temporary_plot = _new_temporary_path(output_plot_path, suffix=".png")
        scores.to_csv(temporary_csv, index=False)
        _write_frame_maximum_plot(scores, temporary_plot)
        _replace_output_pair(
            temporary_csv,
            output_csv_path,
            temporary_plot,
            output_plot_path,
            overwrite=overwrite,
        )
    finally:
        for temporary in (temporary_csv, temporary_plot):
            if temporary is not None and temporary.is_file():
                temporary.unlink()

    return summary


__all__ = ["RiskReportSummary", "generate_video_risk_report"]
