"""Presentation-only helpers for the DashGuard demo workflow."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


_REQUIRED_COLUMNS = frozenset(
    {
        "timestamp_sec",
        "speed_kmh",
        "accel_longitudinal",
        "accel_lateral",
        "brake",
        "telemetry_matched",
        "hard_brake",
    }
)


def _boolean_series(values: pd.Series) -> pd.Series:
    """Return a permissive boolean view of a serialized boolean column."""

    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.isin(("true", "1", "yes"))


def create_telemetry_plot(
    synchronized_csv_path: str | Path,
    output_plot_path: str | Path,
    *,
    telemetry_source: str,
) -> Path:
    """Render synchronized telemetry without introducing a new risk formula.

    Only telemetry values matched by the existing Day 6 synchronization logic
    are plotted.  The output is installed atomically and an existing file is
    never replaced.
    """

    synchronized_path = Path(synchronized_csv_path)
    output_path = Path(output_plot_path)
    if not synchronized_path.is_file():
        raise FileNotFoundError(
            "Synchronized telemetry CSV does not exist or is not a file: "
            f"{synchronized_path}"
        )
    if output_path.suffix.lower() != ".png":
        raise ValueError(f"Telemetry plot output must be a .png file: {output_path}")
    if output_path.exists():
        raise FileExistsError(f"Telemetry plot output already exists: {output_path}")

    rows = pd.read_csv(synchronized_path)
    missing = sorted(_REQUIRED_COLUMNS.difference(rows.columns))
    if missing:
        raise ValueError(
            "Synchronized telemetry CSV is missing required columns: "
            + ", ".join(missing)
        )

    timestamp = pd.to_numeric(rows["timestamp_sec"], errors="coerce")
    matched = _boolean_series(rows["telemetry_matched"]) & timestamp.notna()
    matched_rows = rows.loc[matched]
    matched_timestamp = timestamp.loc[matched]

    title_source = telemetry_source.strip().lower()
    if title_source == "synthetic":
        title = "Telemetry alignment (synthetic; illustrative, not sensor evidence)"
    else:
        title = "Telemetry alignment (provided telemetry)"

    figure, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    temporary_path: Path | None = None
    try:
        figure.suptitle(title)

        if matched_rows.empty:
            for axis in axes:
                axis.text(
                    0.5,
                    0.5,
                    "No matched telemetry",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
        else:
            speed = pd.to_numeric(matched_rows["speed_kmh"], errors="coerce")
            longitudinal = pd.to_numeric(
                matched_rows["accel_longitudinal"], errors="coerce"
            )
            lateral = pd.to_numeric(
                matched_rows["accel_lateral"], errors="coerce"
            )
            brake = _boolean_series(matched_rows["brake"]).astype(int)
            hard_brake = _boolean_series(matched_rows["hard_brake"]).astype(int)

            axes[0].plot(matched_timestamp, speed, color="#2563eb")
            axes[1].plot(
                matched_timestamp,
                longitudinal,
                label="longitudinal",
                color="#dc2626",
            )
            axes[1].plot(
                matched_timestamp,
                lateral,
                label="lateral",
                color="#7c3aed",
            )
            axes[1].legend(loc="best")
            axes[2].step(
                matched_timestamp,
                brake,
                where="mid",
                label="brake",
                color="#ea580c",
            )
            axes[2].step(
                matched_timestamp,
                hard_brake,
                where="mid",
                label="hard brake",
                color="#991b1b",
            )
            axes[2].legend(loc="best")

        axes[2].step(
            timestamp.loc[timestamp.notna()],
            matched.loc[timestamp.notna()].astype(int),
            where="mid",
            label="telemetry matched",
            color="#475569",
            alpha=0.55,
        )
        if matched_rows.empty:
            axes[2].legend(loc="best")

        axes[0].set_ylabel("Speed (km/h)")
        axes[1].set_ylabel("Acceleration (m/s²)")
        axes[2].set_ylabel("Evidence flag (0/1)")
        axes[2].set_xlabel("timestamp_sec")
        for axis in axes:
            axis.grid(alpha=0.25)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}.",
            suffix=".png",
            dir=output_path.parent,
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        figure.savefig(temporary_path, format="png", dpi=150)
        if output_path.exists():
            raise FileExistsError(
                f"Telemetry plot output appeared while writing: {output_path}"
            )
        os.replace(temporary_path, output_path)
        temporary_path = None
        return output_path
    finally:
        plt.close(figure)
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


__all__ = ["create_telemetry_plot"]
