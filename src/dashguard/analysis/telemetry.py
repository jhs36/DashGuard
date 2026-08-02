"""Generate, validate, and align deterministic telemetry with risk rows."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TELEMETRY_COLUMNS = (
    "timestamp_sec",
    "speed_kmh",
    "accel_longitudinal",
    "accel_lateral",
    "brake",
    "telemetry_source",
)

SYNC_COLUMNS = (
    "telemetry_timestamp_sec",
    "telemetry_aligned_timestamp_sec",
    "speed_kmh",
    "accel_longitudinal",
    "accel_lateral",
    "brake",
    "telemetry_source",
    "telemetry_delta_sec",
    "telemetry_matched",
    "hard_brake",
)

MISSING_INTERVAL_SEC = (12.0, 12.4)
HARD_BRAKE_INTERVAL_SEC = (28.2, 29.1)


@dataclass(frozen=True)
class SyntheticTelemetrySummary:
    """Serializable summary of a generated synthetic telemetry CSV."""

    output_path: str
    rows: int
    timestamp_range_sec: tuple[float, float]
    frequency_hz: float
    missing_interval_sec: tuple[float, float]
    hard_brake_rows: int
    telemetry_source: str


@dataclass(frozen=True)
class TelemetrySyncSummary:
    """Serializable summary of a risk/telemetry synchronization CSV."""

    risk_csv_path: str
    telemetry_csv_path: str
    output_path: str
    input_risk_rows: int
    output_rows: int
    output_columns: int
    matched_rows: int
    unmatched_rows: int
    maximum_matched_absolute_delta_sec: float | None
    hard_brake_matched_rows: int
    telemetry_source: str


def _finite_float(value: object, *, setting: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{setting} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{setting} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{setting} must be finite")
    return result


def _positive_finite_float(value: object, *, setting: str) -> float:
    result = _finite_float(value, setting=setting)
    if result <= 0.0:
        raise ValueError(f"{setting} must be greater than zero")
    return result


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(frame[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"telemetry column '{column}' must be numeric") from exc

    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            f"telemetry column '{column}' must contain only finite values"
        )
    return pd.Series(values, index=frame.index, name=column)


def _parse_brake_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true"}:
            return True
        if normalized in {"0", "false"}:
            return False
        raise ValueError
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric) and numeric in {0.0, 1.0}:
            return bool(numeric)
    raise ValueError


def validate_telemetry(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Validate telemetry schema and return a normalized defensive copy."""

    missing = [
        column for column in TELEMETRY_COLUMNS if column not in telemetry.columns
    ]
    if missing:
        raise ValueError(
            "telemetry is missing required columns: " + ", ".join(missing)
        )

    validated = telemetry.copy(deep=True)
    for column in (
        "timestamp_sec",
        "speed_kmh",
        "accel_longitudinal",
        "accel_lateral",
    ):
        validated[column] = _numeric_column(validated, column)

    timestamps = validated["timestamp_sec"].to_numpy(dtype=float)
    if (timestamps < 0.0).any():
        raise ValueError("telemetry column 'timestamp_sec' must be non-negative")
    timestamp_differences = np.diff(timestamps)
    if (timestamp_differences == 0.0).any():
        raise ValueError(
            "telemetry column 'timestamp_sec' must not contain duplicates"
        )
    if (timestamp_differences < 0.0).any():
        raise ValueError(
            "telemetry column 'timestamp_sec' must be strictly increasing"
        )

    if (validated["speed_kmh"].to_numpy(dtype=float) < 0.0).any():
        raise ValueError("telemetry column 'speed_kmh' must be non-negative")

    try:
        brakes = [_parse_brake_value(value) for value in validated["brake"]]
    except ValueError as exc:
        raise ValueError(
            "telemetry column 'brake' must contain only bool or 0/1 values"
        ) from exc
    validated["brake"] = pd.Series(
        [int(value) for value in brakes],
        index=validated.index,
        dtype=np.int64,
    )

    invalid_sources = [
        value
        for value in validated["telemetry_source"]
        if not isinstance(value, str) or not value.strip()
    ]
    if invalid_sources:
        raise ValueError(
            "telemetry column 'telemetry_source' must contain non-empty strings"
        )
    validated["telemetry_source"] = validated["telemetry_source"].map(
        str.strip
    )
    return validated


def generate_synthetic_telemetry(
    *,
    duration_sec: float = 30.0,
    frequency_hz: float = 10.0,
) -> pd.DataFrame:
    """Create simple deterministic telemetry for integration testing."""

    duration = _positive_finite_float(duration_sec, setting="duration_sec")
    frequency = _positive_finite_float(frequency_hz, setting="frequency_hz")

    final_sample_index = int(np.floor(duration * frequency))
    sample_indices = np.arange(final_sample_index + 1, dtype=np.int64)
    timestamps = sample_indices.astype(float) / frequency
    outside_missing_interval = ~(
        (timestamps >= MISSING_INTERVAL_SEC[0])
        & (timestamps <= MISSING_INTERVAL_SEC[1])
    )
    timestamps = timestamps[outside_missing_interval]

    hard_brake = (
        (timestamps >= HARD_BRAKE_INTERVAL_SEC[0])
        & (timestamps <= HARD_BRAKE_INTERVAL_SEC[1])
    )
    speed_kmh = 45.0 + (0.2 * timestamps)
    accel_longitudinal = 0.3 * np.sin(timestamps / 2.0)
    accel_longitudinal[hard_brake] = -3.5
    accel_lateral = 0.2 * np.cos(timestamps / 3.0)

    telemetry = pd.DataFrame(
        {
            "timestamp_sec": timestamps,
            "speed_kmh": speed_kmh,
            "accel_longitudinal": accel_longitudinal,
            "accel_lateral": accel_lateral,
            "brake": hard_brake.astype(np.int64),
            "telemetry_source": "synthetic",
        },
        columns=TELEMETRY_COLUMNS,
    )
    return validate_telemetry(telemetry)


def _validated_risk_timestamps(risk: pd.DataFrame) -> np.ndarray:
    if "timestamp_sec" not in risk.columns:
        raise ValueError("risk is missing required column: timestamp_sec")
    try:
        timestamps = pd.to_numeric(
            risk["timestamp_sec"], errors="raise"
        ).to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("risk column 'timestamp_sec' must be numeric") from exc
    if not np.isfinite(timestamps).all():
        raise ValueError(
            "risk column 'timestamp_sec' must contain only finite values"
        )
    return timestamps


def _nearest_telemetry_indices(
    risk_timestamps: np.ndarray,
    aligned_timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest indices/distances, preferring the earlier sample on ties."""

    insertion_points = np.searchsorted(
        aligned_timestamps,
        risk_timestamps,
        side="left",
    )
    left_indices = np.maximum(insertion_points - 1, 0)
    right_indices = np.minimum(insertion_points, len(aligned_timestamps) - 1)
    left_distances = np.abs(
        risk_timestamps - aligned_timestamps[left_indices]
    )
    right_distances = np.abs(
        risk_timestamps - aligned_timestamps[right_indices]
    )
    choose_left = left_distances <= right_distances
    nearest_indices = np.where(choose_left, left_indices, right_indices)
    nearest_distances = np.where(
        choose_left,
        left_distances,
        right_distances,
    )
    return nearest_indices, nearest_distances


def sync_risk_with_telemetry(
    risk: pd.DataFrame,
    telemetry: pd.DataFrame,
    *,
    offset_sec: float = 0.0,
    tolerance_sec: float = 0.05,
) -> pd.DataFrame:
    """Append nearest telemetry evidence without changing risk rows or columns."""

    offset = _finite_float(offset_sec, setting="offset_sec")
    tolerance = _finite_float(tolerance_sec, setting="tolerance_sec")
    if tolerance < 0.0:
        raise ValueError("tolerance_sec must be non-negative")

    conflicts = [column for column in SYNC_COLUMNS if column in risk.columns]
    if conflicts:
        raise ValueError(
            "risk already contains telemetry output columns: "
            + ", ".join(conflicts)
        )

    risk_timestamps = _validated_risk_timestamps(risk)
    validated_telemetry = validate_telemetry(telemetry)
    if validated_telemetry.empty:
        raise ValueError("telemetry must contain at least one row")

    telemetry_timestamps = validated_telemetry["timestamp_sec"].to_numpy(
        dtype=float
    )
    aligned_timestamps = telemetry_timestamps + offset
    if not np.isfinite(aligned_timestamps).all():
        raise ValueError(
            "aligned telemetry timestamp_sec must be finite after offset_sec"
        )

    nearest_indices, nearest_distances = _nearest_telemetry_indices(
        risk_timestamps,
        aligned_timestamps,
    )
    matched = nearest_distances <= tolerance

    row_count = len(risk)
    telemetry_timestamp_output = np.full(row_count, np.nan)
    aligned_timestamp_output = np.full(row_count, np.nan)
    speed_output = np.full(row_count, np.nan)
    longitudinal_output = np.full(row_count, np.nan)
    lateral_output = np.full(row_count, np.nan)
    delta_output = np.full(row_count, np.nan)
    brake_output = pd.array([pd.NA] * row_count, dtype="Int64")
    hard_brake_output = pd.array([pd.NA] * row_count, dtype="boolean")

    sources = validated_telemetry["telemetry_source"].astype(str)
    unique_sources = sources.unique()
    default_source: object = (
        unique_sources[0] if len(unique_sources) == 1 else pd.NA
    )
    source_output = pd.array([default_source] * row_count, dtype="string")

    if matched.any():
        output_positions = np.flatnonzero(matched)
        selected_indices = nearest_indices[matched]
        selected = validated_telemetry.iloc[selected_indices]

        selected_timestamps = telemetry_timestamps[selected_indices]
        selected_aligned = aligned_timestamps[selected_indices]
        telemetry_timestamp_output[output_positions] = selected_timestamps
        aligned_timestamp_output[output_positions] = selected_aligned
        speed_output[output_positions] = selected["speed_kmh"].to_numpy(
            dtype=float
        )
        longitudinal_values = selected["accel_longitudinal"].to_numpy(
            dtype=float
        )
        longitudinal_output[output_positions] = longitudinal_values
        lateral_output[output_positions] = selected["accel_lateral"].to_numpy(
            dtype=float
        )
        selected_brakes = selected["brake"].to_numpy(dtype=bool)
        brake_output[output_positions] = selected_brakes.astype(np.int64)
        source_output[output_positions] = selected[
            "telemetry_source"
        ].to_numpy(dtype=str)
        delta_output[output_positions] = (
            risk_timestamps[matched] - selected_aligned
        )
        hard_brake_output[output_positions] = selected_brakes & (
            longitudinal_values <= -3.0
        )

    result = risk.copy(deep=True)
    result["telemetry_timestamp_sec"] = telemetry_timestamp_output
    result["telemetry_aligned_timestamp_sec"] = aligned_timestamp_output
    result["speed_kmh"] = speed_output
    result["accel_longitudinal"] = longitudinal_output
    result["accel_lateral"] = lateral_output
    result["brake"] = brake_output
    result["telemetry_source"] = source_output
    result["telemetry_delta_sec"] = delta_output
    result["telemetry_matched"] = matched
    result["hard_brake"] = hard_brake_output
    return result


def _check_output_path(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not path.is_file():
        raise IsADirectoryError(f"Output CSV path is not a file: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(
            "Output CSV already exists; pass overwrite=True to replace it: "
            f"{path}"
        )


def _write_csv_atomic(
    frame: pd.DataFrame,
    path: Path,
    *,
    overwrite: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".csv",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary = Path(temporary_file.name)
        frame.to_csv(temporary, index=False, float_format="%.17g")
        if path.exists() and not overwrite:
            raise FileExistsError(
                "Output CSV appeared while writing; refusing to replace it: "
                f"{path}"
            )
        temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None and temporary.is_file():
            temporary.unlink()


def generate_synthetic_telemetry_csv(
    *,
    output_csv: str | Path,
    duration_sec: float = 30.0,
    frequency_hz: float = 10.0,
    overwrite: bool = False,
) -> SyntheticTelemetrySummary:
    """Generate synthetic telemetry and atomically write it to a CSV file."""

    output_path = Path(output_csv)
    _check_output_path(output_path, overwrite=overwrite)
    telemetry = generate_synthetic_telemetry(
        duration_sec=duration_sec,
        frequency_hz=frequency_hz,
    )
    _write_csv_atomic(telemetry, output_path, overwrite=overwrite)

    return SyntheticTelemetrySummary(
        output_path=str(output_path),
        rows=len(telemetry),
        timestamp_range_sec=(
            float(telemetry["timestamp_sec"].iloc[0]),
            float(telemetry["timestamp_sec"].iloc[-1]),
        ),
        frequency_hz=float(frequency_hz),
        missing_interval_sec=MISSING_INTERVAL_SEC,
        hard_brake_rows=int(telemetry["brake"].sum()),
        telemetry_source="synthetic",
    )


def sync_risk_telemetry_csv(
    *,
    risk_csv: str | Path,
    telemetry_csv: str | Path,
    output_csv: str | Path,
    offset_sec: float = 0.0,
    tolerance_sec: float = 0.05,
    overwrite: bool = False,
) -> TelemetrySyncSummary:
    """Load, synchronize, and atomically write risk and telemetry CSVs."""

    risk_path = Path(risk_csv)
    telemetry_path = Path(telemetry_csv)
    output_path = Path(output_csv)
    for label, path in (
        ("Risk CSV", risk_path),
        ("Telemetry CSV", telemetry_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"{label} does not exist or is not a file: {path}"
            )
    _check_output_path(output_path, overwrite=overwrite)

    risk = pd.read_csv(risk_path)
    risk_serialization_values = pd.read_csv(
        risk_path,
        dtype=str,
        na_filter=False,
    )
    if (
        list(risk_serialization_values.columns) != list(risk.columns)
        or len(risk_serialization_values) != len(risk)
    ):
        raise RuntimeError(
            "Risk CSV changed while it was being read for synchronization"
        )
    telemetry = pd.read_csv(telemetry_path)
    result = sync_risk_with_telemetry(
        risk,
        telemetry,
        offset_sec=offset_sec,
        tolerance_sec=tolerance_sec,
    )
    serialization_frame = pd.concat(
        [
            risk_serialization_values.reset_index(drop=True),
            result.loc[:, SYNC_COLUMNS].reset_index(drop=True),
        ],
        axis=1,
    )
    _write_csv_atomic(serialization_frame, output_path, overwrite=overwrite)

    matched = result["telemetry_matched"].to_numpy(dtype=bool)
    maximum_delta = (
        float(result.loc[matched, "telemetry_delta_sec"].abs().max())
        if matched.any()
        else None
    )
    source_values = telemetry["telemetry_source"].dropna().astype(str).unique()
    source_summary = (
        source_values[0] if len(source_values) == 1 else ",".join(source_values)
    )
    return TelemetrySyncSummary(
        risk_csv_path=str(risk_path),
        telemetry_csv_path=str(telemetry_path),
        output_path=str(output_path),
        input_risk_rows=len(risk),
        output_rows=len(result),
        output_columns=len(result.columns),
        matched_rows=int(matched.sum()),
        unmatched_rows=int((~matched).sum()),
        maximum_matched_absolute_delta_sec=maximum_delta,
        hard_brake_matched_rows=int(
            result["hard_brake"].fillna(False).astype(bool).sum()
        ),
        telemetry_source=source_summary,
    )


__all__ = [
    "HARD_BRAKE_INTERVAL_SEC",
    "MISSING_INTERVAL_SEC",
    "SYNC_COLUMNS",
    "TELEMETRY_COLUMNS",
    "SyntheticTelemetrySummary",
    "TelemetrySyncSummary",
    "generate_synthetic_telemetry",
    "generate_synthetic_telemetry_csv",
    "sync_risk_telemetry_csv",
    "sync_risk_with_telemetry",
    "validate_telemetry",
]
