"""Create validated, rollback-safe video clips from event JSON records."""

from __future__ import annotations

import json
import math
import stat
import subprocess
import tempfile
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dashguard.video.metadata import VideoMetadata, probe_video_metadata


SCHEMA_VERSION = "1.0"
_TOP_LEVEL_FIELDS = (
    "schema_version",
    "source_video",
    "video_duration_sec",
    "events",
)
_EVENT_FIELDS = ("event_id", "clip_start_sec", "clip_end_sec")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_UNSAFE_FILENAME_CHARACTERS = set('<>:"/\\|?*')


class EventClipError(RuntimeError):
    """Base exception for event clip generation failures."""


class EventJSONValidationError(ValueError):
    """Raised when the event JSON does not satisfy the clip input contract."""


class FFmpegNotFoundError(EventClipError):
    """Raised when the configured FFmpeg executable cannot be found."""


class FFmpegExecutionError(EventClipError):
    """Raised when FFmpeg fails or does not create a usable output file."""


class ClipValidationError(EventClipError):
    """Raised when generated clip metadata does not satisfy the contract."""


@dataclass(frozen=True)
class EventClipRequest:
    """Validated clip range belonging to one event."""

    event_id: str
    clip_start_sec: float
    clip_end_sec: float

    @property
    def requested_duration_sec(self) -> float:
        """Return the requested clip duration."""

        return self.clip_end_sec - self.clip_start_sec


@dataclass(frozen=True)
class EventClipDocument:
    """Validated event document fields used by the extraction pipeline."""

    schema_version: str
    source_video: str
    video_duration_sec: float
    events: tuple[EventClipRequest, ...]


@dataclass(frozen=True)
class ValidatedClip:
    """Validated output metadata for a generated clip."""

    actual_duration_sec: float
    duration_difference_sec: float
    width: int
    height: int
    fps: float
    frame_count: int | None
    codec: str
    file_size_bytes: int


@dataclass(frozen=True)
class ClipManifestRecord:
    """One JSON-ready clip manifest record in contract field order."""

    event_id: str
    output_path: str
    clip_start_sec: float
    clip_end_sec: float
    requested_duration_sec: float
    actual_duration_sec: float
    duration_difference_sec: float
    width: int
    height: int
    fps: float
    frame_count: int | None
    codec: str
    file_size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary whose insertion order is the manifest contract."""

        return {
            "event_id": self.event_id,
            "output_path": self.output_path,
            "clip_start_sec": self.clip_start_sec,
            "clip_end_sec": self.clip_end_sec,
            "requested_duration_sec": self.requested_duration_sec,
            "actual_duration_sec": self.actual_duration_sec,
            "duration_difference_sec": self.duration_difference_sec,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "codec": self.codec,
            "file_size_bytes": self.file_size_bytes,
        }


@dataclass(frozen=True)
class EventClipPipelineResult:
    """Serializable result returned after every output is installed."""

    events_json_path: str
    input_video_path: str
    output_directory: str
    manifest_path: str
    event_count: int
    clip_count: int
    clips: tuple[ClipManifestRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the CLI JSON summary."""

        return {
            "events_json_path": self.events_json_path,
            "input_video_path": self.input_video_path,
            "output_directory": self.output_directory,
            "manifest_path": self.manifest_path,
            "event_count": self.event_count,
            "clip_count": self.clip_count,
            "clips": [clip.to_dict() for clip in self.clips],
        }


FFmpegRunner = Callable[..., subprocess.CompletedProcess[str]]
MetadataLoader = Callable[[str | Path, str], VideoMetadata]


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EventJSONValidationError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise EventJSONValidationError(f"{field} must be a finite number")
    return result


def _safe_event_id(value: object, *, index: int) -> str:
    field = f"events[{index}].event_id"
    if not isinstance(value, str) or not value.strip():
        raise EventJSONValidationError(f"{field} must be a non-empty string")
    if value != value.strip():
        raise EventJSONValidationError(
            f"{field} is not a safe filename: leading or trailing whitespace"
        )
    if ".." in value:
        raise EventJSONValidationError(
            f"{field} is not a safe filename: '..' is not allowed"
        )
    if any(character in _UNSAFE_FILENAME_CHARACTERS for character in value):
        raise EventJSONValidationError(
            f"{field} is not a safe filename: path separators or reserved "
            "characters are not allowed"
        )
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise EventJSONValidationError(
            f"{field} is not a safe filename: control characters are not allowed"
        )
    if value in {".", ".."} or value.rstrip(" .") != value:
        raise EventJSONValidationError(f"{field} is not a safe filename")
    if value.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise EventJSONValidationError(
            f"{field} is not a safe filename: reserved device name"
        )
    return value


def _reject_json_constant(value: str) -> None:
    raise EventJSONValidationError(
        f"Event JSON contains a non-finite number: {value}"
    )


def load_event_json(
    events_json_path: str | Path,
    *,
    actual_video_duration_sec: float,
    duration_tolerance_sec: float = 0.10,
) -> EventClipDocument:
    """Load and validate an event JSON document and every clip range."""

    path = Path(events_json_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Event JSON path does not exist or is not a file: {path}"
        )
    actual_duration = _finite_number(
        actual_video_duration_sec,
        field="actual_video_duration_sec",
    )
    tolerance = _finite_number(
        duration_tolerance_sec,
        field="duration_tolerance_sec",
    )
    if actual_duration <= 0.0:
        raise EventJSONValidationError(
            "actual_video_duration_sec must be greater than zero"
        )
    if tolerance < 0.0:
        raise EventJSONValidationError(
            "duration_tolerance_sec must be non-negative"
        )

    try:
        with path.open("r", encoding="utf-8") as source:
            payload = json.load(source, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise EventJSONValidationError(
            f"Event JSON is not valid JSON: {path}: {exc.msg}"
        ) from exc

    if not isinstance(payload, dict):
        raise EventJSONValidationError("Event JSON root must be an object")
    missing = [field for field in _TOP_LEVEL_FIELDS if field not in payload]
    if missing:
        raise EventJSONValidationError(
            "Event JSON is missing required top-level fields: "
            + ", ".join(missing)
        )

    schema_version = payload["schema_version"]
    if not isinstance(schema_version, str) or not schema_version.strip():
        raise EventJSONValidationError(
            "schema_version must be a non-empty string"
        )
    source_video = payload["source_video"]
    if not isinstance(source_video, str) or not source_video.strip():
        raise EventJSONValidationError("source_video must be a non-empty string")
    json_duration = _finite_number(
        payload["video_duration_sec"],
        field="video_duration_sec",
    )
    if json_duration <= 0.0:
        raise EventJSONValidationError(
            "video_duration_sec must be greater than zero"
        )
    duration_difference = abs(json_duration - actual_duration)
    if duration_difference > tolerance and not math.isclose(
        duration_difference,
        tolerance,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise EventJSONValidationError(
            "Event JSON video_duration_sec does not match the actual input "
            f"video duration: JSON={json_duration:.9g}, "
            f"actual={actual_duration:.9g}, tolerance={tolerance:.9g}"
        )

    raw_events = payload["events"]
    if not isinstance(raw_events, list):
        raise EventJSONValidationError("events must be a list")
    if not raw_events:
        raise EventJSONValidationError("events must not be empty")

    seen_ids: set[str] = set()
    events: list[EventClipRequest] = []
    for index, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, dict):
            raise EventJSONValidationError(f"events[{index}] must be an object")
        event_missing = [
            field for field in _EVENT_FIELDS if field not in raw_event
        ]
        if event_missing:
            raise EventJSONValidationError(
                f"events[{index}] is missing required fields: "
                + ", ".join(event_missing)
            )
        event_id = _safe_event_id(raw_event["event_id"], index=index)
        if event_id in seen_ids:
            raise EventJSONValidationError(
                f"Duplicate event_id is not allowed: {event_id!r}"
            )
        start = _finite_number(
            raw_event["clip_start_sec"],
            field=f"events[{index}].clip_start_sec",
        )
        end = _finite_number(
            raw_event["clip_end_sec"],
            field=f"events[{index}].clip_end_sec",
        )
        if start < 0.0:
            raise EventJSONValidationError(
                f"events[{index}].clip_start_sec must be non-negative"
            )
        if start >= end:
            raise EventJSONValidationError(
                f"events[{index}] clip_start_sec must be less than "
                "clip_end_sec"
            )
        if end > actual_duration:
            raise EventJSONValidationError(
                f"events[{index}].clip_end_sec exceeds the actual input video "
                f"duration: end={end:.9g}, actual={actual_duration:.9g}"
            )
        seen_ids.add(event_id)
        events.append(EventClipRequest(event_id, start, end))

    return EventClipDocument(
        schema_version=schema_version,
        source_video=source_video,
        video_duration_sec=json_duration,
        events=tuple(events),
    )


def _positive_finite(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite and greater than zero")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be finite and greater than zero") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and greater than zero")
    return result


def _nonnegative_finite(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite and non-negative")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be finite and non-negative") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def build_ffmpeg_command(
    input_video: str | Path,
    output_path: str | Path,
    *,
    clip_start_sec: float,
    requested_duration_sec: float,
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    """Build an accurate-seek H.264 re-encoding command for one clip."""

    start = _nonnegative_finite(clip_start_sec, field="clip_start_sec")
    duration = _positive_finite(
        requested_duration_sec,
        field="requested_duration_sec",
    )
    return [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(input_video),
        "-ss",
        format(start, ".17g"),
        "-t",
        format(duration, ".17g"),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def _stderr_detail(stderr: str | None, *, limit: int = 1_000) -> str:
    detail = (stderr or "").strip()
    if not detail:
        return "FFmpeg provided no error details"
    if len(detail) > limit:
        return f"{detail[:limit].rstrip()}..."
    return detail


def generate_event_clip(
    input_video: str | Path,
    output_path: str | Path,
    *,
    clip_start_sec: float,
    requested_duration_sec: float,
    ffmpeg_bin: str = "ffmpeg",
    runner: FFmpegRunner | None = None,
) -> Path:
    """Generate one non-overwriting H.264 clip and verify its file payload."""

    source = Path(input_video)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Input video path does not exist or is not a file: {source}"
        )
    if destination.suffix.lower() != ".mp4":
        raise ValueError(f"Clip output path must have an .mp4 extension: {destination}")
    if destination.exists():
        raise FileExistsError(
            f"Clip output already exists; refusing to overwrite it: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = build_ffmpeg_command(
        source,
        destination,
        clip_start_sec=clip_start_sec,
        requested_duration_sec=requested_duration_sec,
        ffmpeg_bin=ffmpeg_bin,
    )
    execute = subprocess.run if runner is None else runner
    try:
        completed = execute(
            command,
            capture_output=True,
            text=True,
            check=False,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError(
            f"FFmpeg executable {ffmpeg_bin!r} was not found while creating "
            f"clip {destination}"
        ) from exc
    except OSError as exc:
        raise FFmpegExecutionError(
            f"Could not execute FFmpeg while creating clip {destination}: {exc}"
        ) from exc

    if completed.returncode != 0:
        detail = _stderr_detail(completed.stderr)
        raise FFmpegExecutionError(
            f"FFmpeg failed for clip {destination} with exit code "
            f"{completed.returncode}: {detail}"
        )
    try:
        valid_output = (
            destination.is_file()
            and stat.S_ISREG(destination.stat().st_mode)
            and destination.stat().st_size > 0
        )
    except OSError as exc:
        raise FFmpegExecutionError(
            f"Could not verify FFmpeg output {destination}: {exc}"
        ) from exc
    if not valid_output:
        raise FFmpegExecutionError(
            f"FFmpeg reported success, but output is missing or empty: "
            f"{destination}"
        )
    return destination


def validate_generated_clip(
    output_path: str | Path,
    *,
    requested_duration_sec: float,
    ffprobe_bin: str = "ffprobe",
    metadata_loader: MetadataLoader = probe_video_metadata,
) -> ValidatedClip:
    """Probe a generated clip and enforce metadata and duration tolerances."""

    path = Path(output_path)
    requested_duration = _positive_finite(
        requested_duration_sec,
        field="requested_duration_sec",
    )
    try:
        file_stat = path.stat()
    except FileNotFoundError as exc:
        raise ClipValidationError(f"Generated clip does not exist: {path}") from exc
    except OSError as exc:
        raise ClipValidationError(
            f"Could not inspect generated clip {path}: {exc}"
        ) from exc
    if not path.is_file() or not stat.S_ISREG(file_stat.st_mode):
        raise ClipValidationError(f"Generated clip is not a regular file: {path}")
    if file_stat.st_size <= 0:
        raise ClipValidationError(f"Generated clip is empty: {path}")

    try:
        metadata = metadata_loader(path, ffprobe_bin)
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise ClipValidationError(
            f"Could not extract metadata from generated clip {path}: {exc}"
        ) from exc

    if metadata.width <= 0 or metadata.height <= 0:
        raise ClipValidationError(
            f"Generated clip dimensions must be positive: {path}"
        )
    if not math.isfinite(metadata.fps) or metadata.fps <= 0.0:
        raise ClipValidationError(f"Generated clip FPS must be positive: {path}")
    if (
        not math.isfinite(metadata.duration_sec)
        or metadata.duration_sec <= 0.0
    ):
        raise ClipValidationError(
            f"Generated clip duration must be positive: {path}"
        )
    codec = metadata.codec.strip() if isinstance(metadata.codec, str) else ""
    if not codec:
        raise ClipValidationError(
            f"Generated clip codec information is missing: {path}"
        )
    if metadata.frame_count is not None and metadata.frame_count <= 0:
        raise ClipValidationError(
            f"Generated clip frame count must be positive when present: {path}"
        )

    difference = metadata.duration_sec - requested_duration
    tolerance = max(0.10, 2.0 / metadata.fps)
    if abs(difference) > tolerance and not math.isclose(
        abs(difference),
        tolerance,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ClipValidationError(
            f"Generated clip duration differs from the request beyond tolerance "
            f"for {path}: requested={requested_duration:.9g}, "
            f"actual={metadata.duration_sec:.9g}, difference={difference:.9g}, "
            f"tolerance={tolerance:.9g}"
        )
    return ValidatedClip(
        actual_duration_sec=float(metadata.duration_sec),
        duration_difference_sec=float(difference),
        width=int(metadata.width),
        height=int(metadata.height),
        fps=float(metadata.fps),
        frame_count=metadata.frame_count,
        codec=codec,
        file_size_bytes=int(file_stat.st_size),
    )


def _assert_standard_json(value: object, *, location: str = "root") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON number at {location}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object key at {location} must be a string")
            _assert_standard_json(item, location=f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_standard_json(item, location=f"{location}[{index}]")
        return
    raise TypeError(
        f"unsupported JSON value at {location}: {type(value).__name__}"
    )


def build_clip_manifest(
    *,
    source_events_json: str | Path,
    source_video: str | Path,
    output_directory: str | Path,
    clips: Sequence[ClipManifestRecord],
) -> dict[str, Any]:
    """Build the clip manifest in its required top-level field order."""

    clip_documents = [clip.to_dict() for clip in clips]
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_events_json": str(source_events_json),
        "source_video": str(source_video),
        "output_directory": str(output_directory),
        "summary": {
            "event_count": len(clips),
            "clip_count": len(clips),
            "total_requested_duration_sec": sum(
                clip.requested_duration_sec for clip in clips
            ),
            "total_actual_duration_sec": sum(
                clip.actual_duration_sec for clip in clips
            ),
        },
        "clips": clip_documents,
    }
    _assert_standard_json(document)
    return document


def write_clip_manifest(
    manifest: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Write a strict, UTF-8 manifest JSON file."""

    path = Path(output_path)
    _assert_standard_json(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(
            manifest,
            output,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        output.write("\n")
    return path


def _new_temporary_path(target: Path, *, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.",
        suffix=suffix,
        dir=target.parent,
        delete=False,
    ) as temporary:
        return Path(temporary.name)


def _new_ffmpeg_output_path(target: Path) -> Path:
    temporary = _new_temporary_path(target, suffix=".mp4")
    temporary.unlink()
    return temporary


def _check_output_paths(
    *,
    clip_targets: Sequence[Path],
    manifest_target: Path,
    events_json_path: Path,
    input_video_path: Path,
    output_directory: Path,
    overwrite: bool,
) -> None:
    if output_directory.exists() and not output_directory.is_dir():
        raise NotADirectoryError(
            f"Output directory path is not a directory: {output_directory}"
        )
    all_targets = (*clip_targets, manifest_target)
    resolved_targets = [target.resolve() for target in all_targets]
    if len(set(resolved_targets)) != len(resolved_targets):
        raise ValueError("Clip and manifest output paths must all be distinct")
    resolved_inputs = {
        events_json_path.resolve(),
        input_video_path.resolve(),
    }
    if any(target in resolved_inputs for target in resolved_targets):
        raise ValueError("Output paths must not replace an input file")

    for target in all_targets:
        if target.exists() and not target.is_file():
            raise IsADirectoryError(f"Output path is not a file: {target}")
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists; pass overwrite=True to replace it: "
                f"{target}"
            )


def _install_outputs(
    pairs: Sequence[tuple[Path, Path]],
    *,
    overwrite: bool,
) -> None:
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for _, target in pairs:
            if target.exists():
                if not target.is_file():
                    raise IsADirectoryError(f"Output path is not a file: {target}")
                if not overwrite:
                    raise FileExistsError(
                        "Output appeared during clip generation; refusing to "
                        f"replace it: {target}"
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
            try:
                if backup.is_file():
                    backup.unlink()
            except OSError:
                pass


def _validate_source_metadata(metadata: VideoMetadata, *, path: Path) -> None:
    if not math.isfinite(metadata.duration_sec) or metadata.duration_sec <= 0.0:
        raise ClipValidationError(
            f"Input video duration must be finite and positive: {path}"
        )
    if not math.isfinite(metadata.fps) or metadata.fps <= 0.0:
        raise ClipValidationError(
            f"Input video FPS must be finite and positive: {path}"
        )


def extract_event_clips(
    *,
    events_json: str | Path,
    input_video: str | Path,
    output_dir: str | Path,
    output_manifest_json: str | Path,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    overwrite: bool = False,
    runner: FFmpegRunner | None = None,
    metadata_loader: MetadataLoader = probe_video_metadata,
) -> EventClipPipelineResult:
    """Generate, validate, and atomically install all event clips and manifest."""

    events_path = Path(events_json)
    input_path = Path(input_video)
    output_directory = Path(output_dir)
    manifest_path = Path(output_manifest_json)
    if not events_path.is_file():
        raise FileNotFoundError(
            f"Event JSON path does not exist or is not a file: {events_path}"
        )
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input video path does not exist or is not a file: {input_path}"
        )

    input_metadata = metadata_loader(input_path, ffprobe_bin)
    _validate_source_metadata(input_metadata, path=input_path)
    source_duration_tolerance = max(0.10, 2.0 / input_metadata.fps)
    document = load_event_json(
        events_path,
        actual_video_duration_sec=input_metadata.duration_sec,
        duration_tolerance_sec=source_duration_tolerance,
    )
    clip_targets = [
        output_directory / f"{event.event_id}.mp4"
        for event in document.events
    ]
    _check_output_paths(
        clip_targets=clip_targets,
        manifest_target=manifest_path,
        events_json_path=events_path,
        input_video_path=input_path,
        output_directory=output_directory,
        overwrite=overwrite,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    install_pairs: list[tuple[Path, Path]] = []
    records: list[ClipManifestRecord] = []
    try:
        for event, target in zip(document.events, clip_targets, strict=True):
            temporary = _new_ffmpeg_output_path(target)
            temporary_paths.append(temporary)
            generate_event_clip(
                input_path,
                temporary,
                clip_start_sec=event.clip_start_sec,
                requested_duration_sec=event.requested_duration_sec,
                ffmpeg_bin=ffmpeg_bin,
                runner=runner,
            )
            validated = validate_generated_clip(
                temporary,
                requested_duration_sec=event.requested_duration_sec,
                ffprobe_bin=ffprobe_bin,
                metadata_loader=metadata_loader,
            )
            records.append(
                ClipManifestRecord(
                    event_id=event.event_id,
                    output_path=str(target),
                    clip_start_sec=event.clip_start_sec,
                    clip_end_sec=event.clip_end_sec,
                    requested_duration_sec=event.requested_duration_sec,
                    actual_duration_sec=validated.actual_duration_sec,
                    duration_difference_sec=(
                        validated.duration_difference_sec
                    ),
                    width=validated.width,
                    height=validated.height,
                    fps=validated.fps,
                    frame_count=validated.frame_count,
                    codec=validated.codec,
                    file_size_bytes=validated.file_size_bytes,
                )
            )
            install_pairs.append((temporary, target))

        manifest = build_clip_manifest(
            source_events_json=events_json,
            source_video=input_video,
            output_directory=output_dir,
            clips=records,
        )
        temporary_manifest = _new_temporary_path(manifest_path, suffix=".json")
        temporary_paths.append(temporary_manifest)
        write_clip_manifest(manifest, temporary_manifest)
        install_pairs.append((temporary_manifest, manifest_path))
        _install_outputs(install_pairs, overwrite=overwrite)
    finally:
        for temporary in temporary_paths:
            try:
                if temporary.is_file():
                    temporary.unlink()
            except OSError:
                pass

    return EventClipPipelineResult(
        events_json_path=str(events_json),
        input_video_path=str(input_video),
        output_directory=str(output_dir),
        manifest_path=str(output_manifest_json),
        event_count=len(document.events),
        clip_count=len(records),
        clips=tuple(records),
    )
