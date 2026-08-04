from __future__ import annotations

from pathlib import Path

import matplotlib.figure
import pytest

from dashguard.demo import presentation


HEADER = (
    "timestamp_sec,speed_kmh,accel_longitudinal,accel_lateral,brake,"
    "telemetry_matched,hard_brake\n"
)


def test_telemetry_plot_closes_figure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "sync.csv"
    source.write_text(HEADER + "0,30,-0.5,0.1,1,True,False\n", encoding="utf-8")
    output = tmp_path / "plot.png"
    closed: list[object] = []
    original_close = presentation.plt.close

    def capture_close(figure: object) -> None:
        closed.append(figure)
        original_close(figure)

    monkeypatch.setattr(presentation.plt, "close", capture_close)

    result = presentation.create_telemetry_plot(
        source, output, telemetry_source="provided"
    )

    assert result == output
    assert output.is_file()
    assert len(closed) == 1


def test_telemetry_plot_handles_no_matched_rows(tmp_path: Path) -> None:
    source = tmp_path / "sync.csv"
    source.write_text(HEADER + "0,30,-0.5,0.1,0,False,False\n", encoding="utf-8")

    result = presentation.create_telemetry_plot(
        source, tmp_path / "plot.png", telemetry_source="provided"
    )

    assert result.is_file()
    assert result.stat().st_size > 0


def test_synthetic_telemetry_is_explicitly_labeled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "sync.csv"
    source.write_text(HEADER + "0,30,-0.5,0.1,0,True,False\n", encoding="utf-8")
    titles: list[str] = []
    original_suptitle = matplotlib.figure.Figure.suptitle

    def capture_title(
        figure: matplotlib.figure.Figure, title: str, *args: object, **kwargs: object
    ) -> object:
        titles.append(title)
        return original_suptitle(figure, title, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "suptitle", capture_title)

    presentation.create_telemetry_plot(
        source, tmp_path / "plot.png", telemetry_source="synthetic"
    )

    assert titles
    assert "synthetic" in titles[0].lower()
    assert "not sensor evidence" in titles[0].lower()
