"""Convenience helpers for generating domain samples."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

from apps.domains.passport.generate import generate_passports
from apps.domains.diploma.generate import generate_diplomas
from apps.domains.cv.generate import generate_cvs
from apps.domains.fehlerprotokoll.generate import generate_fehlerprotokoll_reports


def generate_passport_samples(output_dir: str | Path, *, count: int = 10, seed: int = 7) -> None:
    """Generate passport PDF samples into output_dir."""
    generate_passports(Path(output_dir), count=count, seed=seed)


def generate_diploma_samples(
    output_dir: str | Path,
    *,
    count: int = 10,
    seed: int = 7,
    lang: str = "both",
    layout: str = "random",
    text_layout: str = "random",
) -> None:
    """Generate diploma PDF samples into output_dir."""
    generate_diplomas(
        Path(output_dir),
        count=count,
        seed=seed,
        lang=lang,
        layout=layout,
        text_layout=text_layout,
    )


def generate_cv_samples(
    output_dir: str | Path,
    *,
    count: int = 10,
    seed: int = 7,
    cv_type: str = "lebenslauf",
    all_types: bool = False,
) -> None:
    """Generate CV PDF samples into output_dir."""
    generate_cvs(
        Path(output_dir),
        count=count,
        seed=seed,
        cv_type=cv_type,
        all_types=all_types,
    )


def generate_fehlerprotokoll_samples(
    output_dir: str | Path,
    *,
    report_date: str | None = None,
    seed: int | None = None,
) -> None:
    """Generate a Fehlerprotokoll PDF sample for a report date."""
    date_value = None
    if report_date:
        date_value = datetime.strptime(report_date, "%Y-%m-%d").date()
    generate_fehlerprotokoll_reports(Path(output_dir), report_date=date_value, seed=seed)
