"""Fehlerprotokoll domain plugins and validators."""

from __future__ import annotations

from typing import Dict


def example_plugin(text: str, _results: Dict[str, object]) -> Dict[str, object]:
    """Placeholder plugin for Fehlerprotokoll extraction."""
    return {"fehlerprotokoll_text_length": len(text)}
