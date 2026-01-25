"""Optional Python plugins for complex extraction."""

from __future__ import annotations

from typing import Dict


def example_plugin(_text: str) -> Dict[str, str]:
    """Example plugin hook. Replace with domain-specific logic."""
    return {}
