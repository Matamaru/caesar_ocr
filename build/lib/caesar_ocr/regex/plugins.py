"""Optional Python plugins and validators for complex extraction."""

from __future__ import annotations

import re
from typing import Dict


def example_plugin(text: str, _results: Dict[str, object]) -> Dict[str, object]:
    """Example plugin hook. Replace with domain-specific logic."""
    return {"text_length": len(text)}


def is_invoice(value: str, _ctx: Dict[str, object]) -> bool:
    """Simple validator: allow alphanumerics and dashes only."""
    return re.fullmatch(r"[A-Z0-9-]+", value.upper()) is not None


PLUGINS = {
    "example_plugin": example_plugin,
}

VALIDATORS = {
    "is_invoice": is_invoice,
}
