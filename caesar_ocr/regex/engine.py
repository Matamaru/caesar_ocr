"""Regex rule runner (YAML rules + optional Python plugins)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import yaml


@dataclass
class RegexRule:
    name: str
    pattern: str
    group: int = 0
    output_field: Optional[str] = None
    confidence: Optional[float] = None
    flags: Optional[str] = None
    plugin: Optional[str] = None


@dataclass
class RegexMatch:
    rule: str
    value: str
    field: Optional[str] = None
    confidence: Optional[float] = None


def _parse_flags(flag_str: Optional[str]) -> int:
    if not flag_str:
        return 0
    mapping = {
        "I": re.IGNORECASE,
        "M": re.MULTILINE,
        "S": re.DOTALL,
        "X": re.VERBOSE,
        "A": re.ASCII,
    }
    flags = 0
    for ch in flag_str:
        if ch in mapping:
            flags |= mapping[ch]
    return flags


def load_rules(path: Path) -> List[RegexRule]:
    data = yaml.safe_load(path.read_text()) or []
    rules: List[RegexRule] = []
    for item in data:
        rules.append(
            RegexRule(
                name=item.get("name"),
                pattern=item.get("pattern", ""),
                group=int(item.get("group", 0)),
                output_field=item.get("output_field"),
                confidence=item.get("confidence"),
                flags=item.get("flags"),
                plugin=item.get("plugin"),
            )
        )
    return rules


def run_rules(
    text: str,
    rules: Iterable[RegexRule],
    plugins: Optional[Dict[str, Callable[[str, Dict[str, Any]], Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Run regex rules over text and return extracted fields."""
    results: Dict[str, Any] = {}
    if plugins is None:
        plugins = {}

    for rule in rules:
        if rule.plugin:
            plugin = plugins.get(rule.plugin)
            if plugin:
                results.update(plugin(text, results))
            continue
        if not rule.pattern:
            continue
        flags = _parse_flags(rule.flags)
        match = re.search(rule.pattern, text, flags=flags)
        if not match:
            continue
        value = match.group(rule.group)
        if rule.output_field:
            results[rule.output_field] = value
        else:
            results[rule.name] = value

    return results
