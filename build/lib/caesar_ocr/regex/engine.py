"""Regex rule runner (YAML rules + optional Python plugins)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import yaml


ValidatorFn = Callable[[str, Dict[str, Any]], bool]
PluginFn = Callable[[str, Dict[str, Any]], Dict[str, Any]]


@dataclass
class RegexRule:
    name: str
    pattern: str
    group: int = 0
    output_field: Optional[str] = None
    confidence: Optional[float] = None
    flags: Optional[str] = None
    plugin: Optional[str] = None
    validators: Optional[List[str]] = None


@dataclass
class RegexMatch:
    rule: str
    value: str
    field: Optional[str] = None
    confidence: Optional[float] = None
    span: Optional[Tuple[int, int]] = None


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
                validators=item.get("validators"),
            )
        )
    return rules


def _run_validators(
    value: str,
    validators: Optional[List[str]],
    registry: Dict[str, ValidatorFn],
    ctx: Dict[str, Any],
) -> bool:
    if not validators:
        return True
    for name in validators:
        fn = registry.get(name)
        if fn is None:
            return False
        if not fn(value, ctx):
            return False
    return True


def run_rules(
    text: str,
    rules: Iterable[RegexRule],
    plugins: Optional[Dict[str, PluginFn]] = None,
    validators: Optional[Dict[str, ValidatorFn]] = None,
    *,
    debug: bool = False,
) -> Dict[str, Any]:
    """Run regex rules over text and return extracted fields.

    If debug is True, includes a "__debug__" list with match details.
    """
    results: Dict[str, Any] = {}
    debug_rows: List[Dict[str, Any]] = []
    default_plugins, default_validators = _default_registries()
    if plugins is None:
        plugins = default_plugins
    if validators is None:
        validators = default_validators

    ctx: Dict[str, Any] = {"results": results, "text": text}

    for rule in rules:
        if rule.plugin:
            plugin = plugins.get(rule.plugin)
            if plugin:
                plugin_out = plugin(text, results)
                results.update(plugin_out)
                if debug:
                    debug_rows.append({"rule": rule.name, "plugin": rule.plugin, "output": plugin_out})
            continue
        if not rule.pattern:
            continue
        flags = _parse_flags(rule.flags)
        match = re.search(rule.pattern, text, flags=flags)
        if not match:
            continue
        value = match.group(rule.group)
        field = rule.output_field or rule.name
        if not _run_validators(value, rule.validators, validators, ctx):
            if debug:
                debug_rows.append({"rule": rule.name, "field": field, "value": value, "valid": False})
            continue

        if rule.confidence is not None:
            results[f"{field}_confidence"] = rule.confidence
        results[field] = value
        if debug:
            debug_rows.append(
                {
                    "rule": rule.name,
                    "field": field,
                    "value": value,
                    "confidence": rule.confidence,
                    "span": match.span(rule.group),
                    "valid": True,
                }
            )

    if debug:
        results["__debug__"] = debug_rows
    return results


def _default_registries() -> Tuple[Dict[str, PluginFn], Dict[str, ValidatorFn]]:
    try:
        from . import plugins as plugin_mod
        plugins = getattr(plugin_mod, "PLUGINS", {})
        validators = getattr(plugin_mod, "VALIDATORS", {})
        return dict(plugins), dict(validators)
    except Exception:
        return {}, {}
