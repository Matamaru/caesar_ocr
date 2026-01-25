from pathlib import Path

from caesar_ocr.regex.engine import load_rules, run_rules


def test_load_rules_and_run(tmp_path: Path):
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        """
- name: invoice_number
  pattern: '(?i)invoice\\s*(no|number)?\\s*[:#-]?\\s*([A-Z0-9-]{3,})'
  group: 2
  output_field: invoice_number
"""
    )

    rules = load_rules(rules_path)
    assert len(rules) == 1
    assert rules[0].name == "invoice_number"

    text = "Invoice No: ABC-123"
    out = run_rules(text, rules)
    assert out["invoice_number"] == "ABC-123"


def test_run_rules_with_plugin(tmp_path: Path):
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        """
- name: plugin_rule
  plugin: custom
"""
    )

    def custom_plugin(text, results):
        return {"len": len(text)}

    rules = load_rules(rules_path)
    out = run_rules("hello", rules, plugins={"custom": custom_plugin}, debug=True)
    assert out["len"] == 5
    assert out["__debug__"][0]["plugin"] == "custom"


def test_plugin_and_validator_examples():
    from caesar_ocr.regex.plugins import example_plugin, is_invoice

    out = example_plugin("hello", {})
    assert out["text_length"] == 5
    assert is_invoice("ABC-123", {}) is True
    assert is_invoice("bad value", {}) is False


def test_run_rules_default_registry(tmp_path: Path):
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        """
- name: use_plugin
  plugin: example_plugin
- name: invoice_number
  pattern: '(?i)invoice\\s*(no|number)?\\s*[:#-]?\\s*([A-Z0-9-]{3,})'
  group: 2
  output_field: invoice_number
  validators: [is_invoice]
"""
    )
    rules = load_rules(rules_path)
    out = run_rules("Invoice No: ABC-123", rules)
    assert out["text_length"] == len("Invoice No: ABC-123")
    assert out["invoice_number"] == "ABC-123"


def test_run_rules_with_validator(tmp_path: Path):
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        """
- name: code
  pattern: 'CODE-(\\d+)'
  group: 0
  output_field: code
  validators: [is_even]
"""
    )

    def is_even(value: str, _ctx):
        num = int(value.split("-", 1)[1])
        return num % 2 == 0

    rules = load_rules(rules_path)
    out = run_rules("CODE-3", rules, validators={"is_even": is_even})
    assert "code" not in out

    out = run_rules("CODE-4", rules, validators={"is_even": is_even}, debug=True)
    assert out["code"] == "CODE-4"
    assert out["__debug__"][-1]["valid"] is True


def test_run_rules_confidence_and_debug(tmp_path: Path):
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        """
- name: invoice_number
  pattern: '(?i)invoice\\s*(no|number)?\\s*[:#-]?\\s*([A-Z0-9-]{3,})'
  group: 2
  output_field: invoice_number
  confidence: 0.85
"""
    )
    rules = load_rules(rules_path)
    out = run_rules("Invoice No: ABC-123", rules, debug=True)
    assert out["invoice_number"] == "ABC-123"
    assert out["invoice_number_confidence"] == 0.85
    assert out["__debug__"][0]["confidence"] == 0.85
