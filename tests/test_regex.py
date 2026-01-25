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
    out = run_rules("hello", rules, plugins={"custom": custom_plugin})
    assert out["len"] == 5
