import argparse
import json
import pathlib
import random
from typing import List, Dict


def _read_jsonl(path: pathlib.Path) -> List[Dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: pathlib.Path, records: List[Dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(rec, ensure_ascii=True) for rec in records))


def main() -> None:
    parser = argparse.ArgumentParser(description="Split JSONL into train/val")
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--train", type=pathlib.Path, required=True)
    parser.add_argument("--val", type=pathlib.Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = _read_jsonl(args.input)
    if not records:
        raise SystemExit("No records found in input JSONL")

    random.seed(args.seed)
    random.shuffle(records)

    val_count = max(1, int(len(records) * args.val_ratio))
    val_records = records[:val_count]
    train_records = records[val_count:]

    args.train.parent.mkdir(parents=True, exist_ok=True)
    args.val.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.train, train_records)
    _write_jsonl(args.val, val_records)

    print(f"Train: {len(train_records)} records -> {args.train}")
    print(f"Val: {len(val_records)} records -> {args.val}")


if __name__ == "__main__":
    main()
