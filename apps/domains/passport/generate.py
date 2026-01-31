"""Generate synthetic passport-like PDFs with MRZ for OCR training."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from datetime import date

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

NAME_POOLS = {
    "indian": {
        "first": ["Aarav", "Aditi", "Arjun", "Diya", "Isha", "Kabir", "Meera", "Rohan", "Saanvi", "Vihaan"],
        "last": ["Patel", "Sharma", "Gupta", "Iyer", "Nair", "Reddy", "Singh", "Khan", "Mehta", "Chatterjee"],
    },
    "european": {
        "first": ["Anna", "Ben", "Clara", "David", "Emma", "Felix", "Greta", "Hannah", "Lukas", "Sophie"],
        "last": ["Mueller", "Schmidt", "Weber", "Fischer", "Klein", "Wagner", "Novak", "Kovac", "Rossi", "Nowak"],
    },
    "african": {
        "first": ["Amina", "Chinwe", "Kwame", "Fatou", "Kofi", "Nia", "Zuri", "Amara", "Tunde", "Yara"],
        "last": ["Okafor", "Mensah", "Diallo", "Ndlovu", "Kenyatta", "Mbaye", "Adebayo", "Abebe", "Kone", "Toure"],
    },
    "english": {
        "first": ["James", "Olivia", "William", "Ava", "Henry", "Emily", "George", "Grace", "Jack", "Charlotte"],
        "last": ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Wilson", "Taylor", "Clark"],
    },
}
COUNTRIES = ["DEU", "FRA", "ESP", "ITA", "NLD"]


_MRZ_WEIGHTS = [7, 3, 1]
_MRZ_MAP = {**{str(i): i for i in range(10)}, **{chr(ord("A") + i): 10 + i for i in range(26)}, "<": 0}

_MRZ_NAME_MAP = {
    "Ä": "AE",
    "Ö": "OE",
    "Ü": "UE",
    "ß": "SS",
}


def _mrz_check_digit(value: str) -> str:
    total = 0
    for i, ch in enumerate(value):
        total += _MRZ_MAP.get(ch, 0) * _MRZ_WEIGHTS[i % 3]
    return str(total % 10)


def _mrz_name(value: str) -> str:
    value = value.upper()
    for src, repl in _MRZ_NAME_MAP.items():
        value = value.replace(src, repl)
    cleaned = []
    for ch in value:
        if "A" <= ch <= "Z":
            cleaned.append(ch)
        elif ch in (" ", "-", "<"):
            cleaned.append("<")
    return "".join(cleaned)


def _mrz_line1(surname: str, given: str, issuing: str) -> str:
    name = f"{_mrz_name(surname)}<<{_mrz_name(given)}"
    return ("P<" + issuing + name).ljust(44, "<")


def _mrz_line2(
    passport_no: str,
    nationality: str,
    birth: str,
    sex: str,
    expiry: str,
    personal_no: str,
) -> str:
    passport_no = passport_no.ljust(9, "<")
    pn_check = _mrz_check_digit(passport_no)
    birth_check = _mrz_check_digit(birth)
    expiry_check = _mrz_check_digit(expiry)
    personal_no = personal_no.ljust(14, "<")
    personal_check = _mrz_check_digit(personal_no)
    composite = passport_no + pn_check + birth + birth_check + expiry + expiry_check + personal_no + personal_check
    final_check = _mrz_check_digit(composite)
    return (
        passport_no
        + pn_check
        + nationality
        + birth
        + birth_check
        + sex
        + expiry
        + expiry_check
        + personal_no
        + personal_check
        + final_check
    )


def _rand_date(year_start: int, year_end: int) -> str:
    y = random.randint(year_start, year_end)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"{y:04d}{m:02d}{d:02d}"


def _write_passport_pdf(path: Path, name: str, nationality: str, mrz1: str, mrz2: str) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    w, h = A4
    left = 20 * mm
    top = h - 20 * mm

    c.setFont("Helvetica-Bold", 16)
    c.drawString(left, top, "Passport")
    c.setFont("Helvetica", 10)
    c.drawString(left, top - 18, f"Name: {name}")
    c.drawString(left, top - 32, f"Nationality: {nationality}")

    # MRZ zone
    mrz_y = 30 * mm
    c.setFont("Courier", 14)
    c.drawString(left, mrz_y + 14, mrz1)
    c.drawString(left, mrz_y, mrz2)

    c.save()


def generate_passports(
    output_dir: Path,
    *,
    count: int = 5,
    seed: int = 7,
    manifest_path: Path | None = None,
) -> None:
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for i in range(1, count + 1):
        pool_name = random.choice(list(NAME_POOLS.keys()))
        pool = NAME_POOLS[pool_name]
        first = random.choice(pool["first"])
        last = random.choice(pool["last"])
        issuing = random.choice(COUNTRIES)
        nationality = random.choice(COUNTRIES)
        if random.random() < 0.35:
            passport_no = f"{random.randint(10**6, 10**8 - 1)}"
        else:
            passport_no = f"{random.randint(100000000, 999999999)}"
        birth = _rand_date(1970, 2000)[2:]  # YYMMDD
        expiry = _rand_date(2026, 2035)[2:]
        sex = random.choice(["M", "F"])
        mrz1 = _mrz_line1(last, first, issuing)
        if random.random() < 0.35:
            personal_no = f"{random.randint(10**7, 10**12 - 1)}"
        else:
            personal_no = f"{random.randint(10000000000000, 99999999999999)}"
        mrz2 = _mrz_line2(passport_no, nationality, birth, sex, expiry, personal_no)
        # Validate final check digit
        composite = mrz2[0:10] + mrz2[13:20] + mrz2[21:43]
        if _mrz_check_digit(composite) != mrz2[43:44]:
            raise ValueError("Generated MRZ failed final check digit")
        name = f"{first} {last}"

        path = output_dir / f"passport_{i:03d}.pdf"
        _write_passport_pdf(path, name, nationality, mrz1, mrz2)
        print(f"Wrote {path}")
        if manifest_path:
            manifest_rows.append(
                {
                    "path": str(path),
                    "doc_type": "Passport",
                    "expected": {
                        "passport_number": passport_no.strip("<"),
                        "issuing_country": issuing,
                        "nationality": nationality,
                        "surname": last,
                        "given_names": first,
                        "birth_date_raw": birth,
                        "expiry_date_raw": expiry,
                        "sex": sex,
                        "mrz_line1": mrz1,
                        "mrz_line2": mrz2,
                    },
                }
            )

    if manifest_path:
        manifest_path.write_text(
            "\n".join(json.dumps(row) for row in manifest_rows) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic passport PDFs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--manifest", type=Path, default=None, help="Optional JSONL manifest output")
    args = parser.parse_args()

    generate_passports(
        args.output_dir,
        count=args.count,
        seed=args.seed,
        manifest_path=args.manifest,
    )


if __name__ == "__main__":
    main()
