"""Generate synthetic diploma PDFs for OCR training."""

from __future__ import annotations

import argparse
import random
from datetime import date
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib import colors

NAME_POOLS = {
    "european": {
        "first": ["Anna", "Ben", "Clara", "David", "Emma", "Felix", "Greta", "Hannah", "Lukas", "Sophie"],
        "last": ["Müller", "Schmidt", "Weber", "Fischer", "Klein", "Wagner", "Jäger", "Böhm", "Köhler", "Schäfer"],
    },
    "indian": {
        "first": ["Aarav", "Aditi", "Arjun", "Diya", "Isha", "Kabir", "Meera", "Rohan", "Saanvi", "Vihaan"],
        "last": ["Patel", "Sharma", "Gupta", "Iyer", "Nair", "Reddy", "Singh", "Khan", "Mehta", "Chatterjee"],
    },
    "african": {
        "first": ["Amina", "Chinwe", "Kwame", "Fatou", "Kofi", "Nia", "Zuri", "Amara", "Tunde", "Yara"],
        "last": ["Okafor", "Mensah", "Diallo", "Ndlovu", "Kenyatta", "Mbaye", "Adebayo", "Abebe", "Kone", "Toure"],
    },
}

INSTITUTIONS_DE = [
    "Universitaet Berlin",
    "Hochschule Muenchen",
    "Fachhochschule Koeln",
    "Universitaet Hamburg",
    "Hochschule Rhein-Main",
]
INSTITUTIONS_EN = [
    "University of Berlin",
    "Munich College of Applied Sciences",
    "Cologne University of Applied Sciences",
    "Hamburg University",
    "Rhine-Main University",
]
INSTITUTIONS_IN = [
    "Delhi University",
    "Indian Institute of Technology",
    "Jawaharlal Nehru University",
    "University of Mumbai",
    "Bangalore University",
]
INSTITUTIONS_AF = [
    "University of Lagos",
    "University of Nairobi",
    "University of Cape Town",
    "Makerere University",
    "University of Ghana",
]

PROGRAMS_DE = [
    "Pflegewissenschaft",
    "Medizintechnik",
    "Gesundheitsmanagement",
    "Biomedizin",
    "Physiotherapie",
]
PROGRAMS_EN = [
    "Nursing Science",
    "Medical Engineering",
    "Health Management",
    "Biomedical Science",
    "Physiotherapy",
]

LOCATIONS_DE = ["Berlin", "Muenchen", "Koeln", "Hamburg", "Wiesbaden"]
LOCATIONS_EN = ["Berlin", "Munich", "Cologne", "Hamburg", "Wiesbaden"]
LOCATIONS_IN = ["New Delhi", "Mumbai", "Bengaluru", "Chennai", "Hyderabad"]
LOCATIONS_AF = ["Lagos", "Nairobi", "Cape Town", "Accra", "Kampala"]

DEGREE_TYPES_DE = ["Diplom", "Bachelor", "Master", "Magister", "Staatsexamen"]
DEGREE_TYPES_EN = ["Diploma", "Bachelor", "Master", "Doctor", "PhD"]
DEGREE_TYPES_IN = ["Diploma", "Bachelor", "Master", "Doctor", "PhD"]
DEGREE_TYPES_AF = ["Diploma", "Bachelor", "Master", "Doctor", "PhD"]


def _rand_date(year_start: int, year_end: int) -> str:
    y = random.randint(year_start, year_end)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"{d:02d}.{m:02d}.{y:04d}"


def _diploma_number() -> str:
    return f"DIP-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"


def _write_pdf(
    path: Path,
    *,
    lang: str,
    holder: str,
    institution: str,
    degree: str,
    program: str,
    location: str,
    issue_date: str,
    diploma_number: str,
    certified_copy: bool,
    layout: str,
    text_layout: str,
    stamp_label: str | None,
    signature_name: str,
) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    w, h = A4
    left = 20 * mm
    top = h - 20 * mm

    text_color = _draw_layout(c, w, h, layout)

    _draw_text_layout(
        c,
        w,
        h,
        text_color,
        lang=lang,
        holder=holder,
        institution=institution,
        degree=degree,
        program=program,
        location=location,
        issue_date=issue_date,
        diploma_number=diploma_number,
        layout=text_layout,
    )

    if certified_copy:
        c.setFillColor(text_color)
        c.setFont("Helvetica-Bold", 10)
        stamp = "Beglaubigte Kopie" if lang == "de" else "Certified Copy"
        c.drawString(left, 40 * mm, stamp)

    _draw_signatures(c, w, text_color, signature_name)
    if stamp_label:
        _draw_stamp(c, w - 45 * mm, 55 * mm, 16 * mm, stamp_label, text_color)

    c.save()


def _draw_layout(c: canvas.Canvas, w: float, h: float, layout: str) -> colors.Color:
    if layout == "classic":
        c.setStrokeColor(colors.darkgoldenrod)
        c.setLineWidth(2)
        c.rect(15 * mm, 15 * mm, w - 30 * mm, h - 30 * mm)
        c.setStrokeColor(colors.lightgrey)
        c.setLineWidth(1)
        c.rect(20 * mm, 20 * mm, w - 40 * mm, h - 40 * mm)
        _draw_seal(c, w - 45 * mm, 35 * mm, 14 * mm)
        return colors.black
    if layout == "minimal":
        c.setFillColor(colors.HexColor("#2F4F7F"))
        c.rect(0, h - 18 * mm, w, 18 * mm, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#2F4F7F"))
        c.rect(0, 0, w, 12 * mm, fill=1, stroke=0)
        return colors.black
    if layout == "modern":
        c.setFillColor(colors.HexColor("#1E5AA7"))
        c.saveState()
        p = c.beginPath()
        p.moveTo(0, h)
        p.lineTo(40 * mm, h)
        p.lineTo(0, h - 40 * mm)
        p.close()
        c.drawPath(p, fill=1, stroke=0)
        c.restoreState()

        c.setFillColor(colors.HexColor("#1E5AA7"))
        c.saveState()
        p = c.beginPath()
        p.moveTo(w, 0)
        p.lineTo(w - 40 * mm, 0)
        p.lineTo(w, 40 * mm)
        p.close()
        c.drawPath(p, fill=1, stroke=0)
        c.restoreState()
        return colors.black
    if layout == "dark":
        c.setFillColor(colors.black)
        c.rect(0, 0, w, h, fill=1, stroke=0)
        c.setStrokeColor(colors.HexColor("#C9A646"))
        c.setLineWidth(2)
        c.rect(15 * mm, 15 * mm, w - 30 * mm, h - 30 * mm)
        _draw_seal(c, w - 50 * mm, 35 * mm, 14 * mm, color=colors.HexColor("#C9A646"))
        return colors.white
    if layout == "playful":
        c.setStrokeColor(colors.HexColor("#5AA469"))
        c.setLineWidth(6)
        c.line(0, h - 25 * mm, w, h - 25 * mm)
        c.setStrokeColor(colors.HexColor("#F6C343"))
        c.setLineWidth(6)
        c.line(0, 25 * mm, w, 25 * mm)
        return colors.black
    if layout == "qr":
        c.setStrokeColor(colors.HexColor("#5A2D82"))
        c.setLineWidth(6)
        c.line(0, h - 18 * mm, w, h - 18 * mm)
        c.line(0, 18 * mm, w, 18 * mm)
        c.setStrokeColor(colors.darkgrey)
        c.setLineWidth(1)
        c.rect(w - 45 * mm, h - 55 * mm, 30 * mm, 30 * mm)
        c.setFont("Helvetica", 8)
        c.drawString(w - 44 * mm, h - 60 * mm, "QR")
        return colors.black
    return colors.black


def _draw_seal(c: canvas.Canvas, x: float, y: float, r: float, *, color: colors.Color = colors.red) -> None:
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.circle(x, y, r, fill=0, stroke=1)
    c.circle(x, y, r - 3, fill=0, stroke=1)


def _draw_signatures(c: canvas.Canvas, w: float, text_color: colors.Color, signature_name: str) -> None:
    c.setStrokeColor(text_color)
    c.setLineWidth(1)
    y = 30 * mm
    c.line(25 * mm, y, 80 * mm, y)
    c.line(w - 80 * mm, y, w - 25 * mm, y)
    c.setFont("Helvetica", 8)
    c.setFillColor(text_color)
    c.drawString(25 * mm, y - 10, "Signature")
    c.drawString(w - 80 * mm, y - 10, "Signature")
    _draw_signature_scribble(c, 28 * mm, y + 6, 45 * mm, text_color)
    _draw_signature_scribble(c, w - 78 * mm, y + 6, 45 * mm, text_color)
    c.setFont("Helvetica", 7)
    c.drawString(25 * mm, y - 20, signature_name)
    c.drawString(w - 80 * mm, y - 20, signature_name)


def _draw_signature_scribble(c: canvas.Canvas, x: float, y: float, width: float, color: colors.Color) -> None:
    c.setStrokeColor(color)
    c.setLineWidth(0.8)
    p = c.beginPath()
    p.moveTo(x, y)
    p.curveTo(x + width * 0.2, y + 3, x + width * 0.4, y - 2, x + width * 0.6, y + 2)
    p.curveTo(x + width * 0.7, y - 1, x + width * 0.85, y + 3, x + width, y)
    c.drawPath(p, stroke=1, fill=0)


def _draw_stamp(
    c: canvas.Canvas,
    x: float,
    y: float,
    r: float,
    label: str,
    color: colors.Color,
) -> None:
    c.setStrokeColor(color)
    c.setLineWidth(1)
    c.circle(x, y, r, fill=0, stroke=1)
    c.circle(x, y, r - 3, fill=0, stroke=1)
    c.setFont("Helvetica", 7)
    c.setFillColor(color)
    c.drawCentredString(x, y + 2, label)


def _draw_text_layout(
    c: canvas.Canvas,
    w: float,
    h: float,
    text_color: colors.Color,
    *,
    lang: str,
    holder: str,
    institution: str,
    degree: str,
    program: str,
    location: str,
    issue_date: str,
    diploma_number: str,
    layout: str,
) -> None:
    c.setFillColor(text_color)

    title = "Urkunde" if lang == "de" else "Diploma"
    degree_line = degree if lang == "de" else f"{degree} Degree"
    awarded = "verliehen an" if lang == "de" else "awarded to"
    status = "erfolgreich abgeschlossen" if lang == "de" else "graduated"
    uni_label = "Hochschule" if lang == "de" else "University"
    prog_label = "Studiengang" if lang == "de" else "Program"
    loc_label = "Ort" if lang == "de" else "Location"
    date_label = "Datum" if lang == "de" else "Date"
    num_label = "Urkunden-Nr." if lang == "de" else "Diploma No."

    if layout == "classic_center":
        c.setFont("Helvetica-Bold", 20)
        c.drawCentredString(w / 2, h - 30 * mm, title)
        c.setFont("Helvetica", 12)
        c.drawCentredString(w / 2, h - 42 * mm, degree_line)
        c.setFont("Helvetica", 11)
        c.drawCentredString(w / 2, h - 60 * mm, awarded)
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(w / 2, h - 75 * mm, holder)
        c.setFont("Helvetica", 11)
        c.drawCentredString(w / 2, h - 92 * mm, f"{prog_label}: {program}")
        c.drawCentredString(w / 2, h - 106 * mm, f"{uni_label}: {institution}")
        c.drawCentredString(w / 2, h - 120 * mm, f"Status: {status}")
        c.drawCentredString(w / 2, h - 134 * mm, f"{loc_label}: {location}  |  {date_label}: {issue_date}")
        c.drawCentredString(w / 2, h - 148 * mm, f"{num_label}: {diploma_number}")
        return

    if layout == "modern_side":
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(w / 2, h - 30 * mm, title)
        c.setFont("Helvetica", 12)
        c.drawCentredString(w / 2, h - 45 * mm, degree_line)
        c.setFont("Helvetica", 11)
        c.drawCentredString(w / 2, h - 65 * mm, awarded)
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(w / 2, h - 82 * mm, holder)
        c.setFont("Helvetica", 11)
        c.drawCentredString(w / 2, h - 100 * mm, f"{prog_label}: {program}")
        c.drawCentredString(w / 2, h - 114 * mm, f"{uni_label}: {institution}")
        c.drawCentredString(w / 2, h - 128 * mm, f"Status: {status}")
        x = w - 70 * mm
        y = h - 85 * mm
        c.setFont("Helvetica", 10)
        c.drawString(x, y, f"{date_label}: {issue_date}")
        c.drawString(x, y - 14, f"{loc_label}: {location}")
        c.drawString(x, y - 28, f"{num_label}: {diploma_number}")
        return

    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(w / 2, h - 30 * mm, title)
    c.setFont("Helvetica", 12)
    c.drawCentredString(w / 2, h - 45 * mm, degree_line)
    left_x = 25 * mm
    right_x = w / 2 + 10 * mm
    y = h - 70 * mm
    c.setFont("Helvetica", 11)
    c.drawString(left_x, y, f"{uni_label}: {institution}")
    c.drawString(left_x, y - 16, f"{prog_label}: {program}")
    c.drawString(left_x, y - 32, f"Status: {status}")
    c.setFont("Helvetica-Bold", 13)
    c.drawString(right_x, y, holder)
    c.setFont("Helvetica", 11)
    c.drawString(right_x, y - 16, f"{loc_label}: {location}")
    c.drawString(right_x, y - 32, f"{date_label}: {issue_date}")
    c.drawString(right_x, y - 48, f"{num_label}: {diploma_number}")

def generate_diplomas(
    output_dir: Path,
    *,
    count: int = 20,
    seed: int = 7,
    lang: str = "de",
    certified_copy_rate: float = 0.2,
    layout: str = "random",
    text_layout: str = "random",
    stamp_rate: float = 0.6,
) -> None:
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, count + 1):
        if lang == "both":
            lang_choice = "de" if i % 2 == 0 else "en"
        else:
            lang_choice = lang

        name_pool = random.choice(list(NAME_POOLS.values()))
        holder = f"{random.choice(name_pool['first'])} {random.choice(name_pool['last'])}"
        if lang_choice == "de":
            institution = random.choice(INSTITUTIONS_DE)
            program = random.choice(PROGRAMS_DE)
            degree = random.choice(DEGREE_TYPES_DE)
            location = random.choice(LOCATIONS_DE)
        else:
            region = random.choice(["en", "in", "af"])
            if region == "in":
                institution = random.choice(INSTITUTIONS_IN)
                program = random.choice(PROGRAMS_EN)
                degree = random.choice(DEGREE_TYPES_IN)
                location = random.choice(LOCATIONS_IN)
            elif region == "af":
                institution = random.choice(INSTITUTIONS_AF)
                program = random.choice(PROGRAMS_EN)
                degree = random.choice(DEGREE_TYPES_AF)
                location = random.choice(LOCATIONS_AF)
            else:
                institution = random.choice(INSTITUTIONS_EN)
                program = random.choice(PROGRAMS_EN)
                degree = random.choice(DEGREE_TYPES_EN)
                location = random.choice(LOCATIONS_EN)

        issue_date = _rand_date(2015, 2025)
        diploma_number = _diploma_number()
        certified_copy = random.random() < certified_copy_rate
        stamp_label = None
        if random.random() < stamp_rate:
            stamp_label = random.choice(
                [
                    "Bayern",
                    "NRW",
                    "Hessen",
                    "Sachsen",
                    "Berlin",
                    "Baden-Wuertt.",
                    "Hamburg",
                    "Niedersachsen",
                ]
            )
        layout_choice = layout if layout != "random" else random.choice(
            ["classic", "minimal", "modern", "dark", "playful", "qr"]
        )
        text_layout_choice = text_layout if text_layout != "random" else random.choice(
            ["classic_center", "modern_side", "split_columns"]
        )
        signature_name = f"{random.choice(name_pool['first'])} {random.choice(name_pool['last'])}"

        path = output_dir / f"diploma_{i:03d}_{lang_choice}.pdf"
        _write_pdf(
            path,
            lang=lang_choice,
            holder=holder,
            institution=institution,
            degree=degree,
            program=program,
            location=location,
            issue_date=issue_date,
            diploma_number=diploma_number,
            certified_copy=certified_copy,
            layout=layout_choice,
            text_layout=text_layout_choice,
            stamp_label=stamp_label,
            signature_name=signature_name,
        )
        print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic diploma PDFs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lang", choices=["de", "en", "both"], default="de")
    parser.add_argument("--certified-copy-rate", type=float, default=0.2)
    parser.add_argument(
        "--layout",
        choices=["classic", "minimal", "modern", "dark", "playful", "qr", "random"],
        default="random",
    )
    parser.add_argument(
        "--text-layout",
        choices=["classic_center", "modern_side", "split_columns", "random"],
        default="random",
    )
    parser.add_argument("--stamp-rate", type=float, default=0.6)
    args = parser.parse_args()

    generate_diplomas(
        args.output_dir,
        count=args.count,
        seed=args.seed,
        lang=args.lang,
        certified_copy_rate=args.certified_copy_rate,
        layout=args.layout,
        text_layout=args.text_layout,
        stamp_rate=args.stamp_rate,
    )


if __name__ == "__main__":
    main()
