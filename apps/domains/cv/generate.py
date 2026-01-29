"""Generate synthetic CV samples with multiple layout styles."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from datetime import date

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


FIRST_NAMES = [
    "Anna", "Ben", "Clara", "David", "Emma", "Felix", "Greta", "Hannah",
    "Jonas", "Lena", "Marie", "Noah", "Paul", "Sofie", "Tim", "Laura",
]
LAST_NAMES = [
    "Müller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Wagner",
    "Becker", "Hoffmann", "Schäfer", "Koch", "Bauer", "Richter", "Klein",
]
CITIES = ["Berlin", "Hamburg", "München", "Köln", "Frankfurt", "Stuttgart"]

PROFESSIONS = {
    "Pflegefachkraft": {
        "skills": [
            "Pflegeplanung",
            "Wundversorgung",
            "Medikamentengabe",
            "Vitalzeichenkontrolle",
            "Hygienestandards",
            "Palliativpflege",
            "Patientenkommunikation",
        ],
        "experience": [
            "Grund- und Behandlungspflege",
            "Dokumentation in Pflegesystemen",
            "Interdisziplinäre Zusammenarbeit",
        ],
    },
    "Intensivpfleger/in": {
        "skills": [
            "Intensivpflege",
            "Beatmungsmanagement",
            "EKG-Überwachung",
            "Notfallmanagement",
            "Sedierungsmanagement",
        ],
        "experience": [
            "Überwachung kritisch kranker Patienten",
            "Bedienung und Kontrolle von Beatmungsgeräten",
            "Notfallinterventionen im ICU-Setting",
        ],
    },
    "Pflegefachkraft (OP)": {
        "skills": [
            "OP-Assistenz",
            "Sterilgutaufbereitung",
            "Instrumentenkunde",
            "Hygienestandards",
            "Patientenlagerung",
        ],
        "experience": [
            "Assistenz während chirurgischer Eingriffe",
            "Vorbereitung und Nachbereitung von OP-Sälen",
            "Steriles Arbeiten im OP",
        ],
    },
    "Pflegefachkraft (Anästhesie)": {
        "skills": [
            "Anästhesieassistenz",
            "Überwachung der Vitalparameter",
            "Narkosevorbereitung",
            "Atemwegsmanagement",
            "Notfallmanagement",
        ],
        "experience": [
            "Assistenz bei Einleitung der Anästhesie",
            "Überwachung im Aufwachraum",
            "Vorbereitung von Narkosegeräten",
        ],
    },
    "Pflegefachkraft (Geriatrie)": {
        "skills": [
            "Geriatrische Pflege",
            "Mobilisation",
            "Demenzbetreuung",
            "Sturzprophylaxe",
            "Palliativpflege",
        ],
        "experience": [
            "Pflege älterer Patienten",
            "Mobilisations- und Rehabilitationsmaßnahmen",
            "Kommunikation mit Angehörigen",
        ],
    },
    "Pflegefachkraft (Pädiatrie)": {
        "skills": [
            "Pädiatrische Pflege",
            "Elternberatung",
            "Schmerzmanagement",
            "Medikamentengabe",
            "Hygienestandards",
        ],
        "experience": [
            "Pflege von Kindern und Jugendlichen",
            "Unterstützung bei Diagnostik und Therapie",
            "Begleitung von Eltern",
        ],
    },
    "Pflegefachkraft (Onkologie)": {
        "skills": [
            "Onkologische Pflege",
            "Chemotherapie-Begleitung",
            "Nebenwirkungsmanagement",
            "Schmerzmanagement",
            "Palliativpflege",
        ],
        "experience": [
            "Begleitung onkologischer Therapien",
            "Überwachung von Nebenwirkungen",
            "Beratung von Patienten und Angehörigen",
        ],
    },
    "Pflegefachkraft (Psychiatrie)": {
        "skills": [
            "Psychiatrische Pflege",
            "Deeskalation",
            "Gesprächsführung",
            "Krisenintervention",
            "Dokumentation",
        ],
        "experience": [
            "Betreuung psychiatrischer Patienten",
            "Durchführung von Kriseninterventionen",
            "Arbeit im multiprofessionellen Team",
        ],
    },
    "Pflegefachkraft (Wundmanagement)": {
        "skills": [
            "Wundversorgung",
            "Wunddokumentation",
            "Wundassessment",
            "Hygienestandards",
            "Materialkunde",
        ],
        "experience": [
            "Versorgung chronischer Wunden",
            "Erstellung von Wunddokumentationen",
            "Beratung zur Wundpflege",
        ],
    },
    "Pflegefachkraft (Ambulant/Homecare)": {
        "skills": [
            "Ambulante Pflege",
            "Tourenplanung",
            "Medikamentengabe",
            "Wundversorgung",
            "Patientenkommunikation",
        ],
        "experience": [
            "Durchführung ambulanter Pflegeeinsätze",
            "Dokumentation im mobilen Pflegedienst",
            "Beratung von Patienten und Angehörigen",
        ],
    },
    "Medizinische/r Fachangestellte/r": {
        "skills": [
            "Patientenaufnahme",
            "Impfmanagement",
            "EKG-Assistenz",
            "Blutentnahme",
            "Dokumentation",
        ],
        "experience": [
            "Terminmanagement und Patientenbetreuung",
            "Assistenz bei Untersuchungen",
            "Praxisorganisation",
        ],
    },
    "Pharmazeutisch-technische/r Assistent/in": {
        "skills": [
            "Arzneimittelkenntnisse",
            "Rezepturherstellung",
            "Warenwirtschaft",
            "Beratung zu OTC-Arzneimitteln",
        ],
        "experience": [
            "Herstellung von Rezepturen",
            "Kundenberatung in der Apotheke",
            "Qualitätskontrolle von Arzneimitteln",
        ],
    },
    "Medizinisch-technische/r Assistent/in": {
        "skills": [
            "Laborassistenz",
            "Probenaufbereitung",
            "Gerätewartung",
            "Qualitätssicherung",
        ],
        "experience": [
            "Durchführung labordiagnostischer Verfahren",
            "Dokumentation von Messergebnissen",
            "Wartung und Kalibrierung von Analysegeräten",
        ],
    },
    "OP-Assistenz": {
        "skills": [
            "OP-Assistenz",
            "Sterilgutaufbereitung",
            "Instrumentenkunde",
            "Hygienestandards",
        ],
        "experience": [
            "Vorbereitung und Nachbereitung von OP-Sälen",
            "Assistenz während chirurgischer Eingriffe",
            "Steriles Arbeiten im OP",
        ],
    },
    "Anästhesietechnische/r Assistent/in": {
        "skills": [
            "Anästhesieassistenz",
            "Überwachung der Vitalparameter",
            "Notfallmanagement",
            "Medizintechnik",
        ],
        "experience": [
            "Vorbereitung von Narkosegeräten",
            "Assistenz bei der Einleitung der Anästhesie",
            "Überwachung während Eingriffen",
        ],
    },
    "Notfallsanitäter/in": {
        "skills": [
            "Notfallmanagement",
            "Erste Hilfe",
            "Traumaversorgung",
            "Patientenkommunikation",
        ],
        "experience": [
            "Einsätze im Rettungsdienst",
            "Erstversorgung und Stabilisierung",
            "Kommunikation mit Leitstellen und Kliniken",
        ],
    },
    "Assistenzärztin/Assistenzarzt": {
        "skills": [
            "Anamnese",
            "Diagnostik",
            "Therapieplanung",
            "Visitenführung",
            "Patientenaufklärung",
        ],
        "experience": [
            "Durchführung stationärer Visiten",
            "Aufnahme und Diagnostik von Patienten",
            "Dokumentation und Arztbriefe",
        ],
    },
    "Fachärztin/Facharzt": {
        "skills": [
            "Facharztdiagnostik",
            "Therapieentscheidungen",
            "Supervision",
            "Qualitätsmanagement",
            "Interdisziplinäre Fallbesprechungen",
        ],
        "experience": [
            "Leitung komplexer Behandlungsfälle",
            "Supervision des ärztlichen Teams",
            "Koordination interdisziplinärer Maßnahmen",
        ],
    },
    "Oberärztin/Oberarzt": {
        "skills": [
            "Klinische Leitung",
            "OP-Freigaben",
            "Personalführung",
            "Risikomanagement",
            "Leitlinienumsetzung",
        ],
        "experience": [
            "Leitung einer Station/Abteilung",
            "Fachliche Verantwortung im OP",
            "Mentoring von Assistenzärzten",
        ],
    },
    "Fachärztin/Facharzt (Chirurgie)": {
        "skills": [
            "Präoperative Diagnostik",
            "OP-Planung",
            "Chirurgische Assistenz",
            "Postoperative Betreuung",
            "Wundmanagement",
        ],
        "experience": [
            "Durchführung chirurgischer Eingriffe",
            "Prä- und postoperative Visiten",
            "Zusammenarbeit mit OP-Teams",
        ],
    },
    "Fachärztin/Facharzt (Anästhesie)": {
        "skills": [
            "Anästhesieplanung",
            "Narkoseführung",
            "Monitoring",
            "Schmerztherapie",
            "Notfallmanagement",
        ],
        "experience": [
            "Einleitung und Überwachung von Anästhesien",
            "Postoperative Schmerztherapie",
            "Notfallmanagement im OP",
        ],
    },
    "Fachärztin/Facharzt (Innere Medizin)": {
        "skills": [
            "Internistische Diagnostik",
            "Therapieplanung",
            "EKG-Interpretation",
            "Laborbefundung",
            "Chronische Erkrankungen",
        ],
        "experience": [
            "Betreuung internistischer Stationen",
            "Durchführung von Visiten und Diagnostik",
            "Koordination interdisziplinärer Behandlungen",
        ],
    },
    "Fachärztin/Facharzt (Pädiatrie)": {
        "skills": [
            "Pädiatrische Diagnostik",
            "Impfmanagement",
            "Vorsorgeuntersuchungen",
            "Elternberatung",
            "Notfallmanagement",
        ],
        "experience": [
            "Betreuung pädiatrischer Patienten",
            "Durchführung von U-Untersuchungen",
            "Akutversorgung in der Pädiatrie",
        ],
    },
    "Fachärztin/Facharzt (Kardiologie)": {
        "skills": [
            "EKG-Interpretation",
            "Echokardiographie",
            "Herzkatheter",
            "Risikobeurteilung",
            "Therapieplanung",
        ],
        "experience": [
            "Diagnostik und Behandlung kardialer Erkrankungen",
            "Durchführung von kardiologischen Untersuchungen",
            "Betreuung von Herzkatheterpatienten",
        ],
    },
    "Fachärztin/Facharzt (Neurologie)": {
        "skills": [
            "Neurologische Diagnostik",
            "EEG/EMG",
            "Schlaganfallmanagement",
            "Bildgebungsauswertung",
            "Therapieplanung",
        ],
        "experience": [
            "Betreuung neurologischer Patienten",
            "Diagnostik bei neurologischen Krankheitsbildern",
            "Stroke-Unit-Management",
        ],
    },
}
LANGS = ["Deutsch", "English", "Französisch"]
QUALIFICATIONS = [
    "Führerschein Klasse B",
    "Hygieneschulung",
    "Basiskurs Wundversorgung",
    "Reanimationsschulung (BLS/AED)",
    "Medikamentenmanagement",
    "Strahlenschutz-Unterweisung",
    "Erste-Hilfe-Kurs",
]

SECTION_HEADERS = {
    "profil": ["Profil", "Kurzprofil", "Über mich"],
    "experience": ["Berufserfahrung", "Erfahrung", "Work Experience"],
    "education": ["Ausbildung", "Bildung", "Education"],
    "skills": ["Skills", "Kompetenzen", "Fähigkeiten"],
    "languages": ["Sprachen", "Languages"],
    "qualifications": ["Qualifikationen", "Zusatzqualifikationen", "Qualifications"],
}

SUMMARY_LINES = [
    "Engagierte Fachkraft mit hoher Patientenorientierung.",
    "Erfahrung in interdisziplinären Teams und klinischen Abläufen.",
    "Zuverlässig, strukturiert und empathisch im Umgang mit Patienten.",
]

BULLET_STYLES = ["- ", "• ", "* "]


def _rand_date(year_start: int, year_end: int) -> str:
    year = random.randint(year_start, year_end)
    month = random.randint(1, 12)
    return f"{month:02d}.{year}"


def _maybe_section(text: str, chance: float) -> str:
    return text if random.random() < chance else ""


def _cv_text(first: str, last: str, cv_type: str) -> str:
    city = random.choice(CITIES)
    email = f"{first.lower()}.{last.lower()}@example.com"
    phone = f"+49 170 {random.randint(1000000, 9999999)}"
    profession = random.choice(list(PROFESSIONS.keys()))
    prof = PROFESSIONS[profession]
    skills = ", ".join(random.sample(prof["skills"], k=min(4, len(prof["skills"]))))
    languages = ", ".join(random.sample(LANGS, k=2))
    bullet = random.choice(BULLET_STYLES)
    exp_lines = "\n".join(
        f"{bullet}{line}" for line in random.sample(prof["experience"], k=min(3, len(prof["experience"])))
    )
    qualifications = ", ".join(random.sample(QUALIFICATIONS, k=2))
    summary = random.choice(SUMMARY_LINES)

    h_profil = random.choice(SECTION_HEADERS["profil"])
    h_exp = random.choice(SECTION_HEADERS["experience"])
    h_edu = random.choice(SECTION_HEADERS["education"])
    h_skills = random.choice(SECTION_HEADERS["skills"])
    h_langs = random.choice(SECTION_HEADERS["languages"])
    h_quals = random.choice(SECTION_HEADERS["qualifications"])

    education = random.choice(
        [
            f"Pflegefachmann/Pflegefachfrau, Berufsfachschule {city}",
            f"Gesundheits- und Krankenpflege, Akademie {city}",
            f"B.Sc. Pflegewissenschaft, Hochschule {city}",
            f"MTA-Ausbildung, Medizinische Schule {city}",
            f"PTA-Ausbildung, Berufskolleg {city}",
            f"Medizinische Fachangestellte/r, Berufsschule {city}",
            f"B.Sc. Gesundheitsmanagement, Hochschule {city}",
        ]
    )

    if cv_type == "europass":
        return f"""EUROPASS
{first} {last}

Personal Information
Address: {city}, Deutschland
Email: {email}
Telephone: {phone}

Work Experience
{_rand_date(2018, 2020)} - {_rand_date(2021, 2024)}  {profession}, Beispiel Klinik
Main activities:
{exp_lines}

Education and Training
{_rand_date(2014, 2017)} - {_rand_date(2017, 2018)}  {education}

Personal skills
{skills}

Languages
{languages}

Additional Information
{qualifications}
"""
    if cv_type == "modern":
        return f"""{first} {last} | {city}, DE | {email} | {phone}

SUMMARY
{summary}

EXPERIENCE
{_rand_date(2019, 2021)} – {_rand_date(2022, 2024)}  {profession}, Beispiel Klinik
{bullet}{prof["experience"][0]}
{bullet}{prof["experience"][1]}

EDUCATION
{_rand_date(2014, 2017)} – {_rand_date(2017, 2018)}  {education}

SKILLS
{skills}

LANGUAGES
{languages}

QUALIFICATIONS
{qualifications}
"""
    if cv_type == "academic":
        return f"""{first} {last}
{city}, Deutschland | {email} | {phone}

Academic CV

Research Interests
Klinische Versorgung, Prozessoptimierung, Patientensicherheit.

Education
{_rand_date(2010, 2013)} - {_rand_date(2013, 2015)}  {education}

Publications
- {last}, {first}. "Qualitätsmanagement in der Pflege" (2022)

Teaching
- Pflegeprozesse (2021-2023)

Skills
{skills}

Languages
{languages}

Qualifications
{qualifications}
"""
    # default: lebenslauf
    profil_section = _maybe_section(f"\n{h_profil}\n{summary}\n", 0.9)
    skills_section = _maybe_section(f"\n{h_skills}\n{skills}\n", 0.95)
    langs_section = _maybe_section(f"\n{h_langs}\n{languages}\n", 0.9)
    quals_section = _maybe_section(f"\n{h_quals}\n{qualifications}\n", 0.85)

    return f"""{first} {last}
{city}, Deutschland
Email: {email}
Telefon: {phone}

{profil_section}

{h_exp}
{_rand_date(2018, 2020)} - {_rand_date(2021, 2024)}  {profession}, Beispiel Klinik
{exp_lines}

{h_edu}
{_rand_date(2014, 2017)} - {_rand_date(2017, 2018)}  {education}

{skills_section}
{langs_section}
{quals_section}
"""


def _write_pdf(text: str, path: Path, *, cv_type: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    base_left = 18 * mm
    right = 18 * mm
    top = height - 18 * mm
    line_height = 12
    content_left = base_left
    content_right = width - right

    # Reserve columns/areas based on CV type.
    if cv_type == "lebenslauf":
        content_left = base_left + 50 * mm + 6 * mm
    if cv_type == "europass":
        content_left = base_left + 45 * mm + 6 * mm
    if cv_type == "modern":
        content_right = width - right - 60 * mm - 6 * mm

    max_width = content_right - content_left

    lines = text.splitlines()
    name_line = lines[0] if lines else ""

    # Header (wrap to avoid overlap)
    def _draw_header() -> float:
        c.setFont("Helvetica-Bold", 16)
        header_max = content_right - content_left
        name_words = name_line.split(" ")
        line = []
        y = top
        for word in name_words:
            trial = " ".join(line + [word]) if line else word
            if c.stringWidth(trial, "Helvetica-Bold", 16) <= header_max:
                line.append(word)
            else:
                c.drawString(content_left, y, " ".join(line))
                y -= 18
                line = [word]
        if line:
            c.drawString(content_left, y, " ".join(line))
            y -= 18
        c.setFont("Helvetica", 10)
        c.drawString(content_left, y, "Curriculum Vitae")
        y -= 6
        c.setStrokeColorRGB(0.2, 0.2, 0.2)
        c.line(content_left, y, content_right, y)
        return y - 12

    # Lebenslauf: left sidebar for personal info.
    if cv_type == "lebenslauf":
        col_x = base_left
        col_y = top - 40 * mm
        col_w = 50 * mm
        col_h = 200 * mm
        c.rect(col_x, col_y, col_w, col_h)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(col_x + 4, col_y + col_h - 12, "Persönliche Daten")

    # Europass: left label column.
    if cv_type == "europass":
        col_x = base_left
        col_y = top - 40 * mm
        col_w = 45 * mm
        col_h = 200 * mm
        c.rect(col_x, col_y, col_w, col_h)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(col_x + 4, col_y + col_h - 12, "Europass")
        c.setFont("Helvetica", 8)
        c.drawString(col_x + 4, col_y + col_h - 24, "CV")

    # Modern layout: add a right column for summary/skills and photo placeholder.
    if cv_type == "modern":
        photo_x = width - right - 30 * mm
        photo_y = top - 20 * mm
        c.rect(photo_x, photo_y, 28 * mm, 36 * mm)
        c.setFont("Helvetica", 8)
        c.drawString(photo_x + 2, photo_y + 18, "Photo")

        # Right column box
        col_x = width - right - 60 * mm
        col_y = top - 70 * mm
        col_w = 60 * mm
        col_h = 140 * mm
        c.rect(col_x, col_y, col_w, col_h)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(col_x + 4, col_y + col_h - 12, "Highlights")
        c.setFont("Helvetica", 8)
        c.drawString(col_x + 4, col_y + col_h - 24, "Skills & Languages")

    y = _draw_header()

    def draw_wrapped(line: str) -> None:
        nonlocal y
        words = line.split(" ")
        current = []
        for word in words:
            trial = " ".join(current + [word]) if current else word
            if c.stringWidth(trial, "Helvetica", 10) <= max_width:
                current.append(word)
            else:
                c.setFont("Helvetica", 10)
                c.drawString(content_left, y, " ".join(current))
                y -= line_height
                current = [word]
        if current:
            c.setFont("Helvetica", 10)
            c.drawString(content_left, y, " ".join(current))
            y -= line_height

    for raw_line in lines[1:]:
        line = raw_line.rstrip()
        if not line:
            y -= line_height
            continue
        # Section headers in ALL CAPS or title-like lines get emphasis
        if line.isupper() or line in {"Profil", "Berufserfahrung", "Ausbildung", "Skills", "Sprachen", "Qualifikationen"}:
            y -= 4
            c.setFont("Helvetica-Bold", 11)
            c.drawString(content_left, y, line)
            y -= line_height
            continue
        draw_wrapped(line)
        if y < 20 * mm:
            c.showPage()
            y = top
    c.save()


def generate_cvs(
    output_dir: Path,
    *,
    count: int = 5,
    cv_type: str = "lebenslauf",
    all_types: bool = False,
    seed: int = 7,
) -> None:
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    types = ["lebenslauf", "europass", "modern", "academic"] if all_types else [cv_type]
    counter = 1
    for chosen_type in types:
        for _ in range(count):
            first = random.choice(FIRST_NAMES)
            last = random.choice(LAST_NAMES)
            text = _cv_text(first, last, chosen_type)
            path = output_dir / f"cv_{chosen_type}_{counter:03d}.pdf"
            _write_pdf(text, path, cv_type=chosen_type)
            print(f"Wrote {path}")
            counter += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic CV samples")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument(
        "--type",
        choices=["lebenslauf", "europass", "modern", "academic"],
        default="lebenslauf",
        help="CV type to generate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all CV types (count per type)",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    generate_cvs(
        args.output_dir,
        count=args.count,
        cv_type=args.type,
        all_types=args.all,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
