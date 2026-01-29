"""Generate Fehlerprotokoll PDF samples from the company universe database."""

from __future__ import annotations

import argparse
import calendar
import random
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Sequence

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Table, TableStyle


DB_PATH = Path(__file__).parents[2] / "company_universe" / "db" / "company.sqlite"


@dataclass(frozen=True)
class Company:
    name: str
    street: str
    street_no: str
    zipcode: str
    town: str
    ik: str


@dataclass(frozen=True)
class Customer:
    id: int
    first_name: str
    last_name: str
    nationality: str


@dataclass(frozen=True)
class Service:
    id: int
    code: str
    legal_ref: str


@dataclass(frozen=True)
class Position:
    last_name: str
    first_name: str
    period_start: date
    period_end: date
    service_label: str


def _connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path))


def _load_company(conn: sqlite3.Connection) -> Company:
    row = conn.execute(
        "SELECT name, street, street_no, zipcode, town, ik FROM company ORDER BY id LIMIT 1"
    ).fetchone()
    if not row:
        raise RuntimeError("No company found in database")
    name, street, street_no, zipcode, town, ik = row
    return Company(
        name=name,
        street=street or "",
        street_no=street_no or "",
        zipcode=zipcode or "",
        town=town or "",
        ik=ik or "",
    )


def _load_customers(conn: sqlite3.Connection, *, nationality: str) -> list[Customer]:
    rows = conn.execute(
        "SELECT id, first_name, last_name, nationality FROM customer WHERE nationality = ?",
        (nationality,),
    ).fetchall()
    customers = [Customer(id=row[0], first_name=row[1], last_name=row[2], nationality=row[3]) for row in rows]
    if not customers:
        raise RuntimeError(f"No customers found for nationality {nationality}")
    return customers


def _load_services(conn: sqlite3.Connection) -> list[Service]:
    rows = conn.execute("SELECT id, code, legal_ref FROM service ORDER BY id").fetchall()
    services = [Service(id=row[0], code=row[1], legal_ref=row[2] or "") for row in rows]
    if not services:
        raise RuntimeError("No services found in database")
    return services


def _load_customer_services(conn: sqlite3.Connection) -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = {}
    rows = conn.execute("SELECT customer_id, service_id FROM customer_service").fetchall()
    for customer_id, service_id in rows:
        mapping.setdefault(customer_id, []).append(service_id)
    return mapping


def _month_start_end(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    end = date(year, month, calendar.monthrange(year, month)[1])
    return start, end


def _previous_month(year: int, month: int) -> tuple[int, int]:
    if month == 1:
        return year - 1, 12
    return year, month - 1


def _sample_rate(min_pct: float, max_pct: float) -> float:
    return random.uniform(min_pct, max_pct)


def _pick_customer_subset(customers: Sequence[Customer], pct: float) -> set[int]:
    count = max(1, round(len(customers) * pct))
    ids = [c.id for c in customers]
    random.shuffle(ids)
    return set(ids[:count])


def _add_months(year: int, month: int, offset: int) -> tuple[int, int]:
    total = (year * 12 + (month - 1)) + offset
    return total // 12, total % 12 + 1


def _ensure_signature_statuses(
    conn: sqlite3.Connection,
    customers: Sequence[Customer],
    customer_services: dict[int, list[int]],
    *,
    period_year: int,
    period_month: int,
    seed: int | None,
) -> dict[tuple[int, int], set[int]]:
    if seed is not None:
        random.seed(seed)

    customer_ids = tuple(c.id for c in customers)
    if not customer_ids:
        return {}

    last_year, last_month = _previous_month(period_year, period_month)
    placeholders = ",".join("?" for _ in customer_ids)

    def fetch_unsigned_by_period_before(year: int, month: int) -> dict[tuple[int, int], set[int]]:
        rows = conn.execute(
            f"SELECT DISTINCT period_year, period_month, customer_id "
            f"FROM customer_service_signature "
            f"WHERE customer_id IN ({placeholders}) AND signed = 0 AND "
            f"((period_year < ?) OR (period_year = ? AND period_month < ?))",
            (*customer_ids, year, year, month),
        ).fetchall()
        out: dict[tuple[int, int], set[int]] = {}
        for pyear, pmonth, customer_id in rows:
            out.setdefault((pyear, pmonth), set()).add(customer_id)
        return out

    now = datetime.now().isoformat(timespec="seconds")

    def insert_status(customer_id: int, service_id: int, year: int, month: int, signed: bool) -> None:
        conn.execute(
            "INSERT INTO customer_service_signature (customer_id, service_id, period_year, period_month, signed, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (customer_id, service_id, year, month, int(signed), now),
        )

    # 1) Resolve older months (period < M) for 80-95% of customers that are still unsigned there.
    unsigned_customers_older = conn.execute(
        f"SELECT DISTINCT customer_id FROM customer_service_signature "
        f"WHERE customer_id IN ({placeholders}) AND signed = 0 AND "
        f"((period_year < ?) OR (period_year = ? AND period_month < ?))",
        (*customer_ids, last_year, last_year, last_month),
    ).fetchall()
    unsigned_customer_ids = [row[0] for row in unsigned_customers_older]
    if unsigned_customer_ids:
        p_clear = random.uniform(0.80, 0.95)
        resolved_count = round(len(unsigned_customer_ids) * p_clear)
        resolved_customers = set(
            random.sample(unsigned_customer_ids, k=max(0, resolved_count))
        )
        if resolved_customers:
            resolved_placeholders = ",".join("?" for _ in resolved_customers)
            conn.execute(
                f"UPDATE customer_service_signature SET signed = 1, updated_at = ? "
                f"WHERE customer_id IN ({resolved_placeholders}) AND "
                f"((period_year < ?) OR (period_year = ? AND period_month < ?))",
                (now, *resolved_customers, last_year, last_year, last_month),
            )

    # 2) Create missing statuses for M (CM-1) with 5-15% unsigned.
    existing_last = conn.execute(
        f"SELECT 1 FROM customer_service_signature "
        f"WHERE customer_id IN ({placeholders}) AND period_year = ? AND period_month = ? LIMIT 1",
        (*customer_ids, last_year, last_month),
    ).fetchone()
    if not existing_last:
        unsigned_rate = random.uniform(0.05, 0.15)
        unsigned_customers = _pick_customer_subset(customers, unsigned_rate)
        for customer in customers:
            services = customer_services.get(customer.id, [])
            for service_id in services:
                signed = customer.id not in unsigned_customers
                insert_status(customer.id, service_id, last_year, last_month, signed)

    conn.commit()
    return fetch_unsigned_by_period_before(period_year, period_month)


def _build_positions(
    customers: Sequence[Customer],
    services: dict[int, Service],
    customer_services: dict[int, list[int]],
    unsigned_by_period: dict[tuple[int, int], set[int]],
) -> list[Position]:
    positions: list[Position] = []

    customer_lookup = {c.id: c for c in customers}

    for (year, month), unsigned_customers in unsigned_by_period.items():
        if not unsigned_customers:
            continue
        start, end = _month_start_end(year, month)
        for customer_id in unsigned_customers:
            customer = customer_lookup.get(customer_id)
            if not customer:
                continue
            for service_id in customer_services.get(customer_id, []):
                service = services.get(service_id)
                if not service:
                    continue
                service_label = f"§ {service.code} {service.legal_ref}".strip()
                positions.append(
                    Position(
                        last_name=customer.last_name,
                        first_name=customer.first_name,
                        period_start=start,
                        period_end=end,
                        service_label=service_label,
                    )
                )

    positions.sort(key=lambda p: (p.last_name.lower(), p.first_name.lower(), p.period_start, p.service_label))
    return positions


def generate_fehlerprotokoll_rechnung_pdf(
    output_path: Path,
    *,
    report_date: date,
    company: Company,
    positions: Sequence[Position],
    module: str = "Rechnungsautomatik",
) -> Path:
    datum_str = report_date.strftime("%d.%m.%Y %H:%M")

    styles = getSampleStyleSheet()

    h_main_size = 15
    h_sub_size = 11
    h_meta_size = 9.5

    header_cell_style = ParagraphStyle("header_cell", parent=styles["Normal"], fontSize=8.5, leading=10)
    cell_style = ParagraphStyle("cell", parent=styles["Normal"], fontSize=7.2, leading=8.6)

    doc = BaseDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=34 * mm,
        bottomMargin=10 * mm,
    )

    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal", showBoundary=0)

    def draw_header(canvas, doc_):
        canvas.saveState()

        x_left = doc_.leftMargin
        x_right = doc_.leftMargin + doc_.width * 0.62
        y_top = doc_.pagesize[1] - 10 * mm

        canvas.setFont("Helvetica-Bold", h_main_size)
        canvas.drawString(x_left, y_top, "Fehlerprotokoll")

        canvas.setFont("Helvetica", h_sub_size)
        canvas.drawString(x_left, y_top - 6.5 * mm, module)

        canvas.setFont("Helvetica", h_meta_size)
        canvas.drawString(x_left, y_top - 12 * mm, f"Datum: {datum_str}")
        canvas.drawString(x_left, y_top - 16.5 * mm, f"Seite: {canvas.getPageNumber()}")

        canvas.setFont("Helvetica", h_sub_size)
        canvas.drawString(x_right, y_top, company.name)

        canvas.setFont("Helvetica", h_meta_size)
        canvas.drawString(x_right, y_top - 6.5 * mm, f"{company.street} {company.street_no}".strip())
        canvas.drawString(x_right, y_top - 11 * mm, f"{company.zipcode} {company.town}".strip())
        canvas.drawString(x_right, y_top - 15.5 * mm, f"IK: {company.ik}")

        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="pt", frames=[frame], onPage=draw_header)])

    table_data = [
        [
            Paragraph("<b>Position</b>", header_cell_style),
            Paragraph("<b>Fehlertext</b>", header_cell_style),
        ]
    ]

    error_text = (
        "Für einige Leistungen liegen keine Unterschriften vor. Bitte nutzen Sie die manuelle Druckfunktion, "
        "um die Leistungsnachweise zu drucken."
    )

    fmt = lambda d: d.strftime("%d.%m.%Y")

    for idx, pos in enumerate(positions, start=1):
        first_line = (
            f"Rechnung für {pos.last_name}, {pos.first_name} "
            f"({fmt(pos.period_start)} – {fmt(pos.period_end)} {pos.service_label})"
        )
        fehlertext = f"{first_line}<br/><nobr>{error_text}</nobr>"
        table_data.append([Paragraph(str(idx), cell_style), Paragraph(fehlertext, cell_style)])

    col_pos = 18 * mm
    col_text = doc.width - col_pos
    table = Table(table_data, colWidths=[col_pos, col_text], repeatRows=1)

    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 1), (0, -1), "RIGHT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 1.4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.4),
            ]
        )
    )

    doc.build([table])
    return output_path


def _parse_report_date(value: str | None) -> date:
    if not value:
        return date.today()
    return datetime.strptime(value, "%Y-%m-%d").date()


def generate_fehlerprotokoll_reports(
    output_dir: Path,
    *,
    report_date: date | None = None,
    seed: int | None = None,
) -> Path:
    report_date = report_date or date.today()

    with _connect(DB_PATH) as conn:
        company = _load_company(conn)
        customers = _load_customers(conn, nationality="DE")
        services = {svc.id: svc for svc in _load_services(conn)}
        customer_services = _load_customer_services(conn)

        unsigned_by_period = _ensure_signature_statuses(
            conn,
            customers,
            customer_services,
            period_year=report_date.year,
            period_month=report_date.month,
            seed=seed,
        )

    positions = _build_positions(customers, services, customer_services, unsigned_by_period)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"Fehlerprotokoll_Rechnung_{report_date.year}_{report_date.month:02d}.pdf"
    generate_fehlerprotokoll_rechnung_pdf(
        output_path,
        report_date=datetime.combine(report_date, datetime.now().time()),
        company=company,
        positions=positions,
    )
    print(f"Wrote {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Fehlerprotokoll PDFs from DB data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-date", default=None, help="Report date (YYYY-MM-DD)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    report_date = _parse_report_date(args.report_date)
    generate_fehlerprotokoll_reports(args.output_dir, report_date=report_date, seed=args.seed)


if __name__ == "__main__":
    main()
