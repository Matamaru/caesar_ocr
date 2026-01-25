import argparse
import calendar
import random
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Table, TableStyle


@dataclass
class CustomerRow:
    id: int
    first_name: str
    last_name: str


@dataclass
class ServiceRow:
    id: int
    code: str
    legal_ref: str


def _ensure_signature_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS customer_service_signature (
          id INTEGER PRIMARY KEY,
          customer_id INTEGER NOT NULL,
          service_id INTEGER NOT NULL,
          period_year INTEGER NOT NULL,
          period_month INTEGER NOT NULL,
          signed INTEGER NOT NULL,
          updated_at TEXT,
          FOREIGN KEY (customer_id) REFERENCES customer(id),
          FOREIGN KEY (service_id) REFERENCES service(id)
        );
        """
    )


def _fetch_company(conn: sqlite3.Connection):
    row = conn.execute(
        "SELECT name, street, street_no, zipcode, town, ik FROM company ORDER BY id LIMIT 1"
    ).fetchone()
    if not row:
        raise SystemExit("No company found in database")
    name, street, street_no, zipcode, town, ik = row
    street_line = f"{street} {street_no}".strip()
    city_line = " ".join(part for part in [zipcode, town] if part)
    return {
        "company": name,
        "street": street_line,
        "city": city_line,
        "ik": ik or "",
    }


def _fetch_customers(conn: sqlite3.Connection, nationality: str | None = None) -> list[CustomerRow]:
    if nationality:
        rows = conn.execute(
            "SELECT id, first_name, last_name FROM customer WHERE nationality = ? ORDER BY last_name, first_name",
            (nationality,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, first_name, last_name FROM customer ORDER BY last_name, first_name"
        ).fetchall()
    if not rows:
        raise SystemExit("No customers found in database")
    return [CustomerRow(*row) for row in rows]


def _ensure_services(conn: sqlite3.Connection) -> list[ServiceRow]:
    existing = conn.execute("SELECT id, code, legal_ref FROM service").fetchall()
    if existing:
        return [ServiceRow(*row) for row in existing]

    services = [
        ("45b", "SGB XI"),
        ("36", "SGB XI"),
        ("39", "SGB XI"),
    ]
    conn.executemany("INSERT INTO service (code, legal_ref, name) VALUES (?, ?, ?)", [(c, l, None) for c, l in services])
    conn.commit()
    return [ServiceRow(*row) for row in conn.execute("SELECT id, code, legal_ref FROM service").fetchall()]


def _assign_services(conn: sqlite3.Connection, customers: list[CustomerRow], services: list[ServiceRow], seed: int | None) -> None:
    if seed is not None:
        random.seed(seed)

    service_by_code = {s.code: s for s in services}
    all_assigned = conn.execute("SELECT customer_id FROM customer_service").fetchall()
    assigned_ids = {row[0] for row in all_assigned}

    for cust in customers:
        if cust.id in assigned_ids:
            continue
        # all customers get 45b
        assignments = [service_by_code["45b"].id]

        roll = random.random()
        if roll < 0.05:
            assignments.extend([service_by_code["36"].id, service_by_code["39"].id])
        elif roll < 0.45:
            assignments.append(random.choice([service_by_code["36"].id, service_by_code["39"].id]))

        conn.executemany(
            "INSERT INTO customer_service (customer_id, service_id) VALUES (?, ?)",
            [(cust.id, sid) for sid in assignments],
        )
    conn.commit()


def _prev_month(year: int, month: int, offset: int = 1) -> tuple[int, int]:
    y, m = year, month
    for _ in range(offset):
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return y, m


def _ensure_signature_statuses(
    conn: sqlite3.Connection,
    customers: list[CustomerRow],
    period_year: int,
    period_month: int,
    seed: int | None,
) -> None:
    if seed is not None:
        random.seed(seed + 1)

    p1y, p1m = _prev_month(period_year, period_month, 1)
    p2y, p2m = _prev_month(period_year, period_month, 2)

    total_customers = len(customers)
    unsigned_rate_last = random.uniform(0.05, 0.15)
    unsigned_rate_prev = random.uniform(0.05, 0.15)
    resolve_rate = random.uniform(0.80, 0.95)

    all_ids = [c.id for c in customers]
    random.shuffle(all_ids)
    unsigned_last = set(all_ids[: max(1, int(total_customers * unsigned_rate_last))])

    random.shuffle(all_ids)
    unsigned_prev = set(all_ids[: max(1, int(total_customers * unsigned_rate_prev))])
    # resolve most of prior errors
    unsigned_prev = set(random.sample(list(unsigned_prev), max(1, int(len(unsigned_prev) * (1 - resolve_rate))))) if unsigned_prev else set()

    assignments = conn.execute("SELECT customer_id, service_id FROM customer_service").fetchall()

    for cust_id, service_id in assignments:
        for (y, m, unsigned_set) in [(p1y, p1m, unsigned_last), (p2y, p2m, unsigned_prev)]:
            row = conn.execute(
                "SELECT id FROM customer_service_signature WHERE customer_id=? AND service_id=? AND period_year=? AND period_month=?",
                (cust_id, service_id, y, m),
            ).fetchone()
            if row:
                continue
            signed = 0 if cust_id in unsigned_set else 1
            conn.execute(
                "INSERT INTO customer_service_signature (customer_id, service_id, period_year, period_month, signed) VALUES (?, ?, ?, ?, ?)",
                (cust_id, service_id, y, m, signed),
            )
    conn.commit()


def _iter_unsigned_positions(
    conn: sqlite3.Connection,
    period_year: int,
    period_month: int,
) -> Iterable[tuple[CustomerRow, ServiceRow, int, int]]:
    p1y, p1m = _prev_month(period_year, period_month, 1)
    p2y, p2m = _prev_month(period_year, period_month, 2)
    rows = conn.execute(
        """
        SELECT c.id, c.first_name, c.last_name, s.id, s.code, s.legal_ref, sig.period_year, sig.period_month
        FROM customer_service_signature sig
        JOIN customer c ON c.id = sig.customer_id
        JOIN service s ON s.id = sig.service_id
        WHERE sig.signed = 0
          AND ((sig.period_year=? AND sig.period_month=?) OR (sig.period_year=? AND sig.period_month=?))
        ORDER BY c.last_name, c.first_name, sig.period_year, sig.period_month, s.code
        """,
        (p1y, p1m, p2y, p2m),
    ).fetchall()
    for row in rows:
        cust = CustomerRow(row[0], row[1], row[2])
        svc = ServiceRow(row[3], row[4], row[5])
        yield cust, svc, row[6], row[7]


def generate_fehlerprotokoll_rechnung_pdf(
    output_path: str,
    *,
    period_year: int = 2025,
    period_month: int = 12,
    datum_str: str = "21.01.2026 15:13",
    module: str = "Rechnungsautomatik",
    seed: int | None = None,
    db_path: Path,
) -> str:
    """
    Generate a Fehlerprotokoll PDF for beleg_type=Rechnung only.

    - Header (left/right) is repeated on every page.
    - Table has 2 columns: Position (integer) and Fehlertext (2 lines).
    - Service from customer assignments.
    - Pagination happens automatically if the table overflows.
    """
    if seed is not None:
        random.seed(seed)

    with sqlite3.connect(db_path) as conn:
        _ensure_signature_table(conn)
        company = _fetch_company(conn)
        customers = _fetch_customers(conn, nationality="DE")
        services = _ensure_services(conn)
        _assign_services(conn, customers, services, seed=seed)
        _ensure_signature_statuses(conn, customers, period_year, period_month, seed=seed)

        positions = list(_iter_unsigned_positions(conn, period_year, period_month))

    rechnung_error = (
        "Für einige Leistungen liegen keine Unterschriften vor. Bitte nutzen Sie die manuelle Druckfunktion, "
        "um die Leistungsnachweise zu drucken."
    )

    styles = getSampleStyleSheet()

    # Approved sizes
    h_main_size = 15
    h_sub_size = 11
    h_meta_size = 9.5

    header_cell_style = ParagraphStyle("header_cell", parent=styles["Normal"], fontSize=8.5, leading=10)
    cell_style = ParagraphStyle("cell", parent=styles["Normal"], fontSize=7.2, leading=8.6)

    doc = BaseDocTemplate(
        output_path,
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

        # Left header
        canvas.setFont("Helvetica-Bold", h_main_size)
        canvas.drawString(x_left, y_top, "Fehlerprotokoll")

        canvas.setFont("Helvetica", h_sub_size)
        canvas.drawString(x_left, y_top - 6.5 * mm, module)

        canvas.setFont("Helvetica", h_meta_size)
        canvas.drawString(x_left, y_top - 12 * mm, f"Datum: {datum_str}")
        canvas.drawString(x_left, y_top - 16.5 * mm, f"Seite: {canvas.getPageNumber()}")

        # Right header
        canvas.setFont("Helvetica", h_sub_size)
        canvas.drawString(x_right, y_top, company["company"])

        canvas.setFont("Helvetica", h_meta_size)
        canvas.drawString(x_right, y_top - 6.5 * mm, company["street"])
        canvas.drawString(x_right, y_top - 12 * mm, company["city"])
        canvas.drawString(x_right, y_top - 16.5 * mm, f"IK: {company['ik']}")

        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="pt", frames=[frame], onPage=draw_header)])

    table_data = [
        [
            Paragraph("<b>Position</b>", header_cell_style),
            Paragraph("<b>Fehlertext</b>", header_cell_style),
        ]
    ]

    pos_idx = 1
    for cust, svc, py, pm in positions:
        start_date = date(py, pm, 1)
        end_date = date(py, pm, calendar.monthrange(py, pm)[1])
        fmt = lambda d: d.strftime("%d.%m.%Y")
        service = f"§ {svc.code} {svc.legal_ref}"
        first_line = f"Rechnung für {cust.last_name}, {cust.first_name} ({fmt(start_date)} – {fmt(end_date)} {service})"
        fehlertext = f"{first_line}<br/><nobr>{rechnung_error}</nobr>"
        table_data.append([Paragraph(str(pos_idx), cell_style), Paragraph(fehlertext, cell_style)])
        pos_idx += 1

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Fehlerprotokoll Rechnung PDFs")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--period-year", type=int, default=2025)
    parser.add_argument("--period-month", type=int, default=12)
    parser.add_argument("--datum", default="21.01.2026 15:13")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--db", type=Path, default=Path(__file__).parents[2] / "company_universe" / "db" / "company.sqlite")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"Fehlerprotokoll_Rechnung_{args.period_year}_{args.period_month:02d}.pdf"
    out_path = out_dir / filename

    generate_fehlerprotokoll_rechnung_pdf(
        str(out_path),
        period_year=args.period_year,
        period_month=args.period_month,
        datum_str=args.datum,
        seed=args.seed,
        db_path=args.db,
    )
    print(f"Wrote {out_path}")
