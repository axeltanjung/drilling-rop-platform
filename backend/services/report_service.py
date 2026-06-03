"""Drilling report exporters: PDF (reportlab) and CSV."""
from __future__ import annotations

import io
from datetime import datetime

import pandas as pd

from backend.utils.logger import get_logger

log = get_logger("report_service")


def optimization_to_pdf(result: dict, well_id: str | None = None) -> bytes:
    """Render an AI drilling optimization recommendation as a PDF."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="Drilling Optimization Report")
    styles = getSampleStyleSheet()
    h = ParagraphStyle("h", parent=styles["Title"], textColor=colors.HexColor("#0ea5e9"))
    story = []

    story.append(Paragraph("Drilling ROP Optimization Report", h))
    story.append(Paragraph(
        f"Generated: {datetime.utcnow():%Y-%m-%d %H:%M UTC}"
        + (f" &nbsp;|&nbsp; Well: {well_id}" if well_id else ""),
        styles["Normal"]))
    story.append(Spacer(1, 8 * mm))

    perf = [
        ["Metric", "Baseline", "Optimized", "Improvement"],
        ["ROP (ft/hr)", f"{result['baseline_rop']:.1f}",
         f"{result['optimized_rop']:.1f}", f"{result['improvement_pct']:+.1f}%"],
    ]
    t = Table(perf, colWidths=[55*mm, 40*mm, 40*mm, 40*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(t)
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("Recommended Parameters", styles["Heading2"]))
    rec = result["recommended_parameters"]
    base = result["baseline_parameters"]
    pr = [["Parameter", "Current", "Recommended"]]
    for k in rec:
        pr.append([k.replace("_", " ").title(), f"{base.get(k, 0):.0f}", f"{rec[k]:.0f}"])
    pt = Table(pr, colWidths=[70*mm, 50*mm, 50*mm])
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(pt)
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("Operational Trade-offs", styles["Heading2"]))
    for msg in result.get("trade_offs", []):
        story.append(Paragraph(f"• {msg}", styles["Normal"]))

    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph(
        "<i>Synthetic, physics-inspired demonstration. Not for operational use.</i>",
        styles["Italic"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
