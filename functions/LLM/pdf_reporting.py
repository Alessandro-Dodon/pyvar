# ────────── PDF Report Generator ──────────
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
import os
import datetime
import pandas as pd


def save_report_as_pdf(metrics: dict,
                       weights: pd.Series,
                       interpretation: str,
                       opt_list: list,
                       filename: str = "interpretation_report.pdf"):
    """
    Generates a structured PDF report with:
    - VaR metrics table
    - ES (Expected Shortfall) metrics table
    - Option positions table (if provided)
    - LLM-generated interpretation section
    """
    # Document setup
    doc = SimpleDocTemplate(
        filename, pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm
    )

    # Define styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("RptTitle",
                              fontName="Times-Roman",
                              fontSize=24,
                              alignment=TA_CENTER,
                              spaceAfter=12))
    styles.add(ParagraphStyle("SectHead",
                              fontName="Times-Roman",
                              fontSize=18,
                              spaceAfter=6))
    styles.add(ParagraphStyle("BodyTxt",
                              fontName="Times-Roman",
                              fontSize=12,
                              leading=14,
                              spaceAfter=4))

    story = []

    # — Cover page title
    story.append(Paragraph("Interpretation Report", styles["RptTitle"]))
    story.append(Paragraph(f"Date: {datetime.date.today():%d %B %Y}", styles["BodyTxt"]))
    story.append(Spacer(1, 0.7 * cm))

    # — VaR table
    story.append(Paragraph("VaR Metrics (95%)", styles["SectHead"]))
    story.append(Spacer(1, 0.5 * cm))
    data_var = [["Metric", "Value"]]
    for k, v in metrics.items():
        if "VaR" in k and "ES" not in k:
            data_var.append([k, f"{v:,.2f}"])
    tbl_var = Table(data_var, colWidths=[8 * cm, 6 * cm])
    tbl_var.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
    ]))
    story.append(tbl_var)
    story.append(Spacer(1, 0.7 * cm))

    # — ES table
    story.append(Paragraph("ES Metrics (95%)", styles["SectHead"]))
    story.append(Spacer(1, 0.5 * cm))
    data_es = [["Metric", "Value"]]
    for k, v in metrics.items():
        if "ES" in k:
            data_es.append([k, f"{v:,.2f}"])
    tbl_es = Table(data_es, colWidths=[8 * cm, 6 * cm])
    tbl_es.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
    ]))
    story.append(tbl_es)
    story.append(Spacer(1, 0.7 * cm))

    # — Option positions (if available)
    if opt_list:
        story.append(Paragraph("Option Positions", styles["SectHead"]))
        story.append(Spacer(1, 0.5 * cm))
        data_opts = [["Type", "Underlying", "Quantity", "Strike", "Maturity (y)"]]
        for op in opt_list:
            data_opts.append([
                op['type'].upper(),
                op['under'],
                f"{op['qty']:.0f}",
                f"{op['K']:.2f}",
                f"{op['T']:.2f}"
            ])
        tbl_opts = Table(data_opts, colWidths=[3 * cm, 3 * cm, 3 * cm, 4 * cm, 3 * cm])
        tbl_opts.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
        ]))
        story.append(tbl_opts)
        story.append(Spacer(1, 0.7 * cm))

    # — LLM Interpretation section
    story.append(Paragraph("LLM Interpretation", styles["SectHead"]))
    for para in interpretation.split("\n\n"):
        story.append(Paragraph(para.replace("\n", " "), styles["BodyTxt"]))
        story.append(Spacer(1, 0.3 * cm))

    # — Footer with page number
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Times-Roman", 8)
        canvas.setFillColor("grey")
        canvas.drawCentredString(A4[0] / 2, 1 * cm,
                                 f"Page {doc.page} — Confidential")
        canvas.restoreState()

    # Build PDF
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"✅ PDF report generated: {os.path.abspath(filename)}")

