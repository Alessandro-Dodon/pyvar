"""
PDF Reporting Module
--------------------

Contains functions to generate a structured PDF report of:
 - Portfolio Weights
 - Value at Risk (VaR) Metrics
 - Expected Shortfall (ES) Metrics
 - Option Positions (if provided)
 - Backtest Results (Violations, Violation Rate %, Joint p-value)
 - LLM Interpretation Text

Usage
-----
    Call `generate_pdf_report(...)` with the appropriate arguments:
      - risk_metrics: dict of VaR/ES metrics
      - portfolio_weights: pandas.Series of weights (indexed by ticker, values [0,1])
      - interpretation_text: str with LLM paragraphs separated by double newlines
      - option_positions: list of dicts for option positions (or None)
      - backtest_results_dataframe: pandas.DataFrame with columns ["Violations", "Violation Rate", "Joint p-value"]
      - base_currency: str (e.g. "EUR") to append to VaR/ES values
      - output_path: str path where to save the PDF (optional; if omitted, a temp file is created)

Authors
-------
Niccolò Lecce, Alessandro Dodon, Marco Gasparetti

Created
-------
May 2025

Contents
--------
- ProfessionalReport: Custom document template with header and footer.
- build_portfolio_weights_table: Build a table for portfolio weights.
- build_value_at_risk_table: Build a table for Value at Risk (VaR) metrics.
- build_expected_shortfall_table: Build a table for Expected Shortfall (ES) metrics.
- build_option_positions_table: Build a table for option positions.
- build_backtest_results_table: Build a table for backtest results.
- build_interpretation_paragraphs: Generate interpretation paragraphs for the report.
- generate_pdf_report: Main function to generate the PDF report.
"""


#----------------------------------------------------------
# Packages
#----------------------------------------------------------
import os
import sys
import datetime
import tempfile
import subprocess
from io import BytesIO
from typing import Dict, Any, List, Optional
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    PageTemplate,
    Frame,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)


# -----------------------
# Branding Constants
# -----------------------
FONT_FAMILY_REGULAR = "Helvetica"
FONT_FAMILY_BOLD = "Helvetica-Bold"
FONT_FAMILY_ITALIC = "Helvetica-Oblique"

COLOR_SECTION_HEADER = colors.HexColor("#000F66")
COLOR_TABLE_HEADER_BACKGROUND = colors.HexColor("#EFEFEF")
COLOR_TABLE_GRIDLINES = colors.HexColor("#CCCCCC")
COLOR_HEADER_FOOTER_TEXT = colors.grey


# -----------------------
# Custom Template
# -----------------------
class ProfessionalReport(BaseDocTemplate):
    """
    Custom document template that draws a header and footer on each page.
    """
    def __init__(
        self,
        filename: str,
        pagesize=A4,
        left_margin: float = 2 * cm,
        right_margin: float = 2 * cm,
        top_margin: float = 2 * cm,
        bottom_margin: float = 2 * cm,
    ):
        super().__init__(
            filename=filename,
            pagesize=pagesize,
            leftMargin=left_margin,
            rightMargin=right_margin,
            topMargin=top_margin,
            bottomMargin=bottom_margin,
        )
        # Content frame (leave 1cm at top for header)
        main_frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height - 1 * cm,
            id="main_frame",
        )
        page_template = PageTemplate(
            id="ProfessionalTemplate",
            frames=[main_frame],
            onPage=self._draw_header_and_footer,
        )
        self.addPageTemplates([page_template])

    def _draw_header_and_footer(self, canvas, document):
        # Header
        canvas.saveState()
        canvas.setFont(FONT_FAMILY_REGULAR, 9)
        canvas.setFillColor(COLOR_HEADER_FOOTER_TEXT)
        header_text = "PyVar | Portfolio Risk Analysis Report"
        canvas.drawString(self.leftMargin, A4[1] - 1 * cm, header_text)
        canvas.setLineWidth(0.5)
        canvas.setStrokeColor(COLOR_HEADER_FOOTER_TEXT)
        canvas.line(
            self.leftMargin, A4[1] - 1.1 * cm, A4[0] - self.rightMargin, A4[1] - 1.1 * cm
        )

        # Footer
        footer_text = f"Page {document.page}"
        canvas.setFont(FONT_FAMILY_ITALIC, 8)
        canvas.setFillColor(COLOR_HEADER_FOOTER_TEXT)
        canvas.drawRightString(A4[0] - self.rightMargin, 1 * cm, footer_text)
        canvas.restoreState()


# ----------------------------------------
# Helper to Build Portfolio Weights Table
# ----------------------------------------
def build_portfolio_weights_table(
    portfolio_weights: pd.Series, styles: Dict[str, ParagraphStyle]
) -> Table:
    """
    Build a table for portfolio weights where every cell is centered.

    Parameters
    ----------
    portfolio_weights : pandas.Series
        Series of portfolio weights, indexed by ticker symbols (values between 0 and 1).
    styles : Dict[str, ParagraphStyle]
        Dictionary of paragraph styles.

    Returns
    -------
    Table
        A ReportLab Table flowable with columns: Ticker, Weight (%), all centered.
    """
    if portfolio_weights.empty:
        raise ValueError("The portfolio_weights series is empty.")

    table_data = [["Ticker", "Weight (%)"]]
    for ticker_symbol, weight_value in portfolio_weights.items():
        percentage_value = weight_value * 100
        table_data.append(
            [
                Paragraph(str(ticker_symbol), styles["BodyText"]),
                Paragraph(f"{percentage_value:.2f}%", styles["BodyText"]),
            ]
        )

    column_widths = [6 * cm, 10 * cm]
    weights_table = Table(table_data, colWidths=column_widths, hAlign="CENTER")
    weights_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_TABLE_HEADER_BACKGROUND),
                ("FONTNAME", (0, 0), (-1, 0), FONT_FAMILY_BOLD),
                # Center everything (header + body)
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LINEBELOW", (0, 0), (-1, 0), 1, colors.darkgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, COLOR_TABLE_GRIDLINES),
            ]
        )
    )
    return weights_table


# --------------------------
# Helper to Build VaR Table
# --------------------------
def build_value_at_risk_table(
    risk_metrics: Dict[str, float], styles: Dict[str, ParagraphStyle], base_currency: str
) -> Table:
    """
    Build a table for Value at Risk (VaR) metrics where every cell is centered,
    appending base_currency after each numeric value.

    Parameters
    ----------
    risk_metrics : Dict[str, float]
        Dictionary containing keys with "VaR" (excluding "ES") mapped to numeric values.
    styles : Dict[str, ParagraphStyle]
        Dictionary of paragraph styles.
    base_currency : str
        Currency code to append (e.g. "EUR"), or empty string if none.

    Returns
    -------
    Table
        A ReportLab Table flowable with columns: [Metric, Value <currency>], all centered.
    """
    var_metrics = {
        metric_name: metric_value
        for metric_name, metric_value in risk_metrics.items()
        if "VaR" in metric_name and "ES" not in metric_name
    }
    if not var_metrics:
        raise ValueError("The risk_metrics dictionary does not contain any VaR entries.")

    table_data = [["Metric", "Value"]]
    for metric_name, metric_value in var_metrics.items():
        formatted_value = f"{metric_value:,.2f}"
        if base_currency:
            formatted_value += f" {base_currency}"
        table_data.append(
            [
                Paragraph(metric_name, styles["BodyText"]),
                Paragraph(formatted_value, styles["BodyText"]),
            ]
        )

    column_widths = [9 * cm, 7 * cm]
    var_table = Table(table_data, colWidths=column_widths, hAlign="CENTER")
    var_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_TABLE_HEADER_BACKGROUND),
                ("FONTNAME", (0, 0), (-1, 0), FONT_FAMILY_BOLD),
                # Center all cells (header + body)
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LINEBELOW", (0, 0), (-1, 0), 1, colors.darkgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, COLOR_TABLE_GRIDLINES),
            ]
        )
    )
    return var_table


# --------------------------
# Helper to Build ES Table
# --------------------------
def build_expected_shortfall_table(
    risk_metrics: Dict[str, float], styles: Dict[str, ParagraphStyle], base_currency: str
) -> Table:
    """
    Build a table for Expected Shortfall (ES) metrics where every cell is centered,
    appending base_currency after each numeric value.

    Parameters
    ----------
    risk_metrics : Dict[str, float]
        Dictionary containing keys with "ES" mapped to numeric values.
    styles : Dict[str, ParagraphStyle]
        Dictionary of paragraph styles.
    base_currency : str
        Currency code to append (e.g. "EUR"), or empty string if none.

    Returns
    -------
    Table
        A ReportLab Table flowable with columns: [Metric, Value <currency>], all centered.
    """
    es_metrics = {
        metric_name: metric_value
        for metric_name, metric_value in risk_metrics.items()
        if "ES" in metric_name
    }
    if not es_metrics:
        raise ValueError("The risk_metrics dictionary does not contain any ES entries.")

    table_data = [["Metric", "Value"]]
    for metric_name, metric_value in es_metrics.items():
        formatted_value = f"{metric_value:,.2f}"
        if base_currency:
            formatted_value += f" {base_currency}"
        table_data.append(
            [
                Paragraph(metric_name, styles["BodyText"]),
                Paragraph(formatted_value, styles["BodyText"]),
            ]
        )

    column_widths = [9 * cm, 7 * cm]
    es_table = Table(table_data, colWidths=column_widths, hAlign="CENTER")
    es_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_TABLE_HEADER_BACKGROUND),
                ("FONTNAME", (0, 0), (-1, 0), FONT_FAMILY_BOLD),
                # Center all cells (header + body)
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LINEBELOW", (0, 0), (-1, 0), 1, colors.darkgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, COLOR_TABLE_GRIDLINES),
            ]
        )
    )
    return es_table


# ---------------------------------------
# Helper to Build Option Positions Table
# ---------------------------------------
def build_option_positions_table(
    option_positions: List[Dict[str, Any]],
    styles: Dict[str, ParagraphStyle]
) -> Table:
    """
    Build a table for option positions, centering every cell.

    Each entry in option_positions must be a dict with keys:
      - "under"       (underlying ticker)
      - "type"        ("call" or "put")
      - "contracts"   (float or int, number of contracts)
      - "multiplier"  (int, usually 100)
      - "qty"         (float or int, total quantity = contracts * multiplier)
      - "K"           (float, strike price)
      - "T"           (float, time to maturity in years)

    Parameters
    ----------
    option_positions : List[Dict[str, Any]]
        List of option dicts with exactly the keys above.
    styles : Dict[str, ParagraphStyle]
        Dictionary of paragraph styles (must include "BodyText").

    Returns
    -------
    Table
        A ReportLab Table flowable with columns:
        ["Type", "Underlying", "Quantity", "Strike Price", "Maturity (y)"],
        and all cells centered.
    """
    if not option_positions:
        raise ValueError("The option_positions list is empty or None.")

    # Header row
    header_row = ["Type", "Underlying", "Quantity", "Strike Price", "Maturity (y)"]

    # Build table_data, starting with the header
    table_data = [[Paragraph(col, styles["BodyText"]) for col in header_row]]

    # Populate each option row
    for pos in option_positions:
        # Read fields from your dict:
        opt_type = pos["type"]                # e.g. "call" or "put"
        underlying = pos["under"]             # e.g. "AAPL"
        quantity = pos["qty"]                 # total number of shares (contracts*multiplier)
        strike_price = pos["K"]               # strike price
        maturity_years = pos["T"]             # time to maturity in years

        # Format each cell as a Paragraph
        cell_type = Paragraph(opt_type.upper(), styles["BodyText"])
        cell_underlying = Paragraph(str(underlying), styles["BodyText"])
        cell_quantity = Paragraph(f"{int(quantity):,}", styles["BodyText"])
        cell_strike = Paragraph(f"{strike_price:.2f}", styles["BodyText"])
        cell_maturity = Paragraph(f"{maturity_years:.2f}", styles["BodyText"])

        table_data.append([
            cell_type,
            cell_underlying,
            cell_quantity,
            cell_strike,
            cell_maturity
        ])

    # Choose column widths (adjust as needed)
    column_widths = [3 * cm, 3 * cm, 3 * cm, 3 * cm, 4 * cm]

    option_table = Table(table_data, colWidths=column_widths, hAlign="CENTER")
    option_table.setStyle(
        TableStyle(
            [
                # Header background and bold font
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_TABLE_HEADER_BACKGROUND),
                ("FONTNAME", (0, 0), (-1, 0), FONT_FAMILY_BOLD),

                # Center-align everything (header + body)
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

                # Single line below header
                ("LINEBELOW", (0, 0), (-1, 0), 1, colors.darkgrey),

                # Light grid for all cells
                ("GRID", (0, 0), (-1, -1), 0.5, COLOR_TABLE_GRIDLINES),
            ]
        )
    )
    return option_table


# ---------------------------------------
# Helper to Build Backtest Results Table
# ---------------------------------------
def build_backtest_results_table(
    backtest_results_dataframe: pd.DataFrame, styles: Dict[str, ParagraphStyle]
) -> Table:
    """
    Build a table of backtest results containing only:
      - Violations
      - Violation Rate (formatted as percent)
      - Joint p-value

    Every cell is centered.

    Parameters
    ----------
    backtest_results_dataframe : pandas.DataFrame
        DataFrame with at least these columns:
        ["Violations", "Violation Rate", "Joint p-value", "Decision" (optional)].
        The index must be model names.
    styles : Dict[str, ParagraphStyle]
        Dictionary of paragraph styles.

    Returns
    -------
    Table
        A ReportLab Table flowable showing:
        Model | Violations | Violation Rate (%) | Joint p-value, all centered.
    """
    if backtest_results_dataframe.empty:
        raise ValueError("The backtest_results_dataframe is empty.")

    required_columns = ["Violations", "Violation Rate", "Joint p-value"]
    for col_name in required_columns:
        if col_name not in backtest_results_dataframe.columns:
            raise ValueError(f"The DataFrame is missing column '{col_name}'.")

    header_style = ParagraphStyle(
        name="HeaderCenter",
        parent=styles["BodyText"],
        alignment=TA_CENTER,
        leading=12,
    )
    headers = [
        Paragraph("Model", header_style),
        Paragraph("Violations", header_style),
        Paragraph("Violation Rate", header_style),
        Paragraph("Joint p-value", header_style),
    ]

    table_data = [headers]
    for model_name, row in backtest_results_dataframe.iterrows():
        violation_count = int(row["Violations"])
        violation_rate_value = row["Violation Rate"]
        violation_rate_pct = f"{violation_rate_value * 100:.1f}%"
        joint_p_value = row["Joint p-value"]

        table_data.append(
            [
                Paragraph(str(model_name), styles["BodyText"]),
                Paragraph(str(violation_count), styles["BodyText"]),
                Paragraph(violation_rate_pct, styles["BodyText"]),
                Paragraph(f"{joint_p_value:.3f}", styles["BodyText"]),
            ]
        )

    total_page_width = A4[0] - 4 * cm
    column_widths = [
        total_page_width * 0.35,  # Model
        total_page_width * 0.15,  # Violations
        total_page_width * 0.25,  # Violation Rate
        total_page_width * 0.25,  # Joint p-value
    ]

    backtest_table = Table(table_data, colWidths=column_widths, hAlign="CENTER")
    backtest_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_TABLE_HEADER_BACKGROUND),
                ("FONTNAME", (0, 0), (-1, 0), FONT_FAMILY_BOLD),
                # Center all cells (header + body)
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LINEBELOW", (0, 0), (-1, 0), 1, colors.darkgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, COLOR_TABLE_GRIDLINES),
            ]
        )
    )
    return backtest_table


# ------------------------------------------
# Helper to Build Interpretation Paragraphs
# ------------------------------------------
def build_interpretation_paragraphs(
    interpretation_text: str, styles: Dict[str, ParagraphStyle]
) -> List:
    """
    Split the interpretation text into paragraphs and return a list of flowables.

    Parameters
    ----------
    interpretation_text : str
        Plain-text analysis produced by the LLM, paragraphs separated by double newlines.
    styles : Dict[str, ParagraphStyle]
        Dictionary of paragraph styles.

    Returns
    -------
    List
        List of Paragraph and Spacer flowables.
    """
    flowables: List = []
    paragraphs = [block.strip() for block in interpretation_text.split("\n\n") if block.strip()]
    paragraph_style = ParagraphStyle(
        name="InterpretationBody",
        parent=styles["BodyText"],
        leftIndent=1 * cm,
        leading=14,
    )
    for block in paragraphs:
        flowables.append(Paragraph(block.replace("\n", " "), paragraph_style))
        flowables.append(Spacer(1, 0.4 * cm))
    return flowables


# ------------------------------------
# Main Function to Generate the PDF
# ------------------------------------
def generate_pdf_report(
    risk_metrics: Dict[str, float],
    portfolio_weights: pd.Series,
    interpretation_text: str,
    option_positions: Optional[List[Dict[str, Any]]],
    backtest_results_dataframe: pd.DataFrame,
    base_currency: str,
    output_path: Optional[str] = None,
) -> None:
    """
    Generate a professional PDF report with sections:
      - Portfolio Weights
      - Value at Risk (VaR) Metrics (with currency)
      - Expected Shortfall (ES) Metrics (with currency)
      - Option Positions (if provided)
      - Backtest Results (Violations, Violation Rate %, Joint p-value)
      - LLM Interpretation Text

    Parameters
    ----------
    risk_metrics : Dict[str, float]
        Dictionary mapping metric names to numeric values.
        Example: {"Asset-Normal VaR": 62.02, "Sharpe-Factor ES": 70.21, ...}
    portfolio_weights : pandas.Series
        Series of portfolio weights, indexed by ticker symbols (values between 0 and 1).
    interpretation_text : str
        Plain-text analysis produced by the LLM, paragraphs separated by double newlines.
    option_positions : Optional[List[Dict[str, Any]]]
        List of dictionaries for option positions (each containing keys:
        "under", "type", "contracts", "multiplier", "qty", "K", "T").
    backtest_results_dataframe : pandas.DataFrame
        DataFrame containing at least columns:
        ["Violations", "Violation Rate", "Joint p-value", "Decision" (optional)].
        The index must be model names.
    base_currency : str
        Currency code to append after each VaR/ES value (e.g. "EUR"). Can be "" for none.
    output_path : Optional[str]
        If provided, save the PDF at this path. Otherwise, create a temporary file.

    Returns
    -------
    None
        Creates (or overwrites) the PDF and opens it with the default system PDF viewer.
    """
    # Input validation
    if not isinstance(risk_metrics, dict) or not risk_metrics:
        raise ValueError("The risk_metrics argument must be a non-empty dictionary.")
    if not isinstance(portfolio_weights, pd.Series):
        raise TypeError("The portfolio_weights argument must be a pandas.Series.")
    if not isinstance(interpretation_text, str) or not interpretation_text.strip():
        raise ValueError("The interpretation_text must be a non-empty string.")
    if not isinstance(backtest_results_dataframe, pd.DataFrame):
        raise TypeError("The backtest_results_dataframe must be a pandas.DataFrame.")

    # Determine where to save the PDF
    if output_path:
        temporary_pdf_path = output_path
        os.makedirs(os.path.dirname(temporary_pdf_path) or ".", exist_ok=True)
    else:
        file_descriptor, temporary_pdf_path = tempfile.mkstemp(suffix=".pdf")
        os.close(file_descriptor)

    # Set up the document with custom header/footer
    document = ProfessionalReport(
        filename=temporary_pdf_path,
        pagesize=A4,
        left_margin=2 * cm,
        right_margin=2 * cm,
        top_margin=2 * cm,
        bottom_margin=2 * cm,
    )

    # Define paragraph styles
    shared_styles = getSampleStyleSheet()
    paragraph_styles = {
        "Title": ParagraphStyle(
            name="Title",
            fontName=FONT_FAMILY_BOLD,
            fontSize=28,
            alignment=TA_CENTER,
            spaceAfter=12,
            textColor=COLOR_SECTION_HEADER,
        ),
        "Date": ParagraphStyle(
            name="Date",
            fontName=FONT_FAMILY_REGULAR,
            fontSize=10,
            alignment=TA_CENTER,
            spaceAfter=24,
        ),
        "SectionHeader": ParagraphStyle(
            name="SectionHeader",
            fontName=FONT_FAMILY_BOLD,
            fontSize=16,
            spaceAfter=6,
            textColor=COLOR_SECTION_HEADER,
        ),
        "BodyText": ParagraphStyle(
            name="BodyText",
            fontName=FONT_FAMILY_REGULAR,
            fontSize=11,
            leading=13,
            spaceAfter=4,
            alignment=TA_LEFT,
        ),
    }

    # Build the flowables
    report_flowables: List = []

    # --------------------------------------------------
    # Cover Page: Title + Date (separated)
    # --------------------------------------------------
    report_flowables.append(Paragraph("Portfolio Risk Report", paragraph_styles["Title"]))
    report_flowables.append(Spacer(1, 0.5 * cm))
    report_flowables.append(
        Paragraph(f"Date: {datetime.date.today():%d %B %Y}", paragraph_styles["Date"])
    )
    report_flowables.append(Spacer(1, 1 * cm))

    # --------------------------------------------------
    # Section: Portfolio Weights
    # --------------------------------------------------
    report_flowables.append(
        Paragraph("Portfolio Weights", paragraph_styles["SectionHeader"])
    )
    report_flowables.append(Spacer(1, 0.5 * cm))
    report_flowables.append(
        build_portfolio_weights_table(portfolio_weights, paragraph_styles)
    )
    report_flowables.append(Spacer(1, 1 * cm))

    # --------------------------------------------------
    # Section: Value at Risk (VaR) Metrics
    # --------------------------------------------------
    report_flowables.append(
        Paragraph("Value at Risk (VaR) Metrics", paragraph_styles["SectionHeader"])
    )
    report_flowables.append(Spacer(1, 0.5 * cm))
    report_flowables.append(
        build_value_at_risk_table(risk_metrics, paragraph_styles, base_currency)
    )
    report_flowables.append(Spacer(1, 1 * cm))

    # --------------------------------------------------
    # Section: Expected Shortfall (ES) Metrics
    # --------------------------------------------------
    report_flowables.append(
        Paragraph("Expected Shortfall (ES) Metrics", paragraph_styles["SectionHeader"])
    )
    report_flowables.append(Spacer(1, 0.5 * cm))
    report_flowables.append(
        build_expected_shortfall_table(risk_metrics, paragraph_styles, base_currency)
    )
    report_flowables.append(Spacer(1, 1 * cm))

    # --------------------------------------------------
    # Section: Option Positions (if provided)
    # --------------------------------------------------
    if option_positions:
        report_flowables.append(
            Paragraph("Option Positions", paragraph_styles["SectionHeader"])
        )
        report_flowables.append(Spacer(1, 0.5 * cm))
        report_flowables.append(
            build_option_positions_table(option_positions, paragraph_styles)
        )
        report_flowables.append(Spacer(1, 1.5 * cm))

    # --------------------------------------------------
    # Section: Backtest Results (Violations, Violation Rate %, Joint p-value)
    # --------------------------------------------------
    report_flowables.append(
        Paragraph("Backtest Results", paragraph_styles["SectionHeader"])
    )
    report_flowables.append(Spacer(1, 0.5 * cm))
    report_flowables.append(
        build_backtest_results_table(backtest_results_dataframe, paragraph_styles)
    )

    #page break for llm interpretation
    report_flowables.append(PageBreak())

    # --------------------------------------------------
    # Section: LLM Interpretation
    # --------------------------------------------------
    report_flowables.append(
        Paragraph("LLM Interpretation", paragraph_styles["SectionHeader"])
    )
    report_flowables.append(Spacer(1, 0.5 * cm))
    report_flowables.extend(
        build_interpretation_paragraphs(interpretation_text, paragraph_styles)
    )

    # Build the PDF and open it
    document.build(report_flowables)

    # Open with system default PDF viewer
    if sys.platform.startswith("win"):
        os.startfile(temporary_pdf_path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", temporary_pdf_path])
    else:
        subprocess.Popen(["xdg-open", temporary_pdf_path])
