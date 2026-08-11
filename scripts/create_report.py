"""Build the submission-ready Word report from recorded training results."""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
METRICS = ROOT / "artifacts" / "training_metrics.json"
CHART = ROOT / "outputs" / "scorer_results" / "candidate_december.png"
OUT = REPORTS / "freight_rate_assessment_report.docx"

BLUE = RGBColor(46, 116, 181)
DARK_BLUE = RGBColor(31, 77, 120)
INK = RGBColor(11, 37, 69)
MUTED = RGBColor(89, 99, 110)


def set_font(run, size=11, color=None, bold=None, italic=None):
    run.font.name = "Calibri"
    run._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = color
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def shade(cell, fill):
    props = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    props.append(shd)


def set_cell_width(cell, width_dxa):
    props = cell._tc.get_or_add_tcPr()
    width = props.find(qn("w:tcW"))
    if width is None:
        width = OxmlElement("w:tcW")
        props.append(width)
    width.set(qn("w:w"), str(width_dxa))
    width.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths):
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.first_child_found_in("w:tblW")
    tbl_w.set(qn("w:w"), str(sum(widths)))
    tbl_w.set(qn("w:type"), "dxa")
    indent = OxmlElement("w:tblInd")
    indent.set(qn("w:w"), "120")
    indent.set(qn("w:type"), "dxa")
    tbl_pr.append(indent)
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            set_cell_width(cell, width)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            margins = cell._tc.get_or_add_tcPr().first_child_found_in("w:tcMar")
            if margins is None:
                margins = OxmlElement("w:tcMar")
                cell._tc.get_or_add_tcPr().append(margins)
            for side in ("top", "bottom", "start", "end"):
                edge = OxmlElement(f"w:{side}")
                edge.set(qn("w:w"), "80" if side in ("top", "bottom") else "120")
                edge.set(qn("w:type"), "dxa")
                margins.append(edge)


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.add_run(text)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(5)
    p.add_run(text)


def add_metric_table(doc, summary):
    table = doc.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    set_table_geometry(table, [2400, 2320, 2320, 2320])
    headers = ["Model", "MAE", "RMSE", "R-squared"]
    for cell, text in zip(table.rows[0].cells, headers):
        shade(cell, "F2F4F7")
        r = cell.paragraphs[0].add_run(text)
        set_font(r, size=10, color=INK, bold=True)
    rows = [("Median-rate baseline", summary["baseline_median"]), ("HistGradientBoosting", summary["hist_gradient_boosting"])]
    for label, values in rows:
        cells = table.add_row().cells
        values_text = [label, f"${values['mae']:,.2f}", f"${values['rmse']:,.2f}", f"{values['r2']:.3f}"]
        for cell, text in zip(cells, values_text):
            r = cell.paragraphs[0].add_run(text)
            set_font(r, size=10, color=INK, bold=(label == "HistGradientBoosting"))
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(f"Chronological holdout: {summary['split']['holdout_rows']:,} loads from {summary['split']['holdout_from']} onward.")
    set_font(r, size=9, color=MUTED, italic=True)


def build():
    summary = json.loads(METRICS.read_text(encoding="utf-8"))
    q = summary["data_quality"]
    split = summary["split"]
    model = summary["hist_gradient_boosting"]
    doc = Document()
    section = doc.sections[0]
    section.top_margin = section.bottom_margin = section.left_margin = section.right_margin = Inches(1)
    section.header_distance = section.footer_distance = Inches(0.492)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.10
    for name, size, color, before, after in [("Heading 1", 16, BLUE, 16, 8), ("Heading 2", 13, BLUE, 12, 6), ("Heading 3", 12, DARK_BLUE, 8, 4)]:
        style = doc.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = color
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    set_font(header.add_run("Spotter AI Labs | Freight Rate Prediction Assessment"), size=9, color=MUTED)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    set_font(footer.add_run("Candidate technical report"), size=9, color=MUTED)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    set_font(p.add_run("MACHINE LEARNING ENGINEER ASSESSMENT"), size=10, color=BLUE, bold=True)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    set_font(p.add_run("Freight Rate Prediction"), size=24, color=INK, bold=True)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(16)
    set_font(p.add_run("Technical report | Prepared for Spotter AI Labs | Candidate"), size=11, color=MUTED)

    add_heading(doc, "Executive summary")
    p = doc.add_paragraph()
    p.add_run("Recommendation. ").bold = True
    p.add_run(f"Use the single HistGradientBoosting model for the submission. On a strictly future holdout, it achieved a ${model['mae']:,.2f} MAE and {model['r2']:.3f} R-squared, materially improving on a median-rate baseline (${summary['baseline_median']['mae']:,.2f} MAE).")
    add_bullet(doc, "Validation is chronological: earlier dates train the model, while the most recent dates simulate future pricing decisions.")
    add_bullet(doc, "The final model is refit on all 48,000 labelled loads before predicting the 12,000 unseen validation loads.")
    add_bullet(doc, "The supplied scorer validates both required CSV files and produces the fixed December forecast chart included below.")

    add_heading(doc, "1. Data exploration and quality")
    doc.add_paragraph(f"The development file contains {q['rows']:,} labelled loads from {q['date_min']} through {q['date_max']}. It includes 64 pickup cities, 64 delivery cities, and three equipment types (Dry Van, Flatbed, and Reefer). The target is posted_rate in dollars per load.")
    add_bullet(doc, f"No duplicate rows or duplicate load IDs were found ({q['duplicate_rows']} and {q['duplicate_load_ids']}, respectively).")
    add_bullet(doc, f"Weight has {q['missing_weight']:,} missing values and {q['nonpositive_weight']:,} non-positive values. Non-positive weight is treated as missing because it is physically invalid.")
    add_bullet(doc, f"market_index has {q['missing_market_index']:,} missing values. Numeric fields are median-imputed and categoricals use the most-frequent value, learned only from training data.")

    add_heading(doc, "2. Validation design")
    doc.add_paragraph(f"The hidden validation data begins after the labelled period, so random k-fold validation would let the model learn from future seasonality and market conditions. I instead sorted by date and held out the latest 20% of unique dates: training through {split['train_through']} and evaluating from {split['holdout_from']}. This creates a realistic forward-looking test with {split['development_rows']:,} development loads and {split['holdout_rows']:,} holdout loads.")
    add_metric_table(doc, summary)

    add_heading(doc, "3. Features and model choice")
    doc.add_paragraph("The feature set combines operational, geographic, market, and calendar signals: pickup and delivery, a route key, equipment, latitude/longitude, distance, weight, market index, quote signal, weekday, month, day-of-year, and cyclic seasonality features. load_id is deliberately excluded because it is an identifier rather than a causal pricing feature.")
    doc.add_paragraph("HistGradientBoosting was selected because freight pricing depends on non-linear interactions: distance behaves differently by equipment and route, while market conditions and seasonality shift rate levels. A single regularized model is strong on the holdout while remaining easier to explain, reproduce, and operate than an ensemble.")

    add_heading(doc, "4. Final prediction deliverables")
    doc.add_paragraph("After evaluation, the selected pipeline was retrained on the full labelled development set. It generated 12,000 positive final predictions in the exact required load_id,predicted_rate schema. The December scenario uses the same fitted pipeline; fields not supplied by that fixed scenario are handled through the fitted imputers, while route, equipment, distance, weight, and date are retained.")

    doc.add_page_break()
    add_heading(doc, "5. December 2025 fixed-route forecast")
    doc.add_paragraph("Inputs are fixed to Lexington to Fort Wayne, 360 miles, Dry Van, and 32,000 lb; only the date changes. The chart below was generated by the assessment-provided score.py after validating all file schemas and values.")
    if not CHART.exists():
        raise FileNotFoundError(f"Expected scorer chart not found: {CHART}")
    doc.add_picture(str(CHART), width=Inches(6.35))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(cap.add_run("Figure 1. Validated December 2025 predicted load rate."), size=9, color=MUTED, italic=True)

    add_heading(doc, "6. Reproducibility")
    doc.add_paragraph("The solution runs in Docker with pinned dependencies. Training saves the model and metrics; inference writes validation_predictions.csv and december_predictions.csv; score.py validates the files and creates the chart. The repository README contains the exact commands.")
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
