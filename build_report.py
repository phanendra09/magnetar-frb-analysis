"""
build_report.py - Generate a paper-style PDF report from analysis outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def build_report(results_dir: str = "results_real", output_path: str = "results/report.pdf") -> Path:
    """Build a PDF summary report from generated results."""
    base = Path(results_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fit_path = base / "fit_results_real.json"
    if not fit_path.exists():
        fit_path = base / "fit_results.json"
    catalogue_path = base / "catalogue_real.csv"
    if not catalogue_path.exists():
        catalogue_path = base / "catalogue.csv"

    with fit_path.open("r", encoding="utf-8") as handle:
        fit_results = json.load(handle)
    catalogue = pd.read_csv(catalogue_path)

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodySmall",
            parent=styles["BodyText"],
            fontSize=10,
            leading=14,
            spaceAfter=6,
        )
    )

    story = []
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
    )

    story.append(Paragraph("Statistical Anomaly of the FRB 200428 X-ray Burst", styles["Title"]))
    story.append(Paragraph("A pipeline-generated summary report for SGR 1935+2154", styles["Heading2"]))
    story.append(Spacer(1, 0.3 * cm))

    energy = fit_results.get("energy_distribution", {})
    waiting = fit_results.get("waiting_time_analysis", {})
    frb = fit_results.get("frb_anomaly_test", {})
    soc = fit_results.get("soc_consistency", {})

    story.append(
        Paragraph(
            "This report is generated directly from the repository outputs and "
            "summarises the April 2020 SGR 1935+2154 analysis. The central question "
            "is whether the FRB-producing burst was statistically anomalous relative "
            "to the surrounding X-ray burst population.",
            styles["BodySmall"],
        )
    )

    summary_rows = [
        ["Quantity", "Value"],
        ["Catalogue size", str(len(catalogue))],
        ["NICER bursts excluding FRB", str(int((~catalogue["is_frb_burst"]).sum()))],
        ["Energy index alpha", f"{energy.get('alpha', float('nan')):.3f} +/- {energy.get('alpha_err', float('nan')):.3f}"],
        ["Waiting-time best model", waiting.get("best_model", "n/a")],
        ["Waiting-time CV", f"{waiting.get('coefficient_of_variation', float('nan')):.3f}"],
        ["FRB energy percentile", f"{frb.get('energy_percentile', float('nan')):.1f}%"],
        ["FRB energy z-score", f"{frb.get('energy_zscore', float('nan')):.2f}"],
        ["SOC consistency", "consistent" if soc.get("is_consistent") else "inconsistent"],
    ]
    table = Table(summary_rows, colWidths=[6.5 * cm, 8.5 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDEAF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#A0A0A0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F8F8")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.4 * cm))

    if frb:
        story.append(Paragraph("Headline result", styles["Heading2"]))
        story.append(Paragraph(frb.get("conclusion", "No FRB anomaly conclusion available."), styles["BodySmall"]))

    if soc:
        story.append(Paragraph("SOC interpretation", styles["Heading2"]))
        story.append(Paragraph(soc.get("conclusion", "No SOC conclusion available."), styles["BodySmall"]))

    figures_dir = base / "figures"
    figure_names = [
        ("Figure 1. Light curve and Bayesian Blocks segmentation", "fig1_lightcurve.png"),
        ("Figure 2. Burst energy distribution", "fig2_energy_distribution.png"),
        ("Figure 3. Waiting-time distribution", "fig3_waiting_times.png"),
        ("Figure 4. FRB anomaly assessment", "fig4_frb_anomaly.png"),
        ("Figure 5. SOC consistency check", "fig5_soc_check.png"),
    ]

    for caption, filename in figure_names:
        image_path = figures_dir / filename
        if not image_path.exists():
            continue
        story.append(PageBreak())
        story.append(Paragraph(caption, styles["Heading2"]))
        story.append(Spacer(1, 0.2 * cm))
        story.append(Image(str(image_path), width=17.2 * cm, height=11.2 * cm))
        story.append(Spacer(1, 0.2 * cm))
        story.append(
            Paragraph(
                "Generated directly from the repository plotting pipeline.",
                styles["BodySmall"],
            )
        )

    doc.build(story)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a PDF report from pipeline outputs")
    parser.add_argument("--results-dir", default="results_real", help="Directory containing JSON, CSV, and figures")
    parser.add_argument("--output", default="results/report.pdf", help="Output PDF path")
    args = parser.parse_args()

    output = build_report(results_dir=args.results_dir, output_path=args.output)
    print(output)


if __name__ == "__main__":
    main()
