import os
import tempfile
from datetime import datetime

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, ListFlowable, ListItem, Image, Table, TableStyle
)
from reportlab.lib.enums import TA_LEFT

from .schema import SecurityReport, SEVERITY_ORDER
from .charts import render_severity_chart, render_framework_chart
from ..structured import is_valid_mermaid

SEVERITY_COLORS = {
    "Critical": colors.HexColor("#8B0000"),
    "High": colors.HexColor("#C0392B"),
    "Medium": colors.HexColor("#E67E22"),
    "Low": colors.HexColor("#218738"),
    "Informational": colors.HexColor("#5D6D7E"),
}


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle(name="CenterTitle", parent=ss["Title"], alignment=1))
    ss.add(ParagraphStyle(name="CenterSub", parent=ss["Normal"], alignment=1,
                           textColor=colors.HexColor("#5D6D7E"), fontSize=13))
    ss.add(ParagraphStyle(name="Label", parent=ss["Normal"], fontName="Helvetica-Bold"))
    ss.add(ParagraphStyle(name="Mono", parent=ss["Code"], fontName="Courier", fontSize=9))
    return ss


def _severity_color(sev: str) -> str:
    c = SEVERITY_COLORS.get(sev, colors.black)
    return f"#{c.hexval()[2:]}"


def render_pdf(report: SecurityReport, output_path: str) -> str:
    doc = SimpleDocTemplate(output_path, pagesize=LETTER,
                             topMargin=0.9 * inch, bottomMargin=0.9 * inch)
    ss = _styles()
    story = []

    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph(report.title, ss["CenterTitle"]))
    story.append(Paragraph("Cybersecurity Assessment Report", ss["CenterSub"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(datetime.now().strftime("%d %B %Y"), ss["CenterSub"]))
    story.append(PageBreak())

    story.append(Paragraph("1. Client Context", ss["Heading1"]))
    story.append(Paragraph(report.client_context, ss["BodyText"]))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("2. Scope", ss["Heading1"]))
    story.append(Paragraph(report.scope, ss["BodyText"]))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("3. Executive Summary", ss["Heading1"]))
    story.append(Paragraph(report.executive_summary, ss["BodyText"]))
    risk_color = SEVERITY_COLORS.get(report.overall_risk_rating, colors.black)
    story.append(Paragraph(
        f'<b>Overall Risk Rating: <font color="{risk_color.hexval()}">{report.overall_risk_rating}</font></b>',
        ss["BodyText"],
    ))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("4. Findings Overview", ss["Heading1"]))

    tmpdir = tempfile.mkdtemp()
    chart_paths_to_cleanup = []
    try:
        severity_path = os.path.join(tmpdir, "severity.png")
        framework_path = os.path.join(tmpdir, "framework.png")
        sev_result = render_severity_chart(report, severity_path)
        fw_result = render_framework_chart(report, framework_path)
        if sev_result:
            img = Image(sev_result, width=6 * inch, height=3.5 * inch)
            story.append(img)
            story.append(Spacer(1, 0.2 * inch))
            chart_paths_to_cleanup.append(sev_result)
        if fw_result:
            img = Image(fw_result, width=6 * inch, height=3.5 * inch)
            story.append(img)
            story.append(Spacer(1, 0.2 * inch))
            chart_paths_to_cleanup.append(fw_result)

        sorted_findings = sorted(
            report.findings,
            key=lambda f: SEVERITY_ORDER.index(f.severity) if f.severity in SEVERITY_ORDER else 99,
        )

        table_data = [["#", "Title", "Severity", "Framework / Control", "Source"]]
        if not sorted_findings:
            table_data.append(["", "No findings", "", "", ""])
        else:
            for i, finding in enumerate(sorted_findings, start=1):
                fw_text = f"{finding.framework} — {finding.control_id}" if finding.framework and finding.control_id else "—"
                table_data.append([
                    str(i),
                    finding.title[:50] if len(finding.title) > 50 else finding.title,
                    finding.severity,
                    fw_text,
                    finding.source_file or "—",
                ])

        findings_table = Table(table_data, colWidths=[0.4 * inch, 2.5 * inch, 1 * inch, 2 * inch, 1.5 * inch])
        findings_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("TOPPADDING", (0, 0), (-1, 0), 8),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))

        for row_idx, finding in enumerate(sorted_findings, start=1):
            sev_color = _severity_color(finding.severity)
            findings_table.setStyle(TableStyle([
                ("TEXTCOLOR", (2, row_idx), (2, row_idx), colors.HexColor(sev_color)),
                ("FONTNAME", (2, row_idx), (2, row_idx), "Helvetica-Bold"),
            ]))

        story.append(findings_table)
        story.append(Spacer(1, 0.3 * inch))

        story.append(Paragraph("5. Findings", ss["Heading1"]))
        if not sorted_findings:
            story.append(Paragraph("No findings were identified in the reviewed material.", ss["BodyText"]))

        for i, finding in enumerate(sorted_findings, start=1):
            story.append(Paragraph(f"5.{i} {finding.title}", ss["Heading2"]))
            color = _severity_color(finding.severity)
            story.append(Paragraph(
                f'<b>Severity: <font color="{color}">{finding.severity}</font></b>',
                ss["BodyText"],
            ))
            if finding.source_file:
                story.append(Paragraph(f"<b>Source:</b> {finding.source_file}", ss["BodyText"]))
            if finding.framework and finding.control_id:
                story.append(Paragraph(
                    f"<b>Framework Mapping:</b> {finding.framework} — {finding.control_id}",
                    ss["BodyText"],
                ))
            story.append(Paragraph("<b>Description</b>", ss["Label"]))
            story.append(Paragraph(finding.description, ss["BodyText"]))
            story.append(Paragraph("<b>Evidence</b>", ss["Label"]))
            story.append(Paragraph(finding.evidence, ss["BodyText"]))
            story.append(Paragraph("<b>Remediation</b>", ss["Label"]))
            story.append(Paragraph(finding.remediation, ss["BodyText"]))
            story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("6. Recommendations Summary", ss["Heading1"]))
        items = [ListItem(Paragraph(rec, ss["BodyText"])) for rec in report.recommendations_summary]
        if items:
            story.append(ListFlowable(items, bulletType="bullet"))

        if report.diagram and is_valid_mermaid(report.diagram):
            story.append(PageBreak())
            story.append(Paragraph("7. Architecture Diagram (Mermaid source)", ss["Heading1"]))
            diagram_text = "<br/>".join(report.diagram.splitlines())
            story.append(Paragraph(f'<pre fontName="Courier" fontSize="9">{diagram_text}</pre>', ss["Mono"]))

        doc.build(story)
    finally:
        for path in chart_paths_to_cleanup:
            try:
                os.remove(path)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass

    return output_path
