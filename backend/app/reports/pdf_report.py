from datetime import datetime

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, ListFlowable, ListItem
)

from .schema import SecurityReport, SEVERITY_ORDER

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
    return ss


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

    story.append(Paragraph("4. Findings", ss["Heading1"]))
    sorted_findings = sorted(
        report.findings,
        key=lambda f: SEVERITY_ORDER.index(f.severity) if f.severity in SEVERITY_ORDER else 99,
    )
    if not sorted_findings:
        story.append(Paragraph("No findings were identified in the reviewed material.", ss["BodyText"]))

    for i, finding in enumerate(sorted_findings, start=1):
        story.append(Paragraph(f"4.{i} {finding.title}", ss["Heading2"]))
        color = SEVERITY_COLORS.get(finding.severity, colors.black)
        story.append(Paragraph(
            f'<b>Severity: <font color="{color.hexval()}">{finding.severity}</font></b>',
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

    story.append(Paragraph("5. Recommendations Summary", ss["Heading1"]))
    items = [ListItem(Paragraph(rec, ss["BodyText"])) for rec in report.recommendations_summary]
    if items:
        story.append(ListFlowable(items, bulletType="bullet"))

    doc.build(story)
    return output_path
