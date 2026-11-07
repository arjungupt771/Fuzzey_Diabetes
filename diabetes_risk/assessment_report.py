"""Generate a brief PDF for a research-tool assessment result."""

from datetime import datetime
from io import BytesIO


def build_assessment_pdf(result, conformal_result=None, target_definition=None):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=0.65 * inch,
                                 leftMargin=0.65 * inch, topMargin=0.65 * inch,
                                 bottomMargin=0.65 * inch)
    styles = getSampleStyleSheet()
    story = [
        Paragraph('Research Risk Assessment Summary', styles['Title']),
        Paragraph(datetime.now().strftime('Generated %Y-%m-%d %H:%M'), styles['Normal']),
        Spacer(1, 12),
        Paragraph('This document reports outputs from a research prototype. It is not a diagnosis, screening result, or substitute for professional medical evaluation.', styles['BodyText']),
        Spacer(1, 12),
        Paragraph(f"Most likely model class: <b>{result['predicted_label']}</b>", styles['Heading2']),
    ]
    probability_rows = [['Class', 'Model probability']]
    probability_rows.extend([[label.title(), f'{value:.3f}'] for label, value in result['probabilities'].items()])
    probability_table = Table(probability_rows, hAlign='LEFT')
    probability_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eef7')),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story += [probability_table, Spacer(1, 12)]
    story.append(Paragraph(f"Fuzzy score (BMI, age, family history): {result['fuzzy_score']:.3f}", styles['BodyText']))
    combined = ('not defined for these class labels' if result['fused_score'] is None
                else f"{result['fused_score']:.3f}")
    story.append(Paragraph(f'Combined prototype score: {combined}', styles['BodyText']))
    story.append(Paragraph('Fuzzy rule strengths: ' + ', '.join(
        f'{key.title()} {value:.2f}' for key, value in result['fuzzy_assessment']['rule_strengths'].items()
    ), styles['BodyText']))
    if conformal_result is not None:
        labels = {0: 'Normal', 1: 'Prediabetic', 2: 'Diabetic'}
        selected = [labels.get(label, str(label)) for label in conformal_result]
        story.append(Paragraph('Conformal prediction set (nominal 90%): ' +
                               (', '.join(selected) if selected else 'empty / abstain'), styles['BodyText']))
    story += [Spacer(1, 18), Paragraph(
        f'Interpretation: target definition is {target_definition or "unspecified"}. Model explanations and fuzzy scores describe this prototype and do not establish causation or clinical utility.',
        styles['BodyText'])]
    document.build(story)
    return buffer.getvalue()
