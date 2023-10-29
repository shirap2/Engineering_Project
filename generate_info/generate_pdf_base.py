from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

class BasePDFGenerator:

    def __init__(self, filename):
        self.filename = filename
        self.elements = []


    def add_spacer(self, width, height):
        spacer = Spacer(width, height)
        self.elements.append(spacer)

    def add_text(self, text, style):
        paragraph = Paragraph(text, style)
        self.elements.append(paragraph)

    def generate_pdf(self):
        doc = SimpleDocTemplate(self.filename)
        doc.build(self.elements)


    def get_file_title(title_string: str,sub_title:str):

        title_style = getSampleStyleSheet()['Title']
        title = Paragraph(title_string, title_style)

        subtitle_style = getSampleStyleSheet()['Title']
        sub_title = Paragraph(sub_title, subtitle_style)

        return [title, sub_title]


