from tortoise import fields
from tortoise.models import Model

class PDFLog(Model):
    id = fields.IntField(pk=True)
    
    filename = fields.CharField(max_length=255)
    collection_name = fields.CharField(max_length=255)        

    title = fields.CharField(max_length=512)
    abstract = fields.TextField(null=True)
    summary = fields.TextField(null=True)

    full_text = fields.TextField(null=True)
    text_excerpt = fields.TextField(null=True)

    novelty_insights = fields.JSONField(null=True)
    research_gaps = fields.JSONField(null=True)

    novel_insights = fields.TextField(null=True)
    similarities = fields.TextField(null=True)
    missing_gaps = fields.TextField(null=True)
    
    uploaded_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "pdf_logs"
