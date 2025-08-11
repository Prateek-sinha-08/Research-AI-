from pydantic import BaseModel
from typing import List, Optional

class TopicRequest(BaseModel):
    topic: str
    
class SummarizeByFilenamesRequest(BaseModel):
    filenames: List[str]

class CompareByFilenamesRequest(BaseModel):
    filenames: Optional[List[str]] = None
    
class PaperSummary(BaseModel):
    title: str
    abstract: str
    authors: List[str]
    link: Optional[str] = None

class PDFCompareRequest(BaseModel):
    paper_titles: List[str]
    paper_texts: List[str]

class ComparisonResult(BaseModel):
    title: str
    novel_insights: List[str]
    similarities: List[str]
    missing_gaps: List[str]

class UploadResponse(BaseModel):
    message: str
    collections: List[str]
    papers: List[PaperSummary]

class AskQuestionRequest(BaseModel):
    question: str
    collection_names: List[str]

class AskQuestionResponse(BaseModel):
    answer: str

# 🧠 ADD THIS BELOW
class SimpleSummary(BaseModel):
    title: str
    summary: str
