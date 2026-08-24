from pydantic import BaseModel
from datetime import datetime

class DocumentResponse(BaseModel):
    id: int
    document_id: str
    filename: str
    chunks: int
    upload_date: datetime
    status: str



class ChatRequest(BaseModel):
    question: str
    document_ids: list[str] | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list


class UploadResponse(BaseModel):
    status: str
    message: str
    document: dict
    

class DocumentsListResponse(BaseModel):
    documents: list[DocumentResponse]