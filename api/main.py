from fastapi import FastAPI
from api.routes import documents
from api.routes import chat
from api.routes import upload

app = FastAPI(
    title="ScholarRAG API",
    description="Research Paper RAG Assistant",
    version="1.0"
)

app.include_router(
    documents.router
)

app.include_router(
    chat.router
)

app.include_router(
    upload.router
)

@app.get("/")
def home():
    return {
        "message":
        "ScholarRAG API is running"
    }