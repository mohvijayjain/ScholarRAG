from fastapi import APIRouter, HTTPException

from api.schemas import (
    ChatRequest,
    ChatResponse
)

from rag_pipeline import ask


router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse
)
def chat(request: ChatRequest):

    try:

        result = ask(
            question=request.question,
            document_ids=request.document_ids
        )


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


    sources = []


    for doc in result["retrieved_chunks"]:

        sources.append({

            "paper":
            doc.metadata.get("paper_name"),

            "section":
            doc.metadata.get("section"),

            "page":
            doc.metadata.get("page")

        })


    return {

        "answer":
        result["answer"],

        "sources":
        sources

    }