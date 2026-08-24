from fastapi import APIRouter, HTTPException

from api.schemas import DocumentsListResponse

from document_manager import list_documents


router = APIRouter()


@router.get(
    "/documents",
    response_model=DocumentsListResponse
)
def get_documents():

    try:

        documents = list_documents()


        return {
            "documents": documents
        }


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )