from fastapi import APIRouter, UploadFile, File, HTTPException

from api.schemas import UploadResponse

from upload_manager import upload_document

from pathlib import Path

import shutil


router = APIRouter()


UPLOAD_DIR = Path("uploads")

UPLOAD_DIR.mkdir(
    exist_ok=True
)



@router.post(
    "/upload",
    response_model=UploadResponse
)
def upload_pdf(
    file: UploadFile = File(...)
):

    # Validate PDF

    if file.content_type != "application/pdf":

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )


    try:

        file_path = UPLOAD_DIR / file.filename


        # Save temporary file

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )


        # Ingest document

        result = upload_document(
            file_path
        )


        return {

            "status": "completed",

            "message":
            "Document uploaded and indexed successfully",

            "document": result

        }


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )