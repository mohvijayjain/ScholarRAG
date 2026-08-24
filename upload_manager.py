import shutil
from pathlib import Path

from ingest import ingest_pdf
from database import (
    SessionLocal,
    DocumentHistory
)


# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).parent

DATA_DIR = (
    BASE_DIR / "data"
)

DATA_DIR.mkdir(
    exist_ok=True
)



# =====================================================
# CHECK DUPLICATE
# =====================================================

def document_exists(filename):

    db = SessionLocal()


    document = db.query(
        DocumentHistory
    ).filter(

        DocumentHistory.filename
        ==
        filename

    ).first()


    db.close()


    return document is not None



# =====================================================
# UPLOAD DOCUMENT
# =====================================================

def upload_document(
    file_path
):

    file_path = Path(
        file_path
    )


    if not file_path.exists():

        raise FileNotFoundError(
            file_path
        )



    filename = file_path.name



    # -----------------------------
    # Duplicate check
    # -----------------------------

    if document_exists(filename):

        return {

            "status":
                "already_exists",

            "message":
                f"{filename} already uploaded"

        }



    # -----------------------------
    # Copy PDF
    # -----------------------------

    destination = (
        DATA_DIR / filename
    )


    if file_path.resolve() != destination.resolve():

        shutil.copy(
            file_path,
            destination
        )

    else:

        print(
            "File already exists in data folder"
        )


    print(
        f"Copied: {destination}"
    )



    # -----------------------------
    # Ingest
    # -----------------------------

    result = ingest_pdf(
        destination
    )



    return {

        "status":
            "success",

        "filename":
            filename,

        "document_id":
            result["document_id"],

        "chunks":
            result["chunks_added"]

    }



# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":


    result = upload_document(
        "sample.pdf"
    )


    print(result)