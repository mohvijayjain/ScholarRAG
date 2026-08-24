from database import (
    SessionLocal,
    DocumentHistory
)


def list_documents():

    db = SessionLocal()


    docs = db.query(
        DocumentHistory
    ).all()


    result = []


    for doc in docs:

        result.append(

            {
                "id": doc.id,

                "document_id": doc.document_id,

                "filename": doc.filename,

                "chunks": doc.chunks,

                "upload_date": doc.upload_date,

                "status": doc.status
            }

        )


    db.close()


    return result



def get_document_by_id(
    document_id
):

    db = SessionLocal()


    doc = db.query(
        DocumentHistory
    ).filter(

        DocumentHistory.document_id
        ==
        document_id

    ).first()


    db.close()


    if not doc:
        return None


    return {

        "document_id":
            doc.document_id,

        "filename":
            doc.filename,

        "chunks":
            doc.chunks

    }



if __name__ == "__main__":


    docs = list_documents()


    for doc in docs:

        print(doc)