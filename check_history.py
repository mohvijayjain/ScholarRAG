from database import (
    SessionLocal,
    DocumentHistory
)


db = SessionLocal()


docs = db.query(
    DocumentHistory
).all()


for doc in docs:

    print(
        {
            "filename": doc.filename,
            "chunks": doc.chunks,
            "date": doc.upload_date,
            "status": doc.status
        }
    )


db.close()