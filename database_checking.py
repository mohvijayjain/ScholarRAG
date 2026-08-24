from database import SessionLocal, DocumentHistory

db = SessionLocal()

print(
    db.query(DocumentHistory).count()
)

db.close()