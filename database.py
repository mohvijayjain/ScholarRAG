from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime
)

from sqlalchemy.orm import (
    declarative_base,
    sessionmaker
)

from datetime import datetime


DATABASE_URL = "sqlite:///documents.db"


engine = create_engine(
    DATABASE_URL
)


Base = declarative_base()


class DocumentHistory(Base):

    __tablename__ = "documents"


    id = Column(
        Integer,
        primary_key=True
    )


    document_id = Column(
        String,
        unique=True
    )


    filename = Column(
        String
    )


    file_path = Column(
        String
    )


    chunks = Column(
        Integer
    )


    upload_date = Column(
        DateTime,
        default=datetime.now
    )


    status = Column(
        String,
        default="completed"
    )



Base.metadata.create_all(
    engine
)


SessionLocal = sessionmaker(
    bind=engine
)