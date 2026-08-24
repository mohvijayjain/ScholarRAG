import os
import re
import uuid
from datetime import datetime
from pathlib import Path

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from database import (
    SessionLocal,
    DocumentHistory
)


# ============================================================
# PATH CONFIG
# ============================================================

BASE_DIR = Path(__file__).parent


DOCUMENT_PATH = (
    BASE_DIR / "data"
)


DB_PATH = (
    BASE_DIR
    / "database"
    / "chroma_db"
)


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300


EMBEDDING_MODEL = (
    "BAAI/bge-small-en-v1.5"
)


DOCUMENT_PATH.mkdir(
    exist_ok=True
)


DB_PATH.mkdir(
    exist_ok=True,
)


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)



# ============================================================
# CLEAN PDF TEXT
# ============================================================

def clean_text(text):

    patterns = [

        r"#\s*\*?References\*?.*",

        r"#\s*\*?Acknowledgments\*?.*",

        r"#\s*\*?Appendix\*?.*",

        r"#\s*\*?Broader Impact\*?.*"

    ]


    for pattern in patterns:

        text = re.split(
            pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL
        )[0]


    return text.strip()



# ============================================================
# SECTION TRACKING
# ============================================================

def get_section(
    text,
    current_section="Unknown"
):

    lines = text.split("\n")

    found_section = current_section


    for line in lines:

        line = line.strip()


        if re.match(
            r"^#+\s",
            line
        ):

            section = re.sub(
                r"^#+\s*\*?",
                "",
                line
            )


            section = (
                section
                .replace("*", "")
                .strip()
            )


            found_section = section


    return found_section



# ============================================================
# TEXT SPLITTER
# ============================================================

splitter = RecursiveCharacterTextSplitter(

    chunk_size=CHUNK_SIZE,

    chunk_overlap=CHUNK_OVERLAP,

    separators=[

        "\n# ",

        "\n## ",

        "\n\n",

        "\n",

        " "

    ]
)



# ============================================================
# PROCESS SINGLE PDF
# ============================================================

def process_pdf(path):


    print(
        f"\nProcessing: {path.name}"
    )


    document_id = str(
        uuid.uuid4()
    )


    loader = PyMuPDF4LLMLoader(
        str(path)
    )


    docs = loader.load()


    current_section = "Unknown"


    for doc in docs:


        doc.page_content = clean_text(
            doc.page_content
        )


        current_section = get_section(

            doc.page_content,

            current_section

        )


        doc.metadata.update(

            {

                "document_id":
                    document_id,


                "paper_name":
                    path.name,


                "source":
                    str(path),


                "section":
                    current_section,


                "upload_date":
                    str(datetime.now()),


                "page":
                    doc.metadata.get(
                        "page"
                    )

            }

        )


    return docs, document_id



# ============================================================
# CHROMA CONNECTION
# ============================================================

def get_chroma():

    return Chroma(

        persist_directory=str(
            DB_PATH
        ),

        embedding_function=embeddings

    )



# ============================================================
# INGEST SINGLE PDF
# ============================================================

def ingest_pdf(pdf_path):


    pdf_path = Path(pdf_path)


    if not pdf_path.exists():

        raise FileNotFoundError(
            pdf_path
        )


    db = get_chroma()



    # --------------------------------
    # Process PDF
    # --------------------------------

    documents, document_id = process_pdf(
        pdf_path
    )



    # --------------------------------
    # Create chunks
    # --------------------------------

    chunks = splitter.split_documents(
        documents
    )



    for chunk in chunks:

        if (
            "section"
            not in chunk.metadata
        ):

            chunk.metadata[
                "section"
            ] = "Unknown"



    print(
        f"Chunks created: {len(chunks)}"
    )



    # --------------------------------
    # Store in existing ChromaDB
    # --------------------------------

    db.add_documents(
        chunks
    )
    
    session = SessionLocal()
    record = DocumentHistory(
        document_id=document_id,
        filename=pdf_path.name,
        file_path=str(pdf_path),
        chunks=len(chunks),
        status="completed"
    )
    session.add(
        record
    )
    session.commit()
    session.close()


    print(
        "Added to ChromaDB"
    )


    return {

        "document_id":
            document_id,


        "filename":
            pdf_path.name,


        "chunks_added":
            len(chunks)

    }



# ============================================================
# INGEST ALL DOCUMENTS IN FOLDER
# ============================================================

def ingest_all_documents():


    pdf_files = [

        file

        for file in DOCUMENT_PATH.iterdir()

        if file.suffix.lower()
        == ".pdf"

    ]


    results = []


    for pdf in pdf_files:


        result = ingest_pdf(
            pdf
        )


        results.append(
            result
        )


    return results



# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":


    print(
        "Available PDFs:"
    )


    for file in DOCUMENT_PATH.iterdir():

        print(
            file.name
        )


    print(
        "\nTo ingest a PDF:"
    )


    print(
        "ingest_pdf('documents/file.pdf')"
    )