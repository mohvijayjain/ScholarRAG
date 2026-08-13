import os
import re

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


DATA_PATH = "data/"
DB_PATH = "chroma_db/"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# =========================
# Embedding Model
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# =========================
# Clean PDF Text
# =========================

def clean_text(text):

    patterns = [
        r"#\s*\*?\*?References\*?\*?.*",
        r"#\s*\*?\*?Acknowledgments\*?\*?.*",
        r"#\s*\*?\*?Appendix\*?\*?.*",
        r"#\s*\*?\*?Broader Impact\*?\*?.*"
    ]

    for pattern in patterns:
        text = re.split(
            pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL
        )[0]

    return text.strip()


# =========================
# extract section
# =========================
def extract_section(text):
    
    lines = text.split("\n")

    section = "Unknown"

    for line in lines:
        line = line.strip()

        if line.startswith("#"):

            section = (
                line
                .replace("#", "")
                .replace("*", "")
                .strip()
            )

            break

    return section

# =========================
# Text Splitter
# =========================

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



# =========================
# Load Documents
# =========================

def load_documents():

    documents = []

    for file in os.listdir(DATA_PATH):

        if file.endswith(".pdf"):

            path = os.path.join(
                DATA_PATH,
                file
            )

            print(f"Loading: {file}")

            loader = PyMuPDF4LLMLoader(path)

            docs = loader.load()


            for doc in docs:
                doc.page_content = clean_text(
                    doc.page_content
                )

                doc.metadata["paper_name"] = file

                doc.metadata["section"] = extract_section(
                    doc.page_content
                )

                doc.metadata["source"] = path

                doc.metadata["page"] = (
                    doc.metadata.get("page")
                )


            documents.extend(docs)


    return documents



# =========================
# Build / Update Chroma
# =========================

if os.path.exists(DB_PATH):

    print("Existing ChromaDB found")

    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )


    existing = db.get()

    existing_sources = set()


    for metadata in existing["metadatas"]:

        if metadata:

            existing_sources.add(
                metadata.get("source")
            )


    print(
        f"Existing documents: {len(existing_sources)}"
    )


    new_documents = []


    for file in os.listdir(DATA_PATH):

        if file.endswith(".pdf"):

            path = os.path.join(
                DATA_PATH,
                file
            )


            if path in existing_sources:

                print(
                    f"Skipping: {file}"
                )

                continue


            loader = PyMuPDF4LLMLoader(path)

            docs = loader.load()


            for doc in docs:

                doc.page_content = clean_text(
                    doc.page_content
                )

                doc.metadata["paper_name"] = file
                doc.metadata["source"] = path


            new_documents.extend(docs)



    if new_documents:

        chunks = splitter.split_documents(
            new_documents
        )

        print(
            f"Adding chunks: {len(chunks)}"
        )


        db.add_documents(
            chunks
        )

        print(
            "New documents added"
        )

    else:

        print(
            "No new documents"
        )



else:

    print(
        "Creating new ChromaDB"
    )


    documents = load_documents()


    if not documents:

        raise Exception(
            "No PDF found"
        )


    chunks = splitter.split_documents(
        documents
    )


    print(
        f"Total chunks created: {len(chunks)}"
    )


    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )


    print(
        "ChromaDB created successfully"
    )