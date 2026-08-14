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


# IMPORTANT:
# This must match rag_pipeline.py embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"



# =========================
# Embedding Model
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
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
# Section Tracking
# =========================

def get_section(text, current_section="Unknown"):

    lines = text.split("\n")


    found_section = current_section


    for line in lines:

        line = line.strip()


        if re.match(r"^#+\s", line):


            section = re.sub(
                r"^#+\s*",
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
# Process Single PDF
# =========================

def process_pdf(path):


    print(
        f"Loading: {os.path.basename(path)}"
    )


    loader = PyMuPDF4LLMLoader(path)


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


        doc.metadata["paper_name"] = (
            os.path.basename(path)
        )


        doc.metadata["section"] = (
            current_section
        )


        doc.metadata["source"] = path


        doc.metadata["page"] = (
            doc.metadata.get("page")
        )


    return docs





# =========================
# Load All PDFs
# =========================

def load_documents():


    documents = []


    for file in os.listdir(DATA_PATH):


        if file.endswith(".pdf"):


            path = os.path.join(
                DATA_PATH,
                file
            )


            documents.extend(
                process_pdf(path)
            )


    return documents





# =========================
# Build / Update Chroma
# =========================

def main():


    if os.path.exists(DB_PATH):


        print(
            "Existing ChromaDB found"
        )


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



                new_documents.extend(
                    process_pdf(path)
                )




        if new_documents:


            chunks = splitter.split_documents(
                new_documents
            )


            for chunk in chunks:

                if "section" not in chunk.metadata:

                    chunk.metadata["section"] = "Unknown"



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



        for chunk in chunks:


            if "section" not in chunk.metadata:

                chunk.metadata["section"] = "Unknown"




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
if __name__ == "__main__":

    main()