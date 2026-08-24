import time
import requests

from urllib3.exceptions import ProtocolError
from dotenv import load_dotenv
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from sentence_transformers import CrossEncoder

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document


# ============================================================
# ENV
# ============================================================

load_dotenv()


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# ============================================================
# RERANKER
# ============================================================

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)



# ============================================================
# CHROMA DATABASE
# ============================================================

BASE_DIR = Path(__file__).parent

CHROMA_PATH = (
    BASE_DIR
    / "database"
    / "chroma_db"
)


db = Chroma(
    persist_directory=str(CHROMA_PATH),
    embedding_function=embeddings
)



# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents():

    all_docs = db.get(
        include=[
            "documents",
            "metadatas"
        ]
    )


    docs = []


    print(
        "Chroma chunks loaded:",
        len(all_docs["documents"])
    )


    for text, metadata in zip(
        all_docs["documents"],
        all_docs["metadatas"]
    ):

        docs.append(

            Document(
                page_content=text,
                metadata=metadata
            )

        )


    return docs



documents = load_documents()



# ============================================================
# VECTOR RETRIEVER
# ============================================================

def get_vector_retriever(
    document_ids=None
):

    search_kwargs = {
        "k":15
    }


    if document_ids:

        if len(document_ids) == 1:

            search_kwargs["filter"] = {
                "document_id": document_ids[0]
            }

        else:

            search_kwargs["filter"] = {

                "$or":[

                    {
                        "document_id": doc_id
                    }

                    for doc_id in document_ids

                ]

            }


    return db.as_retriever(

        search_type="mmr",

        search_kwargs=search_kwargs

    )




# ============================================================
# BM25 RETRIEVER
# ============================================================

def get_bm25_retriever(
    document_ids=None
):


    current_documents = load_documents()


    filtered_documents = current_documents



    if document_ids:


        filtered_documents = [

            doc

            for doc in current_documents

            if doc.metadata.get(
                "document_id"
            ) in document_ids

        ]



    print(
        "BM25 documents:",
        len(filtered_documents)
    )



    if not filtered_documents:

        return None



    bm25 = BM25Retriever.from_documents(

        filtered_documents

    )


    bm25.k = 15


    return bm25





# ============================================================
# GLOBAL RETRIEVER
# ============================================================

default_vector = get_vector_retriever()

default_bm25 = get_bm25_retriever()



retriever = EnsembleRetriever(

    retrievers=[

        default_vector,

        default_bm25

    ],


    weights=[

        0.6,

        0.4

    ]

)





# ============================================================
# NVIDIA LLM
# ============================================================

llm = ChatNVIDIA(

    model="meta/llama-3.1-70b-instruct",

    temperature=0.2,

    max_completion_tokens=1024,

    timeout=200

)




# ============================================================
# RERANK
# ============================================================

def rerank_docs(

    question,

    docs,

    top_k=4

):


    pairs=[

        [

            question,

            doc.page_content

        ]

        for doc in docs

    ]


    scores = reranker.predict(
        pairs
    )


    ranked = sorted(

        zip(scores,docs),

        key=lambda x:x[0],

        reverse=True

    )


    return [

        doc

        for score,doc in ranked[:top_k]

    ]





# ============================================================
# NVIDIA RETRY
# ============================================================

def invoke_with_retry(

    prompt,

    max_retries=6

):


    for attempt in range(max_retries):

        try:

            return llm.invoke(
                prompt
            )


        except Exception as e:


            error=str(e)


            if (

                "429" in error

                or

                "Too Many Requests" in error

            ):


                wait=min(
                    5*(2**attempt),
                    60
                )


                print(
                    f"NVIDIA rate limit. Waiting {wait}s"
                )


                time.sleep(wait)



            elif (

                isinstance(
                    e,
                    (
                        requests.exceptions.ConnectionError,
                        ConnectionResetError,
                        ProtocolError
                    )
                )

                or

                "Connection aborted" in error

            ):


                wait=min(
                    5*(2**attempt),
                    60
                )


                print(
                    f"Connection issue. Waiting {wait}s"
                )


                time.sleep(wait)


            else:

                raise



    raise RuntimeError(
        "NVIDIA failed"
    )





# ============================================================
# ASK FUNCTION
# ============================================================

def ask(

    question,

    document_ids=None,

    debug=False

):


    # -----------------------------
    # RETRIEVAL
    # -----------------------------


    if document_ids:


        print(
            "Searching selected documents:",
            document_ids
        )


        vector = get_vector_retriever(
            document_ids
        )


        bm25 = get_bm25_retriever(
            document_ids
        )



        if bm25:


            selected_retriever = EnsembleRetriever(

                retrievers=[

                    vector,

                    bm25

                ],

                weights=[

                    0.6,

                    0.4

                ]

            )


        else:

            selected_retriever = vector



        docs = selected_retriever.invoke(
            question
        )



    else:


        docs = retriever.invoke(
            question
        )




    # -----------------------------
    # RERANK
    # -----------------------------

    docs = rerank_docs(

        question,

        docs

    )



    # -----------------------------
    # DEBUG
    # -----------------------------

    if debug:


        for i,doc in enumerate(docs):

            print(
                f"\n====== CHUNK {i} ======"
            )

            print(
                doc.metadata
            )

            print(
                doc.page_content[:300]
            )




    # -----------------------------
    # CONTEXT
    # -----------------------------

    context="\n\n".join(

        [

f"""
Source: {doc.metadata.get('source')}
Page: {doc.metadata.get('page')}

Content:
{doc.page_content}
"""

        for doc in docs

        ]

    )



    # -----------------------------
    # PROMPT
    # -----------------------------

    prompt=f"""

You are ScholarRAG, an AI research assistant.

Rules:

1. Use only provided context.
2. Do not use outside knowledge.
3. If answer is not available say:
"I could not find this information in the provided documents."
4. Explain technically.
5. Mention paper and section.


Context:

{context}


Question:

{question}


Answer:

"""



    response = invoke_with_retry(
        prompt
    )



    # -----------------------------
    # SOURCES
    # -----------------------------

    print(
        "\n========== SOURCES =========="
    )


    for i,doc in enumerate(docs):

        print(

f"""
{i+1}.
Paper: {doc.metadata.get('paper_name')}
Section: {doc.metadata.get('section')}
Page: {doc.metadata.get('page')}
"""

        )



    return {

        "answer":
        response.content,


        "retrieved_chunks":
        docs

    }