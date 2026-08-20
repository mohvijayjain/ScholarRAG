import time
import requests

from urllib3.exceptions import ProtocolError
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from sentence_transformers import CrossEncoder

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document


# ============================================================
# LOAD ENVIRONMENT VARIABLES
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
# CHROMA VECTOR DATABASE
# ============================================================

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


# ============================================================
# LOAD ALL DOCUMENTS FROM CHROMA
# ============================================================

all_docs = db.get(
    include=["documents", "metadatas"]
)

documents = []

for text, metadata in zip(
    all_docs["documents"],
    all_docs["metadatas"]
):

    documents.append(
        Document(
            page_content=text,
            metadata=metadata
        )
    )


# ============================================================
# BM25 RETRIEVER
# ============================================================

bm25_retriever = BM25Retriever.from_documents(
    documents
)

bm25_retriever.k = 15


# ============================================================
# VECTOR RETRIEVER
# ============================================================

vector_retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 15
    }
)


# ============================================================
# ENSEMBLE RETRIEVER
# ============================================================

retriever = EnsembleRetriever(
    retrievers=[
        vector_retriever,
        bm25_retriever
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
# RERANK DOCUMENTS
# ============================================================

def rerank_docs(question, docs, top_k=4):

    pairs = [
        [question, doc.page_content]
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    return [
        doc
        for score, doc in ranked[:top_k]
    ]


# ============================================================
# NVIDIA API RETRY HANDLER
# ============================================================

def invoke_with_retry(prompt, max_retries=6):

    for attempt in range(max_retries):

        try:

            response = llm.invoke(prompt)

            return response

        except Exception as e:

            error = str(e)

            is_rate_limit = (
                "429" in error
                or "Too Many Requests" in error
            )

            is_connection_error = (
                isinstance(e, (
                    requests.exceptions.ConnectionError,
                    ConnectionResetError,
                    ProtocolError
                ))
                or "ConnectionResetError" in error
                or "Connection aborted" in error
                or "MaxRetryError" in error
                or "NameResolutionError" in error
                or "Failed to resolve" in error
            )

            if is_rate_limit:

                wait_time = min(
                    5 * (2 ** attempt),
                    60
                )

                print(
                    f"\n⚠️ NVIDIA rate limit (429)"
                )

                print(
                    f"Retry {attempt + 1}/{max_retries}"
                )

                print(
                    f"Waiting {wait_time} seconds..."
                )

                time.sleep(wait_time)

            elif is_connection_error:

                wait_time = min(
                    5 * (2 ** attempt),
                    60
                )

                print(
                    f"\n⚠️ NVIDIA connection error"
                )

                print(
                    f"Error: {type(e).__name__}"
                )

                print(
                    f"Retry {attempt + 1}/{max_retries}"
                )

                print(
                    f"Waiting {wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:

                raise

    raise RuntimeError(
        f"NVIDIA API failed after {max_retries} retries."
    )


# ============================================================
# MAIN RAG FUNCTION
# ============================================================

def ask(
    question,
    debug=False
):

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    docs = retriever.invoke(
        question
    )


    # --------------------------------------------------------
    # RERANKING
    # --------------------------------------------------------

    docs = rerank_docs(
        question,
        docs
    )


    # --------------------------------------------------------
    # DEBUG INFORMATION
    # --------------------------------------------------------

    if debug:

        for i, doc in enumerate(docs):

            print(
                f"\n========== CHUNK {i} =========="
            )

            print(
                doc.metadata
            )

            print(
                doc.page_content[:500]
            )


    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context = "\n\n".join(
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


    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are ScholarRAG, an AI research assistant.

Your task is to answer questions about research papers.

Rules:

1. Use ONLY the provided context.
2. Do not use outside knowledge.
3. If the answer is not present in the context, clearly say:
"I could not find this information in the provided documents."
4. Explain concepts clearly with technical depth.
5. Always mention the relevant paper and section when possible.
6. Do not summarize unrelated papers.

Retrieved Context:

{context}


Question:

{question}


Answer format:

## Answer

<Explain the answer>

## Sources

- Paper:
- Section:
- Page:

"""


    # --------------------------------------------------------
    # NVIDIA LLM CALL WITH RETRY
    # --------------------------------------------------------

    response = invoke_with_retry(
        prompt
    )


    # --------------------------------------------------------
    # PRINT SOURCES
    # --------------------------------------------------------

    print(
        "\n========== SOURCES =========="
    )

    for i, doc in enumerate(docs):

        print(
            f"""
{i + 1}.
Paper: {doc.metadata.get('paper_name')}
Section: {doc.metadata.get('section')}
Page: {doc.metadata.get('page')}
"""
        )


    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return {
        "answer": response.content,
        "retrieved_chunks": docs
    }


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    questions = [

        "What is the main idea of the RAG paper?",

        "How does the retriever work in RAG?",

        "What problem does RAG solve?",

        "How does LoRA reduce trainable parameters?",

        "What is Self-RAG?"

    ]


    for question in questions:

        print(
            "\n\n=============================="
        )

        print(
            "QUESTION:",
            question
        )

        print(
            "=============================="
        )


        answer = ask(
            question,
            debug=True
        )


        print(
            answer
        )