from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from sentence_transformers import CrossEncoder

load_dotenv()

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever



embeddings = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-small-en-v1.5"   
)

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

all_docs = db.get(include=["documents", "metadatas"])

documents = []

for text, metadata in zip(
    all_docs["documents"],
    all_docs["metadatas"]
):
    from langchain_core.documents import Document

    documents.append(
        Document(
            page_content=text,
            metadata=metadata
        )
    )

bm25_retriever = BM25Retriever.from_documents(
    documents
)

bm25_retriever.k = 15

vector_retriever = db.as_retriever(
    search_type = "mmr",
    search_kwargs={
    "k": 15
    }
)

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

llm = ChatNVIDIA(
    model="meta/llama-3.1-70b-instruct",
    temperature=0.2,
    max_completion_tokens=1024,
    timeout = 180
)

def rerank_docs(question, docs, top_k=4):
    pairs = [
        [question, doc.page_content]
        for doc in docs
    ]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, docs),
        key=lambda x:x[0],
        reverse=True
    )

    return [
        doc
        for score, doc in ranked[:top_k]
    ]

def ask(question, debug=False):
    docs = retriever.invoke(question)
    
    docs = rerank_docs(question, docs)

    if debug:
        for i, doc in enumerate(docs):
            print(f"\n========== CHUNK {i} ==========")
            print(doc.metadata)
            print(doc.page_content[:500])

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

    response = llm.invoke(prompt)

    print("\n========== SOURCES ==========")
    for i, doc in enumerate(docs):
        print(
            f"""
            {i+1}.
            Paper: {doc.metadata.get('paper_name')}
            Section: {doc.metadata.get('section')}
            Page: {doc.metadata.get('page')}
            """
            )
    return {
    "answer": response.content,
    "retrieved_chunks": docs
}


if __name__ == "__main__":
    questions = [
    "What is the main idea of the RAG paper?",
    "How does the retriever work in RAG?",
    "What problem does RAG solve?",
    "How does LoRA reduce trainable parameters?",
    "What is Self-RAG?"
    ]
    
    for question in questions:
        print("\n\n==============================")
        print("QUESTION:", question)
        print("==============================")

        answer = ask(question, debug=True)
        print(answer)