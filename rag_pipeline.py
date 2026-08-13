from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()



embeddings = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-small-en-v1.5"   
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = db.as_retriever(
    search_type = "mmr",
    search_kwargs={
    "k": 6
    }
)

llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    temperature=0.4,
    max_completion_tokens=2048,
)

def ask(question, debug=False):
    docs = retriever.invoke(question)

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
You are a research assistant.

Answer the question using ONLY the provided context.

If the context does not contain the answer, say:
"I could not find this information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    print("\n========== SOURCES ==========")
    for i, doc in enumerate(docs):
        print(f"{i+1}. {doc.metadata.get('source')} (Page {doc.metadata.get('page')})")

    return response.content


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