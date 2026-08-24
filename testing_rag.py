from rag_pipeline import ask

result = ask(
    "What is the main contribution?",
    debug=True
)

print(result["answer"])