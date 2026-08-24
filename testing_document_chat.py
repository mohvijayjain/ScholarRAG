from rag_pipeline import ask


result = ask(
    "Compare the approaches used in these papers",
    document_ids=[
        "3da138e3-143d-4ea6-a91a-a9fdd77a2989",
        "8d0ece11-5c16-42d5-b61d-f3cb95c88df7"
    ],
    debug=True
)


print(result["answer"])