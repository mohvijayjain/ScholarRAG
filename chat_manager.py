from rag_pipeline import ask


def chat(
    question,
    selected_document_id=None
):

    if selected_document_id:

        return ask(
            question,
            document_id=selected_document_id
        )

    else:

        return ask(
            question
        )