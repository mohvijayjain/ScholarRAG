import json
from datasets import Dataset
from src.rag_pipeline import ask

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)


with open(
    "evaluation/test_questions.json"
) as f:
    questions = json.load(f)


samples = []


for item in questions:

    result = rag_query(
        item["question"]
    )

    contexts = [
        chunk["text"]
        for chunk in result["retrieved_chunks"]
    ]

    samples.append(
        {
            "question": item["question"],
            "answer": result["answer"],
            "contexts": contexts,
            "ground_truth": item["ground_truth"]
        }
    )


dataset = Dataset.from_list(samples)


scores = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)


print(scores)


scores.to_pandas().to_csv(
    "ragas_results.csv",
    index=False
)