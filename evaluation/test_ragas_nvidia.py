import os
from dotenv import load_dotenv

from openai import AsyncOpenAI
from datasets import Dataset

from ragas import evaluate
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
)

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from ragas.run_config import RunConfig


# =====================================================
# ENV
# =====================================================

load_dotenv()

nvidia_api_key = os.getenv("NVIDIA_API_KEY")

if not nvidia_api_key:
    raise ValueError("NVIDIA_API_KEY is not set.")


# =====================================================
# NVIDIA CLIENT
# =====================================================

client = AsyncOpenAI(
    api_key=nvidia_api_key,
    base_url="https://integrate.api.nvidia.com/v1"
)


# =====================================================
# RAGAS LLM
# =====================================================

evaluator_llm = llm_factory(
    "meta/llama-3.1-70b-instruct",
    client=client
)


# =====================================================
# NVIDIA EMBEDDINGS
# =====================================================

evaluator_embedding = LangchainEmbeddingsWrapper(
    NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        api_key=nvidia_api_key
    )
)


# =====================================================
# ONE TEST SAMPLE
# =====================================================

dataset = Dataset.from_list([
    {
        "question": "What is machine learning?",
        "answer": "Machine learning is a method where computers learn patterns from data.",
        "contexts": [
            "Machine learning is a method in which computers learn patterns from data and use those patterns to make predictions."
        ],
        "ground_truth": "Machine learning is a method where computers learn patterns from data."
    }
])


# =====================================================
# TEST
# =====================================================

print("\nStarting RAGAS diagnostic...\n")


run_config = RunConfig(
    max_workers=1,
    timeout=200
)


try:

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embedding,
        run_config=run_config
    )

    print("\n==============================")
    print("RESULT")
    print("==============================")

    print(result.to_pandas())

except Exception as e:

    print("\n==============================")
    print("RAGAS FAILED")
    print("==============================")

    print(type(e).__name__)
    print(str(e))