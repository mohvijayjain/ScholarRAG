import os
import json
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from ragas.run_config import RunConfig

from datasets import Dataset
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)


# =====================================================
# LOAD ENVIRONMENT
# =====================================================

load_dotenv()


# =====================================================
# PATHS
# =====================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

from rag_pipeline import ask


# =====================================================
# NVIDIA CONFIG
# =====================================================

nvidia_api_key = os.getenv("NVIDIA_API_KEY")
hf_token = os.getenv("HF_TOKEN")


if not nvidia_api_key:
    raise ValueError(
        "NVIDIA_API_KEY is not set."
    )


if not hf_token:
    print(
        "Warning: HF_TOKEN not found. "
        "HuggingFace rate limits may apply."
    )


client = AsyncOpenAI(
    api_key=nvidia_api_key,
    base_url="https://integrate.api.nvidia.com/v1"
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
# NVIDIA RAGAS LLM
# =====================================================

evaluator_llm = llm_factory(
    "meta/llama-3.1-70b-instruct",
    client=client
)


# =====================================================
# CLEAN ANSWER FOR RAGAS
# =====================================================

def clean_answer(answer):

    if not answer:
        return ""

    # Remove source section from evaluation text
    if "## Sources" in answer:
        answer = answer.split(
            "## Sources",
            1
        )[0]

    # Remove answer heading
    answer = answer.replace(
        "## Answer",
        ""
    )

    return answer.strip()


# =====================================================
# LOAD DATASET
# =====================================================

DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "test_dataset.json"
)


with open(
    DATASET_PATH,
    "r",
    encoding="utf-8"
) as f:

    questions = json.load(f)


# =====================================================
# BATCH CONFIG
# =====================================================

BATCH_SIZE = 5

BREAK_TIME = 60

QUESTION_SLEEP = 7

MAX_RETRIES = 2


output_dir = (
    PROJECT_ROOT
    / "evaluation"
    / "batch_results_fixed"
)

output_dir.mkdir(
    exist_ok=True,
    parents=True
)


failed_log_path = (
    output_dir
    / "failed_questions.jsonl"
)


# =====================================================
# RUN BATCHES
# =====================================================
questions = questions[:5]
total_questions = len(questions)


for start in range(
    0,
    total_questions,
    BATCH_SIZE
):

    batch_number = (
        start // BATCH_SIZE
    ) + 1

    end = min(
        start + BATCH_SIZE,
        total_questions
    )

    batch_questions = questions[
        start:end
    ]


    output_file = (
        output_dir
        / f"batch_{batch_number}.csv"
    )


    # =================================================
    # RESUME SUPPORT
    # =================================================

    if output_file.exists():

        print(
            f"\nBatch {batch_number} already "
            f"completed. Skipping."
        )

        continue


    print(
        "\n================================="
    )

    print(
        f"Running Batch {batch_number}"
    )

    print(
        f"Questions {start + 1}-{end}"
    )

    print(
        "================================="
    )


    samples = []


    # =================================================
    # GENERATE RAG ANSWERS
    # =================================================

    for item in batch_questions:

        question = item["question"]


        print(
            "\nQuestion:"
        )

        print(
            question
        )


        result = None

        last_error = None


        # ---------------------------------------------
        # Retry ask()
        # ---------------------------------------------

        for attempt in range(
            1,
            MAX_RETRIES + 2
        ):

            try:

                result = ask(
                    question
                )

                break


            except Exception as e:

                last_error = e

                print(
                    f"  Attempt {attempt} failed: {e}"
                )


                if attempt < MAX_RETRIES + 1:

                    time.sleep(5)


        # ---------------------------------------------
        # Question completely failed
        # ---------------------------------------------

        if result is None:

            print(
                f"  Skipping question after "
                f"{MAX_RETRIES + 1} attempts."
            )


            with open(
                failed_log_path,
                "a",
                encoding="utf-8"
            ) as fail_f:

                fail_f.write(
                    json.dumps(
                        {
                            "batch": batch_number,
                            "question": question,
                            "error": str(last_error),
                            "traceback": traceback.format_exc()
                        }
                    )
                    + "\n"
                )


            time.sleep(
                QUESTION_SLEEP
            )

            continue


        # ---------------------------------------------
        # Rate-limit protection
        # ---------------------------------------------

        time.sleep(
            QUESTION_SLEEP
        )


        # ---------------------------------------------
        # Extract contexts
        # ---------------------------------------------

        try:

            contexts = [
                chunk.page_content
                for chunk in result[
                    "retrieved_chunks"
                ]
            ]

        except Exception as e:

            print(
                f"  Warning: couldn't extract "
                f"contexts: {e}"
            )

            contexts = []


        # ---------------------------------------------
        # Clean answer
        # ---------------------------------------------

        raw_answer = result.get(
            "answer",
            ""
        )

        cleaned_answer = clean_answer(
            raw_answer
        )


        print(
            f"  Answer length: "
            f"{len(cleaned_answer)} characters"
        )

        print(
            f"  Contexts: "
            f"{len(contexts)}"
        )


        samples.append(
            {
                "question": question,

                "answer": cleaned_answer,

                "contexts": contexts,

                "ground_truth": item["answer"]
            }
        )


    # =================================================
    # NO SUCCESSFUL QUESTIONS
    # =================================================

    if not samples:

        print(
            f"\nBatch {batch_number} had "
            f"no successful answers."
        )

        continue


    # =================================================
    # DATASET
    # =================================================

    dataset = Dataset.from_list(
        samples
    )


    # =================================================
    # RAGAS
    # =================================================

    print(
        f"\nRunning RAGAS "
        f"for Batch {batch_number}"
    )


    run_config = RunConfig(
        max_workers=1,
        timeout=300,
        max_retries=5
    )


    try:

        scores = evaluate(
            dataset,

            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ],

            llm=evaluator_llm,

            embeddings=evaluator_embedding,

            run_config=run_config
        )


    except Exception as e:

        print(
            f"\nRAGAS evaluation failed "
            f"for Batch {batch_number}:"
        )

        print(
            str(e)
        )


        with open(
            failed_log_path,
            "a",
            encoding="utf-8"
        ) as fail_f:

            fail_f.write(
                json.dumps(
                    {
                        "batch": batch_number,
                        "stage": "ragas_evaluate",
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
                )
                + "\n"
            )


        continue


    # =================================================
    # SAVE RESULTS
    # =================================================

    scores_df = scores.to_pandas()


    scores_df.to_csv(
        output_file,
        index=False
    )


    print(
        f"\nSaved: {output_file}"
    )


    # =================================================
    # SHOW BATCH SUMMARY
    # =================================================

    print(
        "\nBatch scores:"
    )

    print(
        scores_df[
            [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall"
            ]
        ].mean(
            numeric_only=True
        )
    )


    # =================================================
    # COOLDOWN
    # =================================================

    if end < total_questions:

        print(
            f"\nWaiting "
            f"{BREAK_TIME} seconds..."
        )

        time.sleep(
            BREAK_TIME
        )


# =====================================================
# COMPLETE
# =====================================================

print(
    "\n================================="
)

print(
    "All batches completed!"
)

print(
    f"Results saved in:"
)

print(
    output_dir
)

if failed_log_path.exists():

    print(
        f"\nFailed items:"
    )

    print(
        failed_log_path
    )

print(
    "================================="
)