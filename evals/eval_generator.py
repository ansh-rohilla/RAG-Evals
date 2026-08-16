"""
evals/eval_generator.py
=======================
Component-level evaluation of the GENERATOR, in isolation.

Faithfulness:
    Checks whether claims in the generated answer are supported
    by the provided golden context.

Answer Relevancy:
    Checks whether the generated answer is relevant to the query.

Isolation:
    The generator receives the GOLDEN context directly, not the
    retriever's output.

Run:
    python -m evals.eval_generator
"""

import json
import os

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)
from deepeval.models import GeminiModel

from src.generator import generate


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

# DeepEval's Gemini integration expects a Google API key.
# Support both names in case your existing .env uses GEMINI_API_KEY.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "Google Gemini API key not found.\n"
        "Add one of these to your .env file:\n\n"
        "GOOGLE_API_KEY=your_key_here\n"
        "or\n"
        "GEMINI_API_KEY=your_key_here"
    )


# ============================================================
# 2. CONFIGURATION
# ============================================================

GOLDEN_PATH = "goldens/faithfulness_dataset.json"

JUDGE_MODEL = "gemini-3.5-flash-lite"

THRESHOLD = 0.7


# ============================================================
# 3. CREATE GEMINI JUDGE MODEL
# ============================================================

# IMPORTANT:
# Do NOT pass "gemini-3.5-flash-lite" directly as model=...
#
# We explicitly create a GeminiModel so DeepEval knows that
# Gemini, rather than OpenAI, is the evaluation provider.

judge_model = GeminiModel(
    model=JUDGE_MODEL,
    api_key=GOOGLE_API_KEY,
    temperature=0,
)


# ============================================================
# 4. LOAD GOLDEN DATASET
# ============================================================

with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
    goldens = json.load(f)


print(f"\nLoaded {len(goldens)} golden test cases.")
print(f"Judge model: {JUDGE_MODEL}\n")


# ============================================================
# 5. RUN GENERATOR ON GOLDEN CONTEXT
# ============================================================

test_cases = []

for i, g in enumerate(goldens, start=1):

    query = g["query"]

    # Golden / known-good context
    context = g["ideal_context"]

    # Run your generator
    answer = generate(query, context)

    # Create DeepEval test case
    test_cases.append(
        LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=context,
        )
    )

    print(f"Generated test case {i}/{len(goldens)}")


# ============================================================
# 6. DEFINE EVALUATION METRICS
# ============================================================

metrics = [
    FaithfulnessMetric(
        threshold=THRESHOLD,
        model=judge_model,
        include_reason=True,
    ),

    AnswerRelevancyMetric(
        threshold=THRESHOLD,
        model=judge_model,
        include_reason=True,
    ),
]


# ============================================================
# 7. RUN EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("STARTING GENERATOR EVALUATION")
print("=" * 60 + "\n")

evaluate(
    test_cases=test_cases,
    metrics=metrics,
)


print("\n" + "=" * 60)
print("GENERATOR EVALUATION COMPLETE")
print("=" * 60)