RAG Evaluation

A structured project for understanding, implementing, and evaluating Retrieval-Augmented Generation (RAG) systems.

The project follows a progressive evaluation approach — starting with individual evaluation components and gradually moving toward complete RAG applications.

The goal is to understand how to measure RAG quality, identify failure points, and build reliable evaluation workflows rather than treating RAG as a single black-box system.

⸻

Overview

Retrieval-Augmented Generation systems typically consist of multiple stages:

                    RAG SYSTEM
                        │
        ┌───────────────┴───────────────┐
        │                               │
    Retrieval                        Generation
        │                               │
        └───────────────┬───────────────┘
                        │
                   Final Answer

A RAG system can fail at different stages:

* The retriever may return irrelevant documents.
* Relevant information may not be retrieved.
* The generator may hallucinate despite having the correct context.
* The final answer may not address the user’s question.
* Individual components may work well while the complete pipeline performs poorly.

Therefore, this project evaluates RAG systems at multiple levels.

⸻

Evaluation Strategy

The project is organized into four major evaluation levels:

┌──────────────────────────────┐
│      1. Evaluation Suite     │
│   Evaluation methodology     │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│   2. Component-Level Eval    │
│ Retriever + Generator       │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│      3. Pipeline-Level       │
│ Retrieval → Generation       │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│     4. Application-Level     │
│      Complete RAG App        │
└──────────────────────────────┘

⸻

1. Evaluation Suite

The first stage focuses on understanding the fundamentals of RAG evaluation.

An evaluation suite provides:

* Test datasets
* Golden answers
* Golden contexts
* Evaluation metrics
* Thresholds
* Evaluation scripts
* Result analysis

The purpose is to create a repeatable framework where different RAG implementations can be evaluated consistently.

Key Concepts

* Golden datasets
* Ground truth
* Evaluation metrics
* LLM-as-a-judge
* Evaluation thresholds
* Test cases
* Evaluation reports

Example

A test case may contain:

Question
   ↓
Expected Context
   ↓
Expected Answer

The RAG system can then be evaluated against these references.

⸻

2. Component-Level Evaluation

The second stage evaluates individual components of the RAG system independently.

The main components are:

            RAG
             │
       ┌─────┴─────┐
       │           │
   Retriever    Generator

Evaluating components independently helps identify where a failure originates.

⸻

2.1 Retriever Evaluation

The retriever is responsible for finding relevant information from the knowledge base.

A typical retrieval process is:

Query
  ↓
Embedding
  ↓
Vector Search
  ↓
Retrieved Chunks

Important retrieval metrics include:

Recall

Measures whether the relevant information was successfully retrieved.

Precision

Measures how much of the retrieved information is actually relevant.

Context Relevance

Measures whether the retrieved context is relevant to the query.

Context Recall

Measures whether the retrieved context contains the information necessary to answer the question.

⸻

2.2 Generator Evaluation

The generator takes the query and retrieved context and produces the final answer.

Query + Context
       ↓
    Generator
       ↓
     Answer

The generator can be evaluated using metrics such as:

Faithfulness

Measures whether the generated answer is supported by the provided context.

A faithful answer should not introduce unsupported claims.

Answer Relevancy

Measures whether the generated answer actually addresses the user’s question.

⸻

Generator Isolation

Generator evaluation should ideally isolate the generator from retrieval failures.

Instead of:

Query
 ↓
Retriever
 ↓
Retrieved Context
 ↓
Generator

the generator can be tested using a known-good golden context:

Query
 ↓
Golden Context
 ↓
Generator
 ↓
Generated Answer

This makes it possible to determine whether a poor result is caused by:

* Retrieval
* Generation

rather than both simultaneously.

⸻

3. Pipeline-Level Evaluation

After individual components are evaluated, the next stage evaluates the complete RAG pipeline.

                QUERY
                  │
                  ▼
             RETRIEVER
                  │
                  ▼
          RETRIEVED CONTEXT
                  │
                  ▼
             GENERATOR
                  │
                  ▼
             FINAL ANSWER

Unlike component-level evaluation, pipeline evaluation measures how the components work together.

A strong retriever and a strong generator do not automatically guarantee a strong RAG system.

For example:

Retriever Score     = 0.90
Generator Score     = 0.90
Pipeline Score      = 0.65

The interaction between retrieval and generation can introduce additional failures.

⸻

Pipeline Metrics

Typical pipeline-level metrics include:

* Faithfulness
* Answer Relevancy
* Context Relevancy
* Context Recall
* Answer Correctness
* Overall RAG quality

Pipeline evaluation helps answer:

“Does the complete RAG pipeline produce a reliable answer from the user’s query?”

⸻

4. Application-Level Evaluation

The final stage evaluates RAG in the context of a complete application.

Instead of evaluating isolated test cases, application-level evaluation considers the actual user experience.

                    USER
                     │
                     ▼
              RAG APPLICATION
                     │
          ┌──────────┴──────────┐
          │                     │
      Retrieval             Generation
          │                     │
          └──────────┬──────────┘
                     │
                     ▼
                FINAL ANSWER
                     │
                     ▼
                  USER

Application-level evaluation can include:

Quality

* Answer correctness
* Faithfulness
* Relevancy

Reliability

* Consistency
* Failure handling
* Robustness

Performance

* Latency
* Token usage
* Cost
* Throughput

User Experience

* Response quality
* Clarity
* Helpfulness
* Appropriate refusal behavior

⸻

Evaluation Hierarchy

The complete approach can be summarized as:

                    RAG EVALUATION
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
    RETRIEVER          GENERATOR        APPLICATION
        │                 │                 │
        └────────────┬────┴─────────────────┘
                     │
                     ▼
                PIPELINE EVAL

Or as a progression:

Evaluation Suite
       │
       ▼
Component Evaluation
       │
       ├── Retriever
       │
       └── Generator
       │
       ▼
Pipeline Evaluation
       │
       ▼
Application Evaluation

⸻

Metrics

Some of the important metrics explored in this project include:

Metric	Purpose
Faithfulness	Checks whether the answer is supported by the context
Answer Relevancy	Checks whether the answer addresses the query
Context Relevancy	Measures relevance of retrieved context
Context Recall	Measures whether required information was retrieved
Answer Correctness	Measures correctness against a reference answer
Precision	Measures quality of retrieved results
Recall	Measures whether relevant information was retrieved

Metrics are selected based on the stage being evaluated.

⸻

Tools & Technologies

The project uses a combination of Python-based RAG and evaluation tools.

Core

* Python
* Virtual environments
* JSON
* Environment variables

RAG

* Retrieval systems
* Vector databases
* Embeddings
* Large Language Models

Evaluation

* DeepEval
* LLM-as-a-Judge
* Custom evaluation datasets
* Golden datasets

Models

The evaluation framework is designed to work with modern LLM providers and can be configured for different judge models.

⸻

Project Structure

RAG Eval/
│
├── data/
│   └──                    # Source datasets
│
├── goldens/
│   ├── faithfulness_dataset.json
│   └──                    # Golden evaluation datasets
│
├── resources/
│   └──                    # Supporting resources
│
├── src/
│   ├── generator.py
│   ├── retriever.py
│   └──                    # RAG components
│
├── evals/
│   ├── eval_generator.py
│   ├── eval_retriever.py
│   └──                    # Evaluation scripts
│
├── chroma_store/
│   └──                    # Vector database
│
├── .env
├── .gitignore
├── pyproject.toml
├── README.md
└── uv.lock

The structure may evolve as additional evaluation stages are implemented.

⸻

Golden Datasets

Golden datasets provide a reference point for evaluation.

A typical golden test case may contain:

{
    "query": "Example question",
    "ideal_context": [
        "Relevant context chunk 1",
        "Relevant context chunk 2"
    ],
    "expected_output": "Expected answer"
}

Golden datasets make evaluation:

* Repeatable
* Comparable
* Automated
* Easier to debug

They also allow different versions of a RAG system to be compared using the same test cases.

⸻

LLM-as-a-Judge

Several RAG evaluation metrics require an LLM to judge the quality of another model’s output.

For example:

              Query
                │
                ▼
           RAG System
                │
                ▼
             Answer
                │
                ▼
          Judge LLM
                │
                ▼
        Score + Reason

The judge evaluates criteria such as:

* Faithfulness
* Relevancy
* Correctness
* Context quality

Using an LLM as a judge allows qualitative properties of generated answers to be evaluated automatically.

⸻

Environment Setup

Clone the repository and create a virtual environment.

git clone <repository-url>
cd RAG-Eval

Create and activate a virtual environment:

python -m venv .venv
source .venv/bin/activate

Install the project dependencies according to the project’s package configuration.

Create a .env file:

GOOGLE_API_KEY=your_api_key

Additional API keys may be required depending on the models and evaluation providers being used.

⸻

Running Evaluations

Generator Evaluation

Run the generator evaluation from the project root:

python -m evals.eval_generator

This evaluates the generator using the golden context.

⸻

Retriever Evaluation

python -m evals.eval_retriever

This evaluates the retrieval component independently.

⸻

Pipeline Evaluation

Pipeline evaluation combines:

Query
 ↓
Retriever
 ↓
Retrieved Context
 ↓
Generator
 ↓
Final Answer
 ↓
Evaluation

⸻

Debugging RAG Systems

One of the main goals of this project is to use evaluation results for debugging.

A low score should not simply be treated as a failed test.

Instead, trace the failure:

                 LOW SCORE
                     │
          ┌──────────┴──────────┐
          │                     │
     Retrieval Issue       Generation Issue
          │                     │
          ▼                     ▼
   Wrong Context            Hallucination
   Missing Context           Irrelevant Answer
   Poor Ranking              Unsupported Claims

Component-level evaluation helps identify the source before attempting to improve the complete pipeline.

⸻

Evaluation Philosophy

The project follows a simple principle:

Measure each component first, then measure the system as a whole.

Instead of treating RAG as one black box:

Query → RAG → Answer

the system is decomposed into measurable stages:

Query
  │
  ▼
Retriever ───────► Retrieval Evaluation
  │
  ▼
Context
  │
  ▼
Generator ───────► Generation Evaluation
  │
  ▼
Answer
  │
  ▼
Pipeline Evaluation
  │
  ▼
Application Evaluation

This makes evaluation more interpretable and makes it easier to determine why a RAG system performs poorly.

⸻

Goals

The main goals of this project are to understand:

* How RAG systems fail
* How to evaluate retrieval quality
* How to evaluate generated answers
* How to design reliable golden datasets
* How LLM-based evaluation works
* How to isolate component failures
* How to evaluate complete RAG pipelines
* How evaluation changes at the application level
* How to use evaluation results to improve RAG systems

⸻

Key Takeaway

RAG evaluation is not just about calculating a single score.

A reliable evaluation process should help answer:

Is the right information being retrieved?
                ↓
Is the generated answer supported by that information?
                ↓
Is the answer relevant and correct?
                ↓
Does the complete RAG pipeline work reliably?
                ↓
Does the actual application provide a good user experience?

This project explores that progression from evaluation fundamentals → components → pipelines → complete RAG applications.

⸻