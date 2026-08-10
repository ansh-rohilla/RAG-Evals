import os
import glob
import json
import random

from dotenv import load_dotenv

from deepeval.synthesizer import Synthesizer
from deepeval.models import DeepEvalBaseLLM

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. "
        "Make sure it is present in your .env file."
    )

print("Google API key loaded successfully.")


# ============================================================
# GEMINI MODEL FOR DEEPEVAL
# ============================================================

class GeminiDeepEvalLLM(DeepEvalBaseLLM):

    def __init__(self):

        self.model = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key=GOOGLE_API_KEY,
        )

    def load_model(self):
        return self.model

    def get_model_name(self):
        return "Gemini 3.5 Flash Lite"

    # --------------------------------------------------------
    # Synchronous generation
    # --------------------------------------------------------

    def generate(self, prompt: str, schema=None):

        if schema is not None:

            structured_model = self.model.with_structured_output(
                schema
            )

            return structured_model.invoke(prompt)

        response = self.model.invoke(prompt)

        return response.content

    # --------------------------------------------------------
    # Asynchronous generation
    # --------------------------------------------------------

    async def a_generate(self, prompt: str, schema=None):

        if schema is not None:

            structured_model = self.model.with_structured_output(
                schema
            )

            return await structured_model.ainvoke(prompt)

        response = await self.model.ainvoke(prompt)

        return response.content


# ============================================================
# LOAD AND CLEAN VTT FILES
# ============================================================

def load_chunks():

    texts = []

    vtt_files = glob.glob("data/*.vtt")

    if not vtt_files:
        raise FileNotFoundError(
            "No .vtt files found inside the data/ directory."
        )

    print("\nVTT files found:")

    for path in vtt_files:

        print(f"  - {path}")

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            lines = [
                line.strip()
                for line in f
                if line.strip()
                and line.strip() != "WEBVTT"
                and "-->" not in line
            ]

        # Combine subtitle lines
        texts.append(" ".join(lines))

    # ========================================================
    # TEXT CHUNKING
    # ========================================================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_text(
        "\n\n".join(texts)
    )

    return chunks


# ============================================================
# LOAD CHUNKS
# ============================================================

print("\nLoading VTT files...")

chunks = load_chunks()

print(f"Total chunks created: {len(chunks)}")


if len(chunks) == 0:
    raise ValueError(
        "No chunks were created from the VTT files."
    )


# ============================================================
# SELECT RANDOM CHUNKS
# ============================================================

# Keep this small initially to avoid API rate limits.
NUMBER_OF_CHUNKS = min(10, len(chunks))

sample = random.sample(
    chunks,
    NUMBER_OF_CHUNKS
)

print(
    f"Using {NUMBER_OF_CHUNKS} random chunks "
    "for golden generation."
)


# ============================================================
# CREATE DEEPEVAL CONTEXTS
# ============================================================

# Each context represents one document/chunk.

contexts = [
    [chunk]
    for chunk in sample
]


# ============================================================
# INITIALIZE GEMINI
# ============================================================

print("\nInitializing Gemini...")

gemini_model = GeminiDeepEvalLLM()

print(
    f"Model: {gemini_model.get_model_name()}"
)


# ============================================================
# CREATE DEEPEVAL SYNTHESIZER
# ============================================================

synthesizer = Synthesizer(
    model=gemini_model,

    # Start synchronously to avoid rate-limit problems.
    async_mode=False,

    # One request at a time.
    max_concurrent=1
)


# ============================================================
# GENERATE GOLDENS
# ============================================================

print("\nGenerating goldens...")
print("This may take some time.\n")

goldens = synthesizer.generate_goldens_from_contexts(
    contexts=contexts,

    # Generate ideal/reference answers.
    include_expected_output=True,

    # One golden per context.
    max_goldens_per_context=1
)


# ============================================================
# CONVERT TO YOUR JSON FORMAT
# ============================================================

rows = []

for i, golden in enumerate(goldens, start=1):

    rows.append(
        {
            "id": f"g{i:03d}",

            "query": golden.input,

            "ideal_answer": golden.expected_output,

            # We currently don't track which VTT
            # chunk generated the golden.
            "source": "TODO-verify"
        }
    )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    "goldens",
    exist_ok=True
)


# ============================================================
# SAVE GOLDENS
# ============================================================

output_file = (
    "goldens/retriever_deepeval_goldens.json"
)

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        rows,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)

print("GOLDEN GENERATION COMPLETE")

print("=" * 60)

print(
    f"Generated goldens : {len(rows)}"
)

print(
    f"Output file       : {output_file}"
)

print("=" * 60)

print("\nIMPORTANT: Review the generated goldens before")
print("using them for RAG evaluation.")

print("\nCheck:")
print("  1. Is the question grounded in the VTT?")
print("  2. Is the ideal answer correct?")
print("  3. Is the question actually answerable?")
print("  4. Remove unnecessary wording.")
print("  5. Fix leading/biased questions.")
print("  6. Verify the answer against the source transcript.")