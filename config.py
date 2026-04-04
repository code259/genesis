import os
from dotenv import load_dotenv  # type: ignore

load_dotenv()


def _load_groq_keys() -> list[str]:
    keys = []
    primary = os.getenv("GROQ_API_KEY")
    if primary:
        keys.append(primary)

    idx = 2
    while True:
        extra = os.getenv(f"GROQ_API_KEY_{idx}")
        if not extra:
            break
        keys.append(extra)
        idx += 1
    return keys

ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
TOGETHER_KEY = os.getenv("TOGETHER_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
GROQ_KEYS = _load_groq_keys()
GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_API_KEY")

# Active tier: 0 = local/Ollama, 1 = cheap cloud (Groq), 2 = production
MODEL_TIER = int(os.getenv("MODEL_TIER", "0"))

# Cost controls (apply at all tiers)
MAX_TOKENS_EXECUTOR = 4000
MAX_TOKENS_VERIFIER = 2000
MAX_TOKENS_SUPERVISOR = 1000
WORKER_MAX_STEPS = int(os.getenv("GENESIS_WORKER_MAX_STEPS", "6"))
MAX_PARALLEL_TASKS = int(os.getenv("GENESIS_MAX_PARALLEL_TASKS", "2"))
MAX_PARALLEL_GROQ_CALLS = int(os.getenv("GENESIS_MAX_PARALLEL_GROQ_CALLS", "2"))
RUNTIME_IMAGE_TAG = os.getenv("GENESIS_RUNTIME_IMAGE_TAG", "genesis-runtime:py39-v1")
