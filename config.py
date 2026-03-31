import os
from dotenv import load_dotenv  # type: ignore

load_dotenv()

ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
TOGETHER_KEY = os.getenv("TOGETHER_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_API_KEY")

# Active tier: 0 = local/Ollama, 1 = cheap cloud (Groq), 2 = production
MODEL_TIER = int(os.getenv("MODEL_TIER", "0"))

# Cost controls (apply at all tiers)
MAX_TOKENS_EXECUTOR = 4000
MAX_TOKENS_VERIFIER = 2000
MAX_TOKENS_SUPERVISOR = 1000
