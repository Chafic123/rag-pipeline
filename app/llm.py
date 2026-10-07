"""One shared OpenAI client for the whole project.

Created lazily on first use, so importing the app (for example in tests) does
not need an API key, and every module reuses one client and its connection
pool instead of each file creating its own.
"""
from functools import lru_cache

from openai import OpenAI

from app.config import settings


@lru_cache(maxsize=1)
def get_openai() -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env (see .env.example).")
    return OpenAI(api_key=settings.openai_api_key)
