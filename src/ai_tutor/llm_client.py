"""
LLM client factory.

Reads:
    LLM_BASE_URL — Ollama (or any OpenAI-compatible) base URL
                   default: http://localhost:11434/v1
    LLM_MODEL    — model name string
                   default: qwen3:8b
"""

import os
from typing import cast

from openai import AsyncOpenAI as _RawAsyncOpenAI


def get_llm_client() -> _RawAsyncOpenAI:
    from langfuse.openai import AsyncOpenAI
    return cast(_RawAsyncOpenAI, AsyncOpenAI(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        api_key="ollama",
    ))


def get_llm_model() -> str:
    return os.getenv("LLM_MODEL", "qwen3:8b")
