"""
LLM client factory.

Reads:
    LLM_BACKEND   — "openai" | "vllm"  (required)
    LLM_MODEL     — model name string   (required)
    VLLM_BASE_URL — vLLM server URL     (required when LLM_BACKEND=vllm)

openai path: returns langfuse.openai.AsyncOpenAI so traces flow to Langfuse.
vllm path:   returns openai.AsyncOpenAI pointed at VLLM_BASE_URL;
             Langfuse has no vLLM wrapper so tracing is skipped on this path.
"""

import os

from openai import AsyncOpenAI as _RawAsyncOpenAI


def get_llm_client() -> _RawAsyncOpenAI:
    backend = os.environ["LLM_BACKEND"]
    if backend == "openai":
        from langfuse.openai import AsyncOpenAI
        return AsyncOpenAI()
    if backend == "vllm":
        base_url = os.environ["VLLM_BASE_URL"]
        return _RawAsyncOpenAI(base_url=base_url, api_key="vllm")
    raise ValueError(f"Unknown LLM_BACKEND={backend!r}. Must be 'openai' or 'vllm'.")


def get_llm_model() -> str:
    return os.environ["LLM_MODEL"]
