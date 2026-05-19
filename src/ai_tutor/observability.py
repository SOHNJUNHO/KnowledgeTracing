def setup_langfuse() -> None:
    """Wire LlamaIndex auto-instrumentation into the Langfuse OTel pipeline.

    No-op when LANGFUSE_PUBLIC_KEY is absent so tests run without network access.
    """
    import os
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        return
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    LlamaIndexInstrumentor().instrument()
