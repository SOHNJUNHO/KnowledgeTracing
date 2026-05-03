try:
    from llama_index.core.workflow import Event
except ModuleNotFoundError:  # pragma: no cover - fallback for helper/API tests
    from pydantic import BaseModel as Event  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# LlamaIndex Workflow Events — lean routing signals between pipeline steps.
# All shared data lives in ctx: Context, not in event fields.
# ---------------------------------------------------------------------------

class BKTDoneEvent(Event):
    """Emitted by run_bkt step to trigger the diagnose step."""


class DiagnosisDoneEvent(Event):
    """Emitted by diagnose step to trigger the recommend step."""
