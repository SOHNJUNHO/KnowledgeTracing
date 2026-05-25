import asyncio
import csv
from itertools import groupby
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables before any package imports that read them
load_dotenv()

from langfuse import get_client

CSV_PATH = Path(__file__).parent.parent.parent / "test.csv"


def csv_to_workflow_states(csv_path: Path) -> list[dict]:
    """Convert each pre-grouped user block in test.csv into one workflow state dict.

    Only the CSV→BKT-input shape conversion lives here:
    parses ints, builds the obs/output offset pair, wraps in batch dim,
    and pulls the skill name lookup from the `name` column.
    """
    name_by_skill: dict[int, str] = {}
    states: list[dict] = []

    with csv_path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for uid, rows_iter in groupby(reader, key=lambda r: r["user_id"]):
            seq: list[list[int]] = []
            for row in rows_iter:
                sid = int(row["skill_id"])
                seq.append([sid, int(row["correct"])])
                name_by_skill.setdefault(sid, row["name"])
            states.append({
                "student_id":       uid,
                "obs":              [seq[:-1]],
                "output":           [seq[1:]],
                "skill_id_to_name": {sid: name_by_skill[sid] for sid, _ in seq},
            })
    return states


async def main() -> None:
    import torch
    torch.set_num_threads(2)

    from ai_tutor.workflow.workflow import run_tutor
    from ai_tutor.tools.neo4j_tool import close_driver
    from ai_tutor.workflow.diagnosis import _get_bkt_model

    m = _get_bkt_model()
    dummy = torch.zeros(1, 2, 2)
    with torch.no_grad():
        m.infer(dummy, dummy)

    states = csv_to_workflow_states(CSV_PATH)
    print(f"Running pipeline for {len(states)} students concurrently…")

    sem = asyncio.Semaphore(4)

    async def _gated(s):
        async with sem:
            return await run_tutor(s)

    results = await asyncio.gather(
        *[_gated(s) for s in states],
        return_exceptions=True,
    )

    n_ok   = sum(1 for r in results if not isinstance(r, BaseException))
    n_fail = len(results) - n_ok
    print(f"Done. {n_ok} succeeded, {n_fail} failed. See Langfuse for per-student traces.")

    await close_driver()


def cli() -> None:
    import os
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
        LlamaIndexInstrumentor().instrument()
    asyncio.run(main())
    get_client().flush()  # block until all traces are uploaded before the process exits
