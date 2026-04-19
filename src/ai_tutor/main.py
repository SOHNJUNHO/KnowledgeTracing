import asyncio
import csv
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables before any package imports that read them
load_dotenv()

import torch  # noqa: E402

from ai_tutor.agents.graph import run_tutor  # noqa: E402

_DATA_PATH = Path(__file__).parents[3] / "data" / "test.csv"
_BLOCK_SIZE = 512


def _load_students(path: Path) -> tuple[dict[str, torch.Tensor], dict[int, str]]:
    """Read CSV and return per-student sequences and skill_id→name mapping."""
    rows: dict[str, list] = {}
    skill_id_to_name: dict[int, str] = {}

    with open(path) as f:
        for row in csv.DictReader(f):
            uid = row["user_id"]
            sid = int(row["skill_id"])
            correct = int(row["correct"])
            kc_id = row["kc_id"]

            rows.setdefault(uid, []).append([sid, correct])
            skill_id_to_name[sid] = kc_id

    sequences: dict[str, torch.Tensor] = {}
    for uid, seq in rows.items():
        t = torch.tensor(seq[:_BLOCK_SIZE], dtype=torch.float32)
        sequences[uid] = t

    return sequences, skill_id_to_name


async def main() -> None:
    print("Agentic GraphRAG Tutor Initialized.")

    sequences, skill_id_to_name = _load_students(_DATA_PATH)
    print(f"Loaded {len(sequences)} students from {_DATA_PATH.name}\n")

    for student_id, seq in sequences.items():
        # seq: (T, 2) → obs: (1, T-1, 2), output: (1, T-1, 2)
        obs = seq[:-1].unsqueeze(0)
        output = seq[1:].unsqueeze(0)

        state = {
            "student_id":       student_id,
            "obs":              obs,
            "output":           output,
            "skill_id_to_name": skill_id_to_name,
        }

        print(f"Running pipeline for student: {student_id} ({len(seq)} interactions)")
        new_state = await run_tutor(state)

        print("\n" + "=" * 50)
        print(f"FEEDBACK FOR {student_id}")
        print("=" * 50)
        for item in new_state.get("feedback", []):
            print(f"\n개념 (Skill): {item['kc_name']} | 수준 (Level): {item['proficiency_level']}")
            print(f"  분석: {item['reasoning']}")
            print(f"  추천: {item['feedback']}")
        print()


def cli() -> None:
    asyncio.run(main())
