import asyncio
from dotenv import load_dotenv

# Load environment variables before any package imports that read them
load_dotenv()

import torch
from langfuse import Langfuse
from ai_tutor.agents.graph import run_tutor


async def main() -> None:
    print("Agentic GraphRAG Tutor Initialized.")

    # Mock student interaction history — replace with real DB lookup in production
    mock_obs = torch.tensor([[[21.,  0.],
                               [22.,  0.],
                               [21.,  0.],
                               [17.,  1.],
                               [19.,  1.],
                               [18.,  1.],
                               [16.,  0.],
                               [17.,  1.]]])
    mock_output = torch.tensor([[[22.,  0.],
                                  [21.,  0.],
                                  [17.,  1.],
                                  [19.,  1.],
                                  [18.,  1.],
                                  [16.,  0.],
                                  [17.,  1.],
                                  [19.,  0.]]])

    state = {
        "student_id": "test_student_01",
        "obs": mock_obs,
        "output": mock_output,
        "skill_id_to_name": {
            22: "유한소수 및 무한소수",
            16: "유한소수와 무한소수의 판별",
            19: "순환소수",
            17: "순환소수의 표현",
            21: "유리수",
            18: "유리수의 분류",
        },
    }

    print("Running BKT & GraphRAG pipeline...")
    new_state = await run_tutor(state)

    print("\n" + "=" * 50)
    print("FINAL TUTOR FEEDBACK")
    print("=" * 50)

    for item in new_state.get("feedback", []):
        print(f"\n개념 (Skill): {item['kc_name']} | 수준 (Level): {item['proficiency_level']}")
        print(f"  분석: {item['reasoning']}")
        print(f"  추천: {item['feedback']}")


def cli() -> None:
    asyncio.run(main())
    Langfuse().flush()  # block until all traces are uploaded before the process exits
