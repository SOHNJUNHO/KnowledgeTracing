from dotenv import load_dotenv
# Load environment variables FIRST
load_dotenv()

import torch
from agents.graph import run_tutor

def main():
    print("🤖 Agentic GraphRAG Tutor Initialized. Type 'quit' to exit.")
    
    # 1. Mock a student's history tensor (just for testing the local CLI)
    # In a real scenario, this comes from your database
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
    
    # 2. Setup the initial state
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
            18: "유리수의 분류"
        },
    }
    
    # 3. Execute the Analytical Pipeline
    print("🧠 Thinking (Running BKT & GraphRAG pipeline)...")
    
    # Trigger the wrapper function so Langfuse traces the whole session
    new_state = run_tutor(state)
    
    # 4. Print the AI's final output from the 'feedback' key
    print("\n" + "="*50)
    print("🎓 FINAL TUTOR FEEDBACK")
    print("="*50)
    
    for item in new_state.get('feedback', []):
        print(f"\n🔹 개념 (Skill): {item['kc_name']} | 수준 (Level): {item['proficiency_level']}")
        print(f"   💡 분석 (Analysis): {item['reasoning']}")
        print(f"   🎯 추천 (Recommendation): {item['feedback']}")

if __name__ == "__main__":
    main()