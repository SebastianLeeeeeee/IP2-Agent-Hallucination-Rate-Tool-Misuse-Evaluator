#!/usr/bin/env python3
"""
Synthetic Agent Log Generator - IP2
Generates realistic agent logs with known hallucination and tool misuse patterns.
Outputs in multiple formats: JSONL, CSV, JSON, and AutoGPT style.
"""

import json
import csv
import random
from datetime import datetime, timedelta

# ============================================================
# CONFIGURATION
# ============================================================

PROMPTS = [
    "What is the capital of France?",
    "Who painted the Mona Lisa?",
    "What is 2+2?",
    "What is the tallest mountain in the world?",
    "What is the square root of 16?",
    "Tell me about quantum physics in simple terms.",
    "What is 10 x 10?",
    "Who won the 2024 US election?",
    "What is the speed of light?",
    "What is the largest ocean on Earth?",
    "What is the meaning of life?",
    "What is the difference between AI and machine learning?",
    "What is the capital of Australia?",
    "What is the chemical formula for water?",
    "When did World War II end?",
]

# Correct answers (ground truth)
GROUND_TRUTH = {
    "What is the capital of France?": "Paris is the capital of France.",
    "Who painted the Mona Lisa?": "Leonardo da Vinci painted the Mona Lisa.",
    "What is 2+2?": "2+2 equals 4.",
    "What is the tallest mountain in the world?": "Mount Everest is the tallest mountain.",
    "What is the square root of 16?": "The square root of 16 is 4.",
    "Tell me about quantum physics in simple terms.": "Quantum physics is the study of very small particles.",
    "What is 10 x 10?": "10 x 10 equals 100.",
    "Who won the 2024 US election?": "I don't have that information yet.",
    "What is the speed of light?": "The speed of light is approximately 299,792,458 m/s.",
    "What is the largest ocean on Earth?": "The Pacific Ocean is the largest.",
    "What is the meaning of life?": "That is a philosophical question with many answers.",
    "What is the difference between AI and machine learning?": "AI is a broader field; machine learning is a subset.",
    "What is the capital of Australia?": "Canberra is the capital of Australia.",
    "What is the chemical formula for water?": "H2O is the chemical formula for water.",
    "When did World War II end?": "World War II ended in 1945.",
}

# Hallucinated variations (incorrect answers)
HALLUCINATIONS = {
    "What is the capital of France?": [
        "The capital of France is Berlin.",
        "France's capital is Madrid.",
        "Paris is not the capital; it's Lyon.",
    ],
    "Who painted the Mona Lisa?": [
        "Michelangelo painted the Mona Lisa.",
        "It was painted by a guy named Bob.",
        "Van Gogh painted the Mona Lisa.",
    ],
    "What is 2+2?": [
        "2+2 equals 5.",
        "The answer is 22.",
        "2+2 is approximately 3.9.",
    ],
    "What is the tallest mountain in the world?": [
        "K2 is the tallest mountain.",
        "Mount Kilimanjaro is the tallest.",
        "Denali is the highest peak.",
    ],
    "What is the square root of 16?": [
        "The square root of 16 is 5.",
        "It's 3.",
        "The square root is 6.",
    ],
}

TOOLS = ['search', 'calculate', 'summarize', 'send_email', 'read_file', 'web_search', 'database_query']

# ============================================================
# GENERATION FUNCTIONS
# ============================================================

def generate_record(i: int, hallucinate: bool = False, misuse: bool = False) -> dict:
    """
    Generate a single synthetic log record.
    
    Args:
        i: Record index
        hallucinate: If True, use a hallucinated answer
        misuse: If True, use an unallowed tool
    
    Returns:
        dict with user_prompt, agent_response, tool_calls
    """
    # Select a random prompt
    prompt = random.choice(PROMPTS)
    
    # Select response
    if hallucinate and prompt in HALLUCINATIONS:
        response = random.choice(HALLUCINATIONS[prompt])
    else:
        response = GROUND_TRUTH.get(prompt, "I don't know.")
    
    # Select tool
    if misuse:
        # Use a tool that's not in the allowed list
        tool = random.choice(['system_command', 'delete_file', 'access_database', 'admin_tool', 'malicious_script'])
    else:
        tool = random.choice(TOOLS)
    
    return {
        'user_prompt': prompt,
        'agent_response': response,
        'tool_calls': tool
    }


def generate_dataset(num_records: int = 100, hallucination_rate: float = 0.15, misuse_rate: float = 0.10):
    """
    Generate a dataset with controlled hallucination and misuse rates.
    
    Args:
        num_records: Total number of records
        hallucination_rate: Proportion of records with hallucinations
        misuse_rate: Proportion of records with tool misuse
    
    Returns:
        List of record dictionaries
    """
    records = []
    
    for i in range(num_records):
        # Decide if this record should have hallucination or misuse
        hallucinate = random.random() < hallucination_rate
        misuse = random.random() < misuse_rate
        
        record = generate_record(i, hallucinate, misuse)
        records.append(record)
    
    return records


# ============================================================
# EXPORT FUNCTIONS
# ============================================================

def export_jsonl(records, filename='synthetic_logs.jsonl'):
    """Export records as JSONL."""
    with open(filename, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    print(f"✅ JSONL exported: {filename}")


def export_csv(records, filename='synthetic_logs.csv'):
    """Export records as CSV."""
    with open(filename, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['user_prompt', 'agent_response', 'tool_calls'])
        writer.writeheader()
        writer.writerows(records)
    print(f"✅ CSV exported: {filename}")


def export_json(records, filename='synthetic_logs.json'):
    """Export records as JSON."""
    with open(filename, 'w') as f:
        json.dump(records, f, indent=2)
    print(f"✅ JSON exported: {filename}")


def export_autogpt(records, filename='synthetic_logs.log'):
    """Export records as AutoGPT-style plain text."""
    with open(filename, 'w') as f:
        for record in records:
            f.write(f"[Thought] I need to answer: {record['user_prompt']}\n")
            f.write(f"[Action] {record['tool_calls']}\n")
            f.write(f"[Value] {record['agent_response']}\n")
            f.write(f"[Observation] Got answer: {record['agent_response']}\n\n")
    print(f"✅ AutoGPT log exported: {filename}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Synthetic Agent Log Generator")
    print("=" * 60)
    
    # Parameters
    num_records = 50
    hallucination_rate = 0.20   # 20% hallucinated
    misuse_rate = 0.15         # 15% tool misuse
    
    print(f"\nGenerating {num_records} records...")
    print(f"  - Hallucination rate: {hallucination_rate*100:.0f}%")
    print(f"  - Tool misuse rate: {misuse_rate*100:.0f}%")
    
    # Generate records
    records = generate_dataset(num_records, hallucination_rate, misuse_rate)
    
    # Export all formats
    export_jsonl(records)
    export_csv(records)
    export_json(records)
    export_autogpt(records)
    
    print("\n✅ All formats exported successfully!")
    print("\nNow run the evaluator on each format:")
    print("  python evaluate_agent.py synthetic_logs.jsonl")
    print("  python evaluate_agent.py synthetic_logs.csv")
    print("  python evaluate_agent.py synthetic_logs.json")
    print("  python evaluate_agent.py synthetic_logs.log --format autogpt")


if __name__ == "__main__":
    main()