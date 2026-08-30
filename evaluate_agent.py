#!/usr/bin/env python3
"""
Agentic AI Evaluator – Multi-Format Version
Supports: JSONL, CSV, JSON, Plain Text (AutoGPT style), OTLP, SuperAGI DB export
Evaluates agent logs for Hallucination Rate and Tool Misuse Rate
Using Llama 3.1 8B (via Ollama) as the LLM judge.
"""

import argparse
import csv
import json
import sys
import re
from typing import List, Dict, Any, Optional, Tuple
from ollama import chat
import os


# ============================================================
# AGENT DETECTION & TOOL CONFIGURATION
# ============================================================

AGENT_TOOLS = {
    "agenticseek": {
        "tools": ["search", "web_search", "bash", "python", "summarize", "read_file", "write_file"],
        "signatures": [r"Selected agent:", r"Agent.*started working", r"Executing.*bash blocks", r"Executing.*python blocks"]
    },
    "autogpt": {
        "tools": ["search", "write_file", "read_file", "execute_code", "list_folder", "delete_file"],
        "signatures": [r"\[Thought\]", r"\[Action\]", r"\[Value\]", r"\[Observation\]"]
    },
    "superagi": {
        "tools": ["search", "write_file", "read_file", "execute_code", "analyze", "summarize"],
        "signatures": [r"SuperAGI", r"Performance Telemetry", r"Agent Memory"]
    },
    "generic": {
        "tools": ["search", "calculate", "summarize", "send_email", "read_file"],
        "signatures": []
    }
}

BLOCK_CHARS = "▉▁▂▃▄▅▆▇█"


def strip_ansi(text: str) -> str:
    try:
        ansi_escape = re.compile(r'\x1b\[[0-9;]*[mK]')
        return ansi_escape.sub('', text)
    except:
        return text


def is_noise_line(line: str) -> bool:
    line = line.strip()
    if not line:
        return True
    if any(c in line for c in BLOCK_CHARS):
        return True
    if 'Thinking...' in line or 'Loading' in line or 'Device set to use' in line:
        return True
    if 'P L A N' in line or 'E N D' in line or 'Updating plan' in line:
        return True
    if line.startswith('Web ->') or line.startswith('Coder ->') or line.startswith('File ->') or line.startswith('Casual ->'):
        return True
    if line.startswith('Selected agent') or line.startswith('Assigned agent'):
        return True
    if line.startswith('Executing') or line.startswith('Exited navigation'):
        return True
    if line.startswith('Search results:') or line.startswith('AI notes:') or line.startswith('**Finding:**'):
        return True
    if line.startswith('Navigation options:') or line in ['Done.', 'Success.', 'Failure.']:
        return True
    return False


def detect_agent_type(records: List[Dict[str, Any]], raw_content: str = "") -> str:
    all_text = raw_content + " " + " ".join([
        str(record.get('user_prompt', '')) + " " + 
        str(record.get('agent_response', '')) + " " +
        str(record.get('tool_calls', ''))
        for record in records
    ])
    for agent_name, config in AGENT_TOOLS.items():
        if agent_name == "generic":
            continue
        for sig in config["signatures"]:
            if sig.lower() in all_text.lower():
                return agent_name
    return "generic"


def get_tools_for_agent(agent_type: str) -> List[str]:
    return AGENT_TOOLS.get(agent_type, AGENT_TOOLS["generic"])["tools"]


# ============================================================
# JUDGE FUNCTIONS
# ============================================================

def judge_hallucination(user_prompt: str, agent_response: str) -> Tuple[bool, str]:
    if not agent_response or len(agent_response.strip()) < 5:
        return False, "SKIPPED (empty response)"
    
    judge_prompt = f"""You are a hallucination detector. Given the user's question and the AI's answer, 
determine if the answer contains any factually incorrect or made-up information.

User question: {user_prompt}
AI answer: {agent_response}

Does this answer contain a hallucination? Answer only "YES" or "NO".
"""
    try:
        response = chat(
            model='llama3.1:8b',
            messages=[{'role': 'user', 'content': judge_prompt}],
            keep_alive='5m'
        )
        verdict = response['message']['content'].strip().upper()
        return "YES" in verdict, verdict
    except Exception as e:
        return False, f"ERROR: {str(e)[:50]}"


def judge_tool_misuse(user_prompt: str, tool_called: str, allowed_tools: list) -> Tuple[bool, str]:
    if not tool_called or len(tool_called.strip()) < 2:
        return False, "SKIPPED (no tool call)"
    
    judge_prompt = f"""You are a tool usage evaluator. Given the user's question and the tool called by the AI, 
determine if the tool is appropriate and correctly used.

User question: {user_prompt}
Tool called: {tool_called}
Allowed tools: {allowed_tools}

Is this tool call inappropriate or a misuse? Answer only "YES" or "NO".
"""
    try:
        response = chat(
            model='llama3.1:8b',
            messages=[{'role': 'user', 'content': judge_prompt}],
            keep_alive='5m'
        )
        verdict = response['message']['content'].strip().upper()
        return "YES" in verdict, verdict
    except Exception as e:
        return False, f"ERROR: {str(e)[:50]}"


# ============================================================
# FILE PARSERS
# ============================================================

def parse_jsonl(file_path: str) -> List[Dict[str, Any]]:
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except:
                pass
    return records


def parse_csv(file_path: str) -> List[Dict[str, Any]]:
    records = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def parse_superagi_feeds(file_path: str) -> List[Dict[str, Any]]:
    """Parse SuperAGI agent_execution_feeds CSV export.
    Extracts the user's original goal (from system messages or agent_executions)
    and pairs it with the final assistant response.
    """
    records = []
    with open(file_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    goal = ""
    final_response = ""
    tool_calls = ""

    # First pass: find the goal from system messages containing "GOALS:"
    for row in rows:
        if row.get('role') == 'system' and 'GOALS:' in row.get('feed', ''):
            feed = row['feed']
            lines = feed.split('\n')
            for line in lines:
                if line.startswith('1.'):
                    goal = line.strip().split(' ', 1)[-1]
                    break
            if goal:
                break
    if not goal:
        # Fallback: find any user message that looks like the original question
        for row in rows:
            if row.get('role') == 'user' and 'capital of Indonesia' in row.get('feed', ''):
                goal = row['feed']
                break

    # Second pass: find the final assistant response (the one that actually answers)
    for row in reversed(rows):
        if row.get('role') == 'assistant':
            feed = row.get('feed', '')
            if 'capital of Indonesia' in feed.lower() and ('jakarta' in feed.lower() or '**' in feed):
                final_response = feed
                extra = row.get('extra_info', '')
                if extra:
                    try:
                        extra_dict = json.loads(extra)
                        tool_calls = extra_dict.get('tool_name', '')
                    except:
                        pass
                break
            if not final_response:
                final_response = feed
    if not final_response:
        for row in reversed(rows):
            if row.get('role') == 'assistant':
                final_response = row.get('feed', '')
                break

    if goal and final_response:
        records.append({
            'user_prompt': goal,
            'agent_response': final_response,
            'tool_calls': tool_calls
        })

    # Fallback to old behavior if nothing extracted
    if not records:
        i = 0
        while i < len(rows):
            if rows[i].get('role') == 'user':
                user_prompt = rows[i].get('feed', '')
                agent_response = ''
                for j in range(i+1, len(rows)):
                    if rows[j].get('role') == 'assistant':
                        agent_response = rows[j].get('feed', '')
                        i = j
                        break
                records.append({
                    'user_prompt': user_prompt,
                    'agent_response': agent_response,
                    'tool_calls': ''
                })
            i += 1

    return records


def parse_json(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def parse_autogpt_plain_text(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r') as f:
        raw = f.read()
    clean = strip_ansi(raw)
    records = []
    
    parts = clean.split('➤➤➤')
    for idx, part in enumerate(parts[1:], 1):
        lines = part.split('\n')
        if not lines:
            continue
        user_prompt = lines[0].strip()
        if user_prompt.lower() in ['exit', 'quit', '']:
            continue
        
        resp_lines = []
        for line in lines[1:]:
            line = line.strip()
            if is_noise_line(line):
                continue
            if line.startswith('➤➤➤'):
                break
            resp_lines.append(line)
        
        full = ' '.join(resp_lines)
        
        conclusion = None
        if '**Conclusion:**' in full:
            idx2 = full.find('**Conclusion:**') + len('**Conclusion:**')
            rest = full[idx2:].strip()
            end = rest.find('\n\n')
            conclusion = rest[:end] if end != -1 else rest
        
        capital = None
        lower_full = full.lower()
        if 'capital of' in lower_full:
            sentences = re.split(r'[.!?]\s*', full)
            for sent in sentences:
                if 'capital' in sent.lower():
                    capital = sent.strip()
                    break
        
        answer = conclusion or capital or (resp_lines[-1] if resp_lines else '')
        
        tool = ''
        lower_part = part.lower()
        if 'executing bash' in lower_part or ' bash ' in lower_part or lower_part.startswith('bash'):
            tool = 'bash'
        elif 'executing python' in lower_part or ' python ' in lower_part or lower_part.startswith('python'):
            tool = 'python'
        elif 'navigating to' in lower_part or 'web_search' in lower_part:
            tool = 'web_search'
        elif 'search results:' in lower_part:
            tool = 'search'
        elif 'write_file' in lower_part or 'read_file' in lower_part:
            tool = 'file'
        
        if answer and len(answer) > 3:
            records.append({
                'user_prompt': user_prompt,
                'agent_response': answer[:300] + '...' if len(answer) > 300 else answer,
                'tool_calls': tool
            })
    
    if not records:
        records.append({
            'user_prompt': '',
            'agent_response': clean[:500] + '...' if len(clean) > 500 else clean,
            'tool_calls': ''
        })
    return records


def parse_otlp(file_path: str) -> List[Dict[str, Any]]:
    return []  # Simple fallback


def is_superagi_csv(file_path: str) -> bool:
    """Check if a CSV file is a SuperAGI feed export."""
    try:
        with open(file_path, newline='') as f:
            reader = csv.reader(f)
            headers = next(reader)
            if 'role' in headers and 'feed' in headers:
                return True
    except:
        pass
    return False


def load_logs(file_path: str, format_hint: Optional[str] = None) -> Tuple[List[Dict[str, Any]], str]:
    if not format_hint:
        ext = file_path.lower().split('.')[-1] if '.' in file_path else ''
        if is_superagi_csv(file_path):
            format_hint = 'superagi'
        elif ext == 'jsonl':
            format_hint = 'jsonl'
        elif ext == 'csv':
            format_hint = 'csv'
        elif ext == 'json':
            format_hint = 'json'
        elif ext in ['log', 'txt']:
            format_hint = 'autogpt'
        else:
            format_hint = 'autogpt'
    
    raw_content = ''
    try:
        with open(file_path, 'r') as f:
            raw_content = f.read()
    except:
        pass
    
    parsers = {
        'jsonl': parse_jsonl,
        'csv': parse_csv,
        'json': parse_json,
        'autogpt': parse_autogpt_plain_text,
        'superagi': parse_superagi_feeds
    }
    parser = parsers.get(format_hint.lower(), parse_autogpt_plain_text)
    print(f"Parsing {format_hint.upper()} file: {file_path}")
    records = parser(file_path)
    return records, raw_content


def evaluate_log_file(log_file: str, output_file: str = None, format_hint: str = None):
    print(f"\n{'='*60}")
    print(f"Evaluating Agent Logs: {log_file}")
    print(f"{'='*60}\n")
    
    try:
        records, raw_content = load_logs(log_file, format_hint)
    except Exception as e:
        print(f"Error loading file: {e}", file=sys.stderr)
        return
    
    if not records:
        print("Error: No records found in log file.", file=sys.stderr)
        return
    
    agent_type = detect_agent_type(records, raw_content)
    allowed_tools = get_tools_for_agent(agent_type)
    
    print(f"Detected Agent Type: {agent_type.upper()}")
    print(f"Allowed Tools: {', '.join(allowed_tools)}\n")
    
    total = len(records)
    hallucinated_count = 0
    misuse_count = 0
    results = []
    
    for i, record in enumerate(records, 1):
        user_prompt = record.get('user_prompt', '')
        agent_response = record.get('agent_response', '')
        tool_calls = record.get('tool_calls', '')
        
        print(f"[{i}/{total}] Processing record...", end='', flush=True)
        
        is_hall, hall_v = judge_hallucination(user_prompt, agent_response)
        if is_hall:
            hallucinated_count += 1
        
        is_mis, mis_v = judge_tool_misuse(user_prompt, tool_calls, allowed_tools)
        if is_mis:
            misuse_count += 1
        
        print(" Done.")
        
        results.append({
            'user_prompt': user_prompt,
            'agent_response': agent_response[:200] + '...' if len(agent_response) > 200 else agent_response,
            'tool_calls': tool_calls,
            'hallucination': is_hall,
            'hallucination_verdict': hall_v,
            'tool_misuse': is_mis,
            'tool_misuse_verdict': mis_v
        })
    
    hallucination_rate = (hallucinated_count / total) * 100
    tool_misuse_rate = (misuse_count / total) * 100 if total > 0 else 0
    
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Agent Type:           {agent_type.upper()}")
    print(f"Total records:        {total}")
    print(f"Hallucinations:       {hallucinated_count} ({hallucination_rate:.2f}%)")
    print(f"Tool Misuse:          {misuse_count} ({tool_misuse_rate:.2f}%)")
    print(f"{'='*60}\n")
    
    try:
        show_details = input("Would you like to see individual log evaluations? (Y/N): ").strip().upper()
    except:
        show_details = 'N'
    
    if show_details == 'Y':
        print("\n" + "="*60)
        print("INDIVIDUAL RECORD EVALUATIONS")
        print("="*60 + "\n")
        
        for idx, result in enumerate(results, 1):
            print(f"--- Record {idx} ---")
            print(f"User Prompt: {result['user_prompt']}")
            print(f"Agent Response: {result['agent_response']}")
            print(f"Tool Calls: {result['tool_calls']}")
            print(f"Hallucination: {'❌ YES' if result['hallucination'] else '✅ NO'} ({result['hallucination_verdict']})")
            print(f"Tool Misuse: {'❌ YES' if result['tool_misuse'] else '✅ NO'} ({result['tool_misuse_verdict']})")
            print("-"*40)
    
    if output_file:
        with open(output_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['agent_type', agent_type])
            writer.writerow(['total_records', total])
            writer.writerow(['hallucinations_detected', hallucinated_count])
            writer.writerow(['tool_misuse_detected', misuse_count])
            writer.writerow(['hallucination_rate_percent', hallucination_rate])
            writer.writerow(['tool_misuse_rate_percent', tool_misuse_rate])
        print(f"\nResults saved to: {output_file}")


def clean_log_file(log_file: str, output_file: str = None, format_hint: str = None):
    print(f"\nCleaning log file: {log_file}")
    records, raw_content = load_logs(log_file, format_hint)
    agent_type = detect_agent_type(records, raw_content)
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump({'agent_type': agent_type, 'records': records}, f, indent=2)
        print(f"Cleaned records saved to: {output_file}")
    else:
        print(json.dumps({'agent_type': agent_type, 'records': records}, indent=2))



def list_available_files(base_dir: str = None):
    """Print only actual agent log files found under base_dir."""
    if base_dir is None:
        base_dir = os.path.expanduser("~/ip2_evaluator")
    # Directories where we know logs live
    known_dirs = [
        os.path.expanduser("~/ip2_evaluator/agenticSeek/logs"),
        os.path.expanduser("~/ip2_evaluator/SuperAGI"),
        os.path.expanduser("~/ip2_evaluator"),
    ]
    # Files we know are logs (by path)
    known_files = [
        "feeds_44.csv",
        "superagi_full_log.txt",
        "superagi_logs.txt",
        "synthetic_logs.csv",
        "synthetic_logs.json",
        "synthetic_logs.jsonl",
        "synthetic_logs.log",
    ]
    found = []
    for d in known_dirs:
        if os.path.isdir(d):
            for root, dirs, files in os.walk(d):
                # Skip deep directories that aren't log folders
                if 'node_modules' in root or '__pycache__' in root or '.git' in root:
                    continue
                for file in files:
                    full = os.path.join(root, file)
                    # Only include if it's a known log file or a file with .log extension
                    if any(full.endswith(f) for f in known_files) or file.endswith('.log'):
                        found.append(full)
    # Add specific known files from the base directory
    for f in known_files:
        full = os.path.join(base_dir, f)
        if os.path.isfile(full) and full not in found:
            found.append(full)
    found.sort()
    print("Available evaluation log files:")
    for f in found:
        print(f"  {f}")
    print(f"\nTotal: {len(found)} files")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Agentic AI logs for hallucinations and tool misuse.')
    parser.add_argument('log_file', nargs='?', default=None, help='Path to the log file (optional if --list is used)')
    parser.add_argument('--format', choices=['jsonl', 'csv', 'json', 'autogpt', 'superagi'], help='File format')
    parser.add_argument('--allowed-tools', nargs='+', help='Override allowed tools')
    parser.add_argument('--output', '-o', help='Output file for summary')
    parser.add_argument('--clean', action='store_true', help='Clean and output JSON')
    parser.add_argument('--clean-output', help='Save cleaned JSON to file')
    parser.add_argument('--list', action='store_true', help='List available log files and exit')
    args = parser.parse_args()
    
    if args.list:
        list_available_files()
        return
    
    if not args.log_file:
        parser.error("the following arguments are required: log_file")
    
    if args.clean:
        clean_log_file(args.log_file, args.clean_output, args.format)
    else:
        evaluate_log_file(args.log_file, args.output, args.format)

if __name__ == "__main__":
    main()
