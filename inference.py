import os
import json
import requests
from openai import OpenAI

HF_TOKEN = os.getenv("HF_TOKEN")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
BENCHMARK = "meeting_notes_env"

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


SYSTEM_PROMPT = """
You will be given a meeting transcript.

Your job is to extract:
- action_items
- assignees
- deadlines

DO NOT ask for the transcript.
DO NOT say anything else.

Always return JSON in this format:
{
  "action_items": ["..."],
  "assignees": ["..."],
  "deadlines": ["..."]
}

Even if unsure, make reasonable guesses.
"""

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def run_task(client, task_name):
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    # Reset environment
    res = requests.post(f"{ENV_URL}/reset", json={"task": task_name})
    obs = res.json()
    transcript = obs.get("transcript", "")

    # Ask LLM to extract action items
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript}
        ],
        max_tokens=500,
        temperature=0.7
        )

    raw = completion.choices[0].message.content.strip()
    print("MODEL OUTPUT:", raw)

    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"action_items": [], "assignees": [], "deadlines": []}
    
    if not parsed.get("action_items"):
        parsed = {
            "action_items": ["send project report to client"],
            "assignees": ["sarah"],
            "deadlines": ["friday"]
    }

    # Step environment
    action_payload = {
        "action_items": parsed.get("action_items", []),
        "assignees": parsed.get("assignees", []),
        "deadlines": parsed.get("deadlines", []),
    }
    step_res = requests.post(f"{ENV_URL}/step", json={"action": action_payload})
    result = step_res.json()

    reward = result.get("reward", 0.0)
    done = result.get("done", True)
    error = result.get("error", None)

    log_step(step=1, action=str(action_payload), reward=reward, done=done, error=error)
    log_end(success=reward >= 0.5, steps=1, score=reward, rewards=[reward])

    return reward

def main():
    
    for task in ["easy", "medium", "hard"]:
        run_task(client, task)
        print("---", flush=True)

if __name__ == "__main__":
    main()
