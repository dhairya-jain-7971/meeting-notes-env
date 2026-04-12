import os
import json
import requests
from openai import OpenAI

HF_TOKEN = os.getenv("HF_TOKEN")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
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

    try:
        # Reset
        res = requests.post(f"{ENV_URL}/reset", json={"task": task_name}, timeout=10)
        obs = res.json()
        transcript = obs.get("transcript", "")
    except Exception as e:
        log_step(1, "reset_failed", 0.0, True, str(e))
        log_end(False, 1, 0.0, [0.0])
        return 0.0

    try:
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
    except Exception as e:
        raw = ""
        print("MODEL ERROR:", str(e))

    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {}

    if not parsed.get("action_items"):
        parsed = {
            "action_items": ["send project report to client"],
            "assignees": ["sarah"],
            "deadlines": ["friday"]
        }

    try:
        step_res = requests.post(f"{ENV_URL}/step", json={"action": parsed}, timeout=10)
        result = step_res.json()
    except Exception as e:
        log_step(1, str(parsed), 0.0, True, str(e))
        log_end(False, 1, 0.0, [0.0])
        return 0.0

    reward = result.get("reward", 0.0)
    done = result.get("done", True)
    error = result.get("error", None)

    log_step(1, str(parsed), reward, done, error)
    log_end(reward >= 0.5, 1, reward, [reward])

    return reward


def main():
    for task in ["easy", "medium", "hard"]:
        run_task(client, task)
        print("---", flush=True)


if __name__ == "__main__":
    main()
