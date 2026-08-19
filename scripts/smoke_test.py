#!/usr/bin/env python3
"""Smoke test: verify the local verifier backend is reachable and returns
logprobs. Uses OPENAI_BASE_URL / MODEL_ALIAS from .env (never committed).

Usage:
    python scripts/smoke_test.py
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from llm_verifier.fine_grained_reward import (
    create_client,
    load_dotenv,
    resolve_model,
)

load_dotenv(ROOT_DIR)

base_url = os.environ.get("OPENAI_BASE_URL")
model_alias = os.environ.get("MODEL_ALIAS", "qwen3.5-9b")

if not base_url:
    print("FAIL: OPENAI_BASE_URL not set in .env or environment")
    sys.exit(1)

print(f"Backend: {base_url}")
print(f"Model alias: {model_alias}")

client = create_client()
model = resolve_model(client, model_alias)
print(f"Resolved model: {model}")

# Minimal chat call with logprobs to confirm the backend works end-to-end.
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Say hello."}],
    max_tokens=16,
    temperature=0.0,
    logprobs=True,
    top_logprobs=5,
)

text = response.choices[0].message.content
has_logprobs = bool(response.choices[0].logprobs and response.choices[0].logprobs.content)

print(f"Response: {text!r}")
print(f"Logprobs returned: {has_logprobs}")

if not has_logprobs:
    print("FAIL: backend did not return token-level logprobs")
    sys.exit(1)

print("PASS: backend reachable, logprobs present")
