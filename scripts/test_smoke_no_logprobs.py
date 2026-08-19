#!/usr/bin/env python3
"""Verify the public smoke test rejects a response without logprobs."""

import os
import runpy
import sys
import types
from pathlib import Path


class FakeCompletions:
    @staticmethod
    def create(**_kwargs):
        message = types.SimpleNamespace(content="hello")
        choice = types.SimpleNamespace(message=message, logprobs=None)
        return types.SimpleNamespace(choices=[choice])


class FakeOpenAI:
    def __init__(self, **_kwargs):
        self.chat = types.SimpleNamespace(completions=FakeCompletions())


fake_openai = types.ModuleType("openai")
setattr(fake_openai, "OpenAI", FakeOpenAI)
sys.modules["openai"] = fake_openai
os.environ["OPENAI_BASE_URL"] = "http://mock.invalid/v1"

repo = os.environ.get("DOCKER_VERIFIER_REPO")
assert repo, "set DOCKER_VERIFIER_REPO to the docker-llm-as-a-verifier checkout"
script = Path(repo) / "scripts" / "smoke_test.py"
assert script.exists(), f"missing public script: {script}"
try:
    runpy.run_path(str(script), run_name="__main__")
except SystemExit as exc:
    assert exc.code == 1, f"expected exit 1, got {exc.code}"
else:
    raise AssertionError("smoke test accepted a response without logprobs")

print("PASS: response without logprobs was rejected")
