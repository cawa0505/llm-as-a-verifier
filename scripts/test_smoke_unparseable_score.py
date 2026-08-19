#!/usr/bin/env python3
"""Phase E: verifier score 無法解析 → public verifier_smoke_test.py 必須非零 exit。

Mock backend 回傳合法 token-level logprobs，但 content 完全不含 <score_A> /
<score_B> 標籤。extract_score 會退回中性值 0.5，compare() 回傳 (0.5, 0.5)，
public verifier_smoke_test.py 的 `good_score > bad_score` assertion 必須失敗：
非零 exit、不輸出 PASS、不列印虛構分數。

用法:
    python scripts/test_smoke_unparseable_score.py
"""

import os
import runpy
import sys
import types
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))  # llm_verifier importable (host run)

repo = os.environ.get("DOCKER_VERIFIER_REPO")
assert repo, "set DOCKER_VERIFIER_REPO to the docker-llm-as-a-verifier checkout"
PUBLIC_SCRIPT = Path(repo) / "scripts" / "verifier_smoke_test.py"
assert PUBLIC_SCRIPT.exists(), f"missing public script: {PUBLIC_SCRIPT}"


class FakeToken:
    def __init__(self, token):
        self.token = token
        self.logprob = -1.0
        self.top_logprobs = [types.SimpleNamespace(token=token, logprob=-1.0)]


class FakeCompletions:
    @staticmethod
    def create(**_kwargs):
        # llama.cpp 不支援 prefill（continue_final_message）→ 拋例外，
        # call_openai 退回 (text, None, None) → extract_score 0.5 fallback。
        if _kwargs.get("extra_body", {}).get("continue_final_message"):
            raise RuntimeError("prefill not supported (llama.cpp)")
        # Valid logprobs, but the text never emits <score_A>/<score_B> tags.
        message = types.SimpleNamespace(content="I cannot evaluate this response.")
        logprobs = types.SimpleNamespace(content=[FakeToken(t) for t in
                                                  ("I", " cannot", " evaluate")])
        choice = types.SimpleNamespace(message=message, logprobs=logprobs)
        return types.SimpleNamespace(choices=[choice])


class FakeModels:
    @staticmethod
    def list():
        return [types.SimpleNamespace(id="mock-model")]


class FakeChat:
    completions = FakeCompletions()


class FakeOpenAI:
    def __init__(self, **_kwargs):
        self.chat = FakeChat()
        self.models = FakeModels()
        self._llm_verifier_model = "mock-model"


fake_openai = types.ModuleType("openai")
setattr(fake_openai, "OpenAI", FakeOpenAI)
sys.modules["openai"] = fake_openai

# No real backend needed for the mock path.
os.environ["OPENAI_BASE_URL"] = "http://mock.invalid/v1"

try:
    runpy.run_path(str(PUBLIC_SCRIPT), run_name="__main__")
except SystemExit as exc:
    print(f"verifier_smoke_test exited with code {exc.code} (expected non-zero)")
    sys.exit(0 if exc.code not in (0, None) else 1)
except AssertionError as exc:
    print(f"AssertionError raised as expected: {exc}")
    sys.exit(0)

# Script completed normally -> it printed PASS on an unparseable score.
print("FAIL: verifier_smoke_test PASSED despite an unparseable score")
sys.exit(1)
