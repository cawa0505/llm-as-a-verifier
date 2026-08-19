#!/usr/bin/env python3
"""Phase E: backend timeout → verifier 必須非零 exit，不假成功。

以真實 openai client 對一個接受連線但永不回應的本地 hang server 發出
chat completion，client timeout=3s。預期 openai.APITimeoutError 被拋出
（未被吞掉 → 呼叫端會以非零 exit 結束）。

註：llm_verifier CLI 路徑的 client 使用 openai 預設 600s timeout；HTTP
server 路徑才有 VERIFIER_BACKEND_TIMEOUT（預設 120）。本測試以短 timeout
快速演練同一條失敗路徑。

用法:
    python scripts/test_smoke_timeout.py
"""

import socket
import sys
import threading

HANG_PORT = 18923


def _hang_server():
    """Accept one connection, read the request, never respond."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", HANG_PORT))
        srv.listen(1)
        conn, _ = srv.accept()
        try:
            conn.settimeout(30)
            conn.recv(65536)  # swallow the request; never reply
            while True:
                import time
                time.sleep(3600)
        finally:
            conn.close()


thread = threading.Thread(target=_hang_server, daemon=True)
thread.start()

from openai import OpenAI  # real client, no mock

client = OpenAI(
    base_url=f"http://127.0.0.1:{HANG_PORT}/v1",
    api_key="EMPTY",
    timeout=3.0,
    max_retries=0,
)

try:
    client.chat.completions.create(
        model="qwen3.5-9b",
        messages=[{"role": "user", "content": "Say hello."}],
        max_tokens=16,
        logprobs=True,
    )
except Exception as exc:
    if exc.__class__.__name__ == "APITimeoutError":
        print(f"PASS: backend timeout raised APITimeoutError as expected: {exc}")
        sys.exit(0)
    print(f"FAIL: unexpected exception type {exc.__class__.__name__}: {exc}")
    sys.exit(1)

print("FAIL: request succeeded against a hanging backend (no timeout)")
sys.exit(1)
