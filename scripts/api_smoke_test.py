#!/usr/bin/env python3
"""Import/API smoke test: verify the llm_verifier package imports and
basic API functions work with the local backend."""

import os
import sys
import math

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# Test imports
from llm_verifier import select, compare, VerifierResult
from llm_verifier.fine_grained_reward import (
    create_client,
    load_dotenv,
    DEFAULT_MODEL,
    GRANULARITY,
    SCALE,
)
from llm_verifier.prompts import load_prompts, normalize_criteria

print("✓ All imports successful")

# Test that .env is loaded and client can be created
load_dotenv(ROOT_DIR)
client = create_client()
print(f"✓ Client created: {type(client).__name__}")

# Test a simple compare call with minimal criteria
criteria = [
    {"id": "correctness", "name": "Correctness", "description": "Does the code solve the task?"}
]

problem = "Write a function that returns 42."
trace_a = "def foo():\n    return 42"
trace_b = "def foo():\n    return 0"

print("Running compare()...")
r_a, r_b = compare(
    problem=problem,
    trace_a=trace_a,
    trace_b=trace_b,
    criteria=criteria,
    n_evaluations=1,
    max_workers=1,
)

print(f"✓ compare() returned: R_A={r_a:.3f}, R_B={r_b:.3f}")
assert math.isfinite(r_a) and math.isfinite(r_b)
assert 0.0 <= r_a <= 1.0 and 0.0 <= r_b <= 1.0
assert r_a > r_b, f"good candidate scored no higher: {r_a} <= {r_b}"

# Test select with 2 candidates
print("Running select()...")
result = select(
    problem=problem,
    candidates=[trace_a, trace_b],
    criteria=criteria,
    n_evaluations=1,
    pivots=1,
    max_workers=1,
)

print(f"✓ select() returned: index={result.index}, best={result.best[:20]!r}...")
print(f"  scores={result.scores}, comparisons={result.n_comparisons}")
assert result.index == 0, f"selected candidate {result.index}, expected good candidate 0"
assert result.best == trace_a, "select() did not return the good candidate"
assert len(result.scores) == 2
assert all(math.isfinite(score) and 0.0 <= score <= 1.0 for score in result.scores)

print("\nAll API smoke tests PASSED")
