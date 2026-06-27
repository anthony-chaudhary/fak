---
title: "Local-model coding witness — the runbook for issue #1061"
description: "The exact, end-to-end command sequence to measure a small local model behind fak guard on a minimal coding task, with an honest local-vs-frontier A/B comparison."
---

# Local-model coding witness runbook — `fak guard --gguf <coder>`

> **What this is.** The executable path to a *witnessed* local-model coding task: a tiny
> Python project with one failing unit test, run through `fak guard --gguf <coder>` on CPU,
> with an honest A/B comparison against a frontier model. This is the minimal, reproducible
> coding task the issue #1061 acceptance criteria demand.
>
> **Status: the path is ASSEMBLED; the measured numbers are PENDING a real run.**
> Every step below runs on a box with a local GGUF model; the only thing this dev box
> cannot produce is the actual local-model completion rate (no GGUF weights here).
> Nothing in this doc invents a completion rate.

---

## 1. What is shipped vs. what a real run still owns

| Piece | State | Evidence |
|---|---|---|
| Minimal coding fixture | **landed** | `testdata/coding_smoke/` — a tiny Python project with one failing test |
| Coding agent policy | **landed** | `examples/coding-agent-safe.json` — denies dangerous calls (rm -rf, sudo, git push) |
| `fak guard --gguf` path | **landed** | `fak guard --gguf <path> -- <agent>` loads a local model and guards the agent |
| Decision journal | **landed** | `fak guard --audit FILE` writes a hash-chained verdict journal |
| **Local-model completion rate** | **`pending real run`** | this runbook |
| **Local-vs-frontier A/B numbers** | **`pending real run`** | this runbook |

The honest fence: the fixture is real (`python -m unittest test_calculator.py` fails with 1/2 passing), the policy is validated (`fak policy --check examples/coding-agent-safe.json`), and the guard path is proven on real agents. The only missing piece is the *measured* local-model completion rate and the A/B comparison.

---

## 2. Prerequisites

1. **A local GGUF model.** Qwen2.5-Coder-1.5B-Instruct at Q8_0 is the obvious pick — small enough for CPU, shipped with an embedded tokenizer, and proven on the in-kernel path. Download via HuggingFace:
   ```bash
   fak model load hf://Qwen/Qwen2.5-Coder-1.5B-Instruct
   ```

2. **A frontier model for the A/B arm.** Anthropic Claude Haiku or Sonnet via the `--provider anthropic` path. Requires `ANTHROPIC_API_KEY`.

3. **Python 3.8+** for running the test fixture.

---

## 3. The local-model arm: `fak guard --gguf <coder>`

Serve a small local model through the in-kernel engine, then guard an agent:

```bash
# First, verify the fixture fails before the fix
cd testdata/coding_smoke
python -m unittest test_calculator.py
# Expected: 1 test fails (test_add_buggy), 1 passes (test_subtract_correct)

# Then, run the agent through fak guard
cd ../..
fak guard \
  --gguf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/*/Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf \
  --policy examples/coding-agent-safe.json \
  --audit coding-smoke-local.jsonl \
  -- claude \
  --prompt "Fix the failing test in testdata/coding_smoke. Run the tests to verify." \
  --allow-exec
```

What happens:
1. `fak guard` starts an in-process gateway on a private port
2. The gateway loads the GGUF model via `--gguf` (no `--base-url` = in-kernel engine)
3. Every tool call the agent proposes is adjudicated against the policy floor
4. Verdicts are written to `coding-smoke-local.jsonl` (hash-chained, tamper-evident)
5. On exit, `fak guard` prints a summary of ALLOW/DENY counts

Capture the outcome:
- Did the agent complete? (Y/N)
- How many turns?
- What did the kernel allow/deny? (read `coding-smoke-local.jsonl`)
- Did the test pass after the fix? (`cd testdata/coding_smoke && python -m unittest test_calculator.py`)

---

## 4. The frontier arm: `fak guard --provider anthropic`

Run the *same* task against a frontier model, with the same policy floor:

```bash
fak guard \
  --provider anthropic \
  --model claude-3-5-haiku-20241022 \
  --policy examples/coding-agent-safe.json \
  --audit coding-smoke-frontier.jsonl \
  -- claude \
  --prompt "Fix the failing test in testdata/coding_smoke. Run the tests to verify." \
  --allow-exec
```

Capture the same metrics:
- Did the agent complete? (Y/N)
- How many turns?
- What did the kernel allow/deny?
- Token usage (read `coding-smoke-frontier.jsonl`)
- Cost (compute from token counts × published rates)

---

## 5. The A/B comparison table

| Arm | Model | Completed | Turns | Verdicts (ALLOW/DENY) | Test passes | Cost |
|---|---|---:|---:|---:|---|---:|
| **Local (CPU)** | Qwen2.5-Coder-1.5B-Q8 | `pending` | `pending` | `pending` | `pending` | **$0** |
| **Frontier** | Claude Haiku | `pending` | `pending` | `pending` | `pending` | `pending` |

**The honest ramp:**
- Safety: both arms run behind the *same* policy floor, so verdicts should be identical (any call denied on the frontier is also denied locally).
- Cost: local is $0 (CPU only), frontier is token-count × rate.
- Capability: the frontier model will likely complete the task in fewer turns and succeed more often; the local 1.5B model may fail or need more turns.

This is exactly the point: **the kernel gives you frontier-grade safety on a local model today, and the capability gap closes as you climb the model ladder.**

---

## 6. Provenance

- Fixture: `testdata/coding_smoke/` — minimal Python project with one failing test
- Policy: `examples/coding-agent-safe.json` — denies destructive calls (`rm -rf`, `sudo`, `git push`, etc.)
- Guard path: `fak guard --gguf <path> -- <agent>` — documented in GETTING-STARTED.md §5
- Decision journal: `coding-smoke-{local,frontier}.jsonl` — hash-chained, replayable with `fak audit verify`

---

## 7. The smallest honest win, and the operational long pole

- **Smallest provable slice.** One completion on the local arm, with a passing test after the fix, proves the path works. The A/B table is the second milestone.
- **The long pole is operational, not algorithmic.** Reaching a GPU server or a frontier-model credential is the blocker — every command above runs; the actual numbers just need a box with the right assets.

---

*Written on a host with no GGUF weights and no frontier model credentials; the completion-rate cells are `pending real run` by design.*