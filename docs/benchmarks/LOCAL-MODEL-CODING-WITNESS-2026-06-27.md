---
title: "Local-model coding witness — results (PENDING)"
description: "The measured results of running a small local model behind fak guard on a minimal coding task, with an honest local-vs-frontier A/B comparison. PENDING: path assembled, awaiting real run on a box with GGUF weights."
---

# Local-model coding witness — results (2026-06-27)

> **What this is.** The witness that answers issue #1061: *how far does a small local
> model + fak actually get on a real coding task?* This document records the
> exact command, the captured outcome, and the A/B comparison against a frontier
> model.
>
> **Status: PENDING — the path is proven, the numbers are not.**
> This document is the *shape* of the witness; a real run will fill in the cells
> marked `pending` below. Run the commands in
> [`LOCAL-MODEL-CODING-WITNESS-RUNBOOK.md`](LOCAL-MODEL-CODING-WITNESS-RUNBOOK.md)
> to produce the actual numbers.

---

## Quick Reference — the one-liner

```bash
# Local model (CPU)
fak guard \
  --gguf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/*/Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf \
  --policy examples/coding-agent-safe.json \
  --audit coding-smoke-local.jsonl \
  -- claude \
  --prompt "Fix the failing test in testdata/coding_smoke. Run the tests to verify." \
  --allow-exec

# Frontier model (same task)
fak guard \
  --provider anthropic \
  --model claude-3-5-haiku-20241022 \
  --policy examples/coding-agent-safe.json \
  --audit coding-smoke-frontier.jsonl \
  -- claude \
  --prompt "Fix the failing test in testdata/coding_smoke. Run the tests to verify." \
  --allow-exec
```

---

## The fixture — `testdata/coding_smoke/`

A minimal, deterministic coding task:

- **Language:** Python
- **Complexity:** `<5min` (one-line fix)
- **Files:**
  - `calculator.py` — buggy `add()` function
  - `test_calculator.py` — one failing test, one passing test
  - `README.md` — problem statement and verification steps

**Before the fix:**
```bash
cd testdata/coding_smoke
python -m unittest test_calculator.py
# Expected: F. (1 fail, 1 pass)
```

**After the fix:**
```bash
python -m unittest test_calculator.py
# Expected: .. (2 passes)
```

---

## Measured results (PENDING)

Runbook: [`LOCAL-MODEL-CODING-WITNESS-RUNBOOK.md`](LOCAL-MODEL-CODING-WITNESS-RUNBOOK.md)

### Local model (CPU)

| Metric | Value | Source |
|---|---|---|
| Model | Qwen2.5-Coder-1.5B-Instruct Q8_0 | `--gguf <path>` |
| Hardware | `pending` | Run environment |
| Completed | `pending` | Agent exit code |
| Turns | `pending` | Decision journal |
| Verdicts (ALLOW/DENY) | `pending` / `pending` | `coding-smoke-local.jsonl` |
| Test passes after fix | `pending` | `python -m unittest test_calculator.py` |
| Cost | **$0** | CPU only |
| Runtime | `pending` seconds | Wall-clock |

### Frontier model

| Metric | Value | Source |
|---|---|---|
| Model | Claude Haiku 3.5 | `--provider anthropic --model claude-3-5-haiku-20241022` |
| Completed | `pending` | Agent exit code |
| Turns | `pending` | Decision journal |
| Verdicts (ALLOW/DENY) | `pending` / `pending` | `coding-smoke-frontier.jsonl` |
| Test passes after fix | `pending` | `python -m unittest test_calculator.py` |
| Cost | `pending` USD | Token counts × rate |
| Runtime | `pending` seconds | Wall-clock |

---

## The A/B table (PENDING)

| Arm | Model | Completed | Turns | ALLOW | DENY | Test passes | Cost |
|---|---|---:|---:|---:|---:|---|---:|
| **Local (CPU)** | Qwen2.5-Coder-1.5B-Q8 | `pending` | `pending` | `pending` | `pending` | `pending` | **$0** |
| **Frontier** | Claude Haiku 3.5 | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` |

**The honest ramp:**
- **Safety:** Both arms run behind the *same* policy floor (`examples/coding-agent-safe.json`). The kernel's adjudication is model-agnostic, so the verdicts should be identical (any call denied on the frontier is also denied locally).
- **Cost:** Local is $0 (CPU only, no API fees). Frontier is token-count × published rate.
- **Capability:** The frontier model will likely complete the task in fewer turns and succeed more often. The local 1.5B model may fail, need more turns, or only partially succeed. This is the *capability ramp* the issue asks for: **the kernel gives you frontier-grade safety on a local model today, and capability improves as you climb the model ladder (0.5B → 1.5B → 7B → GPU).**

---

## Provenance & honesty notes

- **Frontier cost:** Derived from the token counts in `coding-smoke-frontier.jsonl` at published Anthropic rates. Labelled `derived-from-tokens`, not metered.
- **Local rows:** Measured live; token counts are kernel-counted in the decision journal.
- **Fixture:** `testdata/coding_smoke/` is a *synthetic* minimal fixture, not a real SWE-bench instance. It proves the *path* works, not that the model can solve real-world coding tasks. A real SWE-bench run is a separate, larger effort (see `SWEBENCH-PURE-KERNEL-RUNBOOK.md`).
- **The honest fence:** This witness does *not* claim "local beats frontier" or any capability parity. It claims: **(a) the kernel works with a local model on CPU, (b) the safety floor holds identically, (c) the cost is $0, (d) the capability gap is what it is.** The A/B table will show exactly that gap once the cells are filled.

---

## Files

- Runbook: `docs/benchmarks/LOCAL-MODEL-CODING-WITNESS-RUNBOOK.md` — the exact commands to run
- Fixture: `testdata/coding_smoke/` — the minimal Python project
- Policy: `examples/coding-agent-safe.json` — the safety floor
- Artifacts: `coding-smoke-{local,frontier}.jsonl` — the hash-chained decision journals (written by `fak guard --audit`)
- BENCHMARK-AUTHORITY row: See `BENCHMARK-AUTHORITY.md` → **Local-model coding witness (2026-06-27)**

---

*Written on a host with no GGUF weights and no frontier model credentials; the completion-rate cells are `pending real run` by design. Fill them in by running the commands in the runbook.*