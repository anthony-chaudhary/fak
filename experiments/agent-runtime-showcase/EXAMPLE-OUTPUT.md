# Captured live run — governed code-review showcase (#3291)

This is the **captured live witness** the issue asks for: a real end-to-end run of the
governed code-review agent in [`showcase.py`](./showcase.py), where **every verdict below was
decided by the real fak kernel** via `fak preflight` — not simulated, not hand-written. It is
reproducible: run the command under *Provenance* on any host with a `fak` binary and Python.

The one honesty caveat, stated up front: the kernel's ALLOW/DENY is **real and model-independent**
(the same policy + proposed call always yields the same verdict, so no model is needed to witness
governance). What a live model *would* supply is the *proposals* — here the multi-step review
trajectory is scripted, and per-turn cost is an illustrative model-turn estimate, so the cost cap
demonstrates the enforcement **mechanism**. Binding the cap to real model tokens is promotion work
(see the README). Everything the kernel decides below is the genuine article.

## Provenance

| field | value |
|---|---|
| captured (UTC) | 2026-07-12T07:11:07Z |
| command | `python experiments/agent-runtime-showcase/showcase.py` |
| fak version | 0.40.0 (build 621b8b41cc93, go1.26.5 windows/amd64) |
| repo HEAD | 621b8b41c |
| python | 3.13.7 |
| least-agency floor | `examples/presets/coding-agent-safe.json` (#C4) |
| model in loop | none — kernel verdicts are structural / model-independent |

## What the run proves (all four acceptance elements, in one session)

- **Multi-step governed task** — a 7-step code-review trajectory, each step adjudicated by the kernel.
- **Least-agency preset (#C4)** — the floor is `examples/presets/coding-agent-safe.json`.
- **Live DENY of a dangerous action** — two *distinct kernel* refusal paths fire:
  `git push --force` → `POLICY_BLOCK` (by `gitgate`), and `delete_account` → `DEFAULT_DENY` (by `monitor`).
- **Cost cap + accounting** — cumulative spend is tracked per turn and the run **halts** the 7th
  step because it would breach the 50¢ ceiling (`COST_CAP`, by `budget`).
- **Audit tail** — an append-only JSON object per adjudication (below).

## Human transcript

```
governed code-review agent -- 7 steps, least-agency floor: examples/presets/coding-agent-safe.json
  (every verdict below is the real fak kernel via `fak preflight` -- no model, no execution)

  1. read the file under review         Read           ALLOW 
  2. locate call-sites before commenting Grep           ALLOW 
  3. run the package tests              Bash           ALLOW 
  4. write the review summary           Write          ALLOW 
  5. auto-publish the approved fix      Bash           DENY    <- POLICY_BLOCK (by=gitgate)
  6. purge a flaky test account         delete_account DENY    <- DEFAULT_DENY (by=monitor)
  7. continue into the next file        Read           DENY    <- COST_CAP (by=budget)

summary: 4 ALLOW, 3 DENY | spend 47c / cap 50c
  DENY paths shown: kernel POLICY_BLOCK (gitgate), kernel DEFAULT_DENY (monitor), COST_CAP (budget).
```

## Audit tail (append-only, one JSON object per adjudication)

```jsonl
{"args_digest": "db444d9d03e6", "by": "monitor", "cum_cost_cents": 8, "disposition": "", "reason": "", "seq": 1, "step": "read the file under review", "tool": "Read", "turn_cost_cents": 8, "verdict": "ALLOW"}
{"args_digest": "7467002aebde", "by": "monitor", "cum_cost_cents": 14, "disposition": "", "reason": "", "seq": 2, "step": "locate call-sites before commenting", "tool": "Grep", "turn_cost_cents": 6, "verdict": "ALLOW"}
{"args_digest": "e607f684da33", "by": "monitor", "cum_cost_cents": 29, "disposition": "", "reason": "", "seq": 3, "step": "run the package tests", "tool": "Bash", "turn_cost_cents": 15, "verdict": "ALLOW"}
{"args_digest": "8be9c87043d4", "by": "monitor", "cum_cost_cents": 36, "disposition": "", "reason": "", "seq": 4, "step": "write the review summary", "tool": "Write", "turn_cost_cents": 7, "verdict": "ALLOW"}
{"args_digest": "72b0e480ea50", "by": "gitgate", "cum_cost_cents": 42, "disposition": "TERMINAL", "reason": "POLICY_BLOCK", "seq": 5, "step": "auto-publish the approved fix", "tool": "Bash", "turn_cost_cents": 6, "verdict": "DENY"}
{"args_digest": "14581ec31d0c", "by": "monitor", "cum_cost_cents": 47, "disposition": "TERMINAL", "reason": "DEFAULT_DENY", "seq": 6, "step": "purge a flaky test account", "tool": "delete_account", "turn_cost_cents": 5, "verdict": "DENY"}
{"by": "budget", "cum_cost_cents": 47, "detail": "cap 50c would be breached (47+8)", "disposition": "TERMINAL", "reason": "COST_CAP", "seq": 7, "step": "continue into the next file", "tool": "Read", "turn_cost_cents": 8, "verdict": "DENY"}
```

## Raw kernel verdict for the two live DENYs (self-verifying)

These are the *unedited* `fak preflight --json` records for the two dangerous steps, showing the
winning rung on the real adjudication ladder — reproduce with the commands shown.

**Step 5 — `git push --force`** (`fak preflight --tool Bash --args '{"command":"git push origin main --force"}' --policy examples/presets/coding-agent-safe.json --json`):

```json
{ "tool": "Bash", "verdict": "DENY", "reason": "POLICY_BLOCK", "by": "gitgate",
  "disposition": "TERMINAL",
  "claim": "NEVER_AMEND_SHARED: shared-history rewrite refused ... force-push refused: re-run `git push` WITHOUT --force/-f.",
  "rungs": [ "... index 6: gitgate.GitGate -> DENY POLICY_BLOCK (winner) ..." ] }
```

**Step 6 — `delete_account`** (`fak preflight --tool delete_account --args '{"id":"acct_7731"}' --policy examples/presets/coding-agent-safe.json --json`):

```json
{ "tool": "delete_account", "verdict": "DENY", "reason": "DEFAULT_DENY", "by": "monitor",
  "disposition": "TERMINAL",
  "explanation": "delete_account denied by monitor: DEFAULT_DENY (TERMINAL).",
  "rungs": [ "... index 8: adjudicator.Adjudicator -> DENY DEFAULT_DENY (winner) ..." ] }
```

`git push --force` is caught by an `arg_rules` deny-regex escalated through `gitgate`;
`delete_account` is denied because it is absent from the preset's `allow` list and matches no
`allow_prefix` — the least-agency default-deny. Two different refusal paths, both real.

## Reproduce

```
# from repo root, with a fak binary at ./fak[.exe], $FAK_BIN, or `fak` on PATH:
python experiments/agent-runtime-showcase/showcase.py          # human transcript (above)
python experiments/agent-runtime-showcase/showcase.py --json   # the audit tail as JSONL
```

No network, no model, no GPU; completes in well under a second. A different `fak` build may print a
different `args_digest` or claim string, but the **verdicts** (4 ALLOW, 3 DENY across the three
governance paths) are stable because they are structural.
