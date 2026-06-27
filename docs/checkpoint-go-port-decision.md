---
title: "Decision: do NOT port the session checkpoint to Go (gate #634 chose periodic-only)"
description: "tools/session_checkpoint.py stays Python. A Go port was only justified if #634 moved checkpointing onto the PostToolUse hot path; #634 closed on the periodic-only arm, so the precondition is unmet. Revisit only if a future PostToolUse checkpoint arm lands."
---

# Decision: keep the session checkpoint in Python — do not port to Go (`#635`)

**Verdict: DO NOT PORT.** `tools/session_checkpoint.py` (commit `23f1261`) stays a
Python tool. The companion question in [#635](https://github.com/anthony-chaudhary/fak/issues/635)
asked whether it should be compiled to a Go binary the way `repo_guard.py` →
`cmd/repoguard` and `dispatch_worker.py` → `cmd/dispatchworker` were. The answer is
**no — and the condition that would have flipped it did not occur.**

## Why a Go port was ever on the table

The repoguard/dispatchworker ports were forced by a **hard architectural law, not a
general "prefer Go" preference**: the request hot path stays interpreter-free.

- `cmd/repoguard/main.go` runs as a single compiled binary "so the Claude Code
  PreToolUse hook fires WITHOUT spawning a Python interpreter on every tool call
  (DIRECTION.md: the request path stays interpreter-free)."
- Commit `90f6f1d`: the repo_guard PreToolUse hook fires on **every** Bash/Write/Edit
  tool call and re-spawned a Python interpreter each time — a per-decision subprocess
  on the live request path, "exactly the boundary DIRECTION.md / architest
  `TestHotPathHasNoExec` say must stay Go-only and interpreter-free."

That law binds the **hot path**: PreToolUse (every tool call) and the serving request
path. It does **not** bind Stop-time tooling.

## Why the checkpoint is not subject to that law

The checkpoint's `Stop` hook is **not the hot path**. It fires **once per turn-end**,
and Python is the accepted norm for Stop-time tooling — two Python Stop hooks already
live there uncontested (`memory_sync.py`, `switcher_shadow.py`). A Go port of the
Stop+periodic checkpoint buys ~40 ms (Python ~50 ms cold start vs Go ~10 ms) **once
per turn** — real, but not load-bearing, and not on any path `TestHotPathHasNoExec`
protects.

A Go port also has a real *cost* on the safety side: the checkpoint reuses
`tools/scrub_public_copy.py`'s live leak primitives (`AUDIT_REGEXES`, `REPLACEMENTS`,
`load_private_needles`). A Go port would have to either re-implement those needle
regexes (drift risk — two copies of the leak gate) or shell back out to Python
(defeating the interpreter-free win). The scrub gate staying single-sourced in Python
is itself a safety property.

## The gate: #634 chose the periodic-only arm

[#635](https://github.com/anthony-chaudhary/fak/issues/635) was explicitly
**downstream of [#634](https://github.com/anthony-chaudhary/fak/issues/634)**: a Go
port becomes justified *only* if #634 moved checkpointing onto `PostToolUse` (every
successful tool call), because that arm joins the hot path and the same
`TestHotPathHasNoExec` law would then apply.

**#634 did not choose that arm.** It is CLOSED, resolved by commit `7785287`
(`fix(checkpoint): close the within-turn gap by discovering the transcript pointer in
the periodic crash-survivor (#634)`). #634 took the periodic-only arm: the periodic
crash-survivor writer now *discovers* the active transcript pointer from disk, closing
the within-turn coverage gap **without** adding a per-tool-call hook. The evidence in
the tree confirms no hot-path arm exists:

- `session_checkpoint.py`'s `--source` accepts only `["stop", "periodic"]` — there is
  no `posttooluse` source.
- `.claude/settings.json`'s `PostToolUse` block wires the DOS `posttool` hook only; the
  checkpoint is not registered there.

So the checkpoint never moved onto the hot path. **The precondition for a Go port is
unmet, and this issue resolves as "do not port."**

## Revisit criteria

Reopen this decision **only if** a future change adds a `PostToolUse` checkpoint arm
(an unconditional one — a rate-limited arm that no-ops most calls is a judgment call,
not an automatic trigger). If that lands, port **only the hot-path arm** to Go:

- a `cmd/checkpoint/` standalone binary, parity-tested against the Python tool (same
  record shape, same gate verdicts);
- a `make build` drop into `tools/.bin/`;
- a prefers-binary-falls-back-to-Python `.claude/settings.json` wiring, exactly like
  the repoguard pattern;
- **resolve the scrub-primitive duplication first** — extract the needle regexes into a
  single source of truth both languages read (or a `fak scrub --audit` subcommand the
  Python tool also calls), so the leak gate never forks into two drifting copies.

Grounding: `tools/session_checkpoint.py` (`23f1261`), `cmd/repoguard/main.go`,
`.claude/settings.json`, `Makefile`; #634 resolving commit `7785287`. Hook firing
model per <https://code.claude.com/docs/en/hooks.md>.
