---
title: "Claude Code borrow study: 14 single-axis children filed under #4040 from a 46-candidate scan of anthropics/claude-code"
description: >
  A deep /study-repo pass over anthropics/claude-code @ 15a21e1b (the Claude Code
  plugins/skills/hooks distribution — security-guidance LLM review hooks, hookify
  NL->gate compiler, ralph-wiggum loop harness, plugin-dev authoring skills, the
  feature-dev/pr-review agents, and the repo's own critic scripts), scanned for
  disciplines fak's guard/reviewer/routing/context surfaces lack. 46 grounded
  candidates, one dogfooded capability-witness each (fak_feature_query / fak_capabilities
  + grep): 40 PARTIAL, 2 ABSENT, 4 NO-WITNESS. The signal is "fak has the seam without
  the mechanism": it owns capability-narrowing but not CLI-grammar attenuate-and-admit;
  owns diff-witnessed distrust but not per-finding in_diff/off_diff anchoring; owns a
  deterministic runtime gate but not the authoring-time NL->regex compile. Clustered the
  fileable onto 14 single-axis children under epic #4040 (siblings folded as checklist
  items); deduped 2 into already-open #3165/#2753; companion-linked 3 clusters to
  #3846/#2396/#2398. All borrows inspire (Python/bash->Go clean-room); no bytes vendored.
metadata:
  type: project
---

# Claude Code borrow study (2026-07-10)

## What was studied

- **Repo:** [anthropics/claude-code](https://github.com/anthropics/claude-code)
- **Pinned SHA:** `15a21e1b4e240e2da6a4953d5f148a806c9c9bb2` (`15a21e1b`). All `path:line@15a21e1b` citations in the children resolve against this pin.
- **What it is:** the Claude Code plugin/skill/hook distribution plus the repo's own governance scripts. The subsystems mined: `plugins/security-guidance/hooks/` (an LLM-in-the-loop commit/diff security reviewer — diff-anchored triage, adversarial self-refute, injection-safe prompt fencing, whole-file-rewrite churn filtering), `plugins/hookify/` (a two-stage natural-language → deterministic-regex gate compiler), `plugins/ralph-wiggum/` (a filesystem-sentinel autonomous loop harness), `plugins/plugin-dev/skills/` (skill/hook authoring skills with per-load-tier word budgets and a prompt-hook type), the `feature-dev`/`pr-review` agents, `CHANGELOG.md` (context-hygiene + worktree-isolation + out-of-band-consent changes), and the repo's critic scripts (`scripts/gh.sh` read-only facade, `edit-issue-labels.sh` label clamp, `non-write-users-check.yml` privilege-widening guard, `issue-lifecycle.ts` shared stage table).

## Method

Parallel subsystem readers over the pinned tree → 46 grounded candidate borrows (each ablated to the one axis it optimizes, each anchored at a real `path:line@15a21e1b`) → one capability-witness per candidate that **dogfooded fak's own self-index** (`fak_feature_query` / `fak_capabilities` / `fak_index_*`, falling back to grep+read when the feature-query blob overflowed) to grade fak on that axis and name the concrete fak seam → completeness critic. The deep-read fan-out + witness ran as a workflow (session task #2).

## License gate

`anthropics/claude-code` ships **no OSI `LICENSE` file** in the studied tree (source-available distribution). Every borrow here is a bash/Python→Go **clean-room reimplementation of a discipline**, not a byte copy ⇒ all **`inspire`**, **no bytes vendored**. Same posture as the sibling KernelWiki / APEX / ktransformers passes.

## Decisive finding

The pattern across all 46 candidates is **"fak has the seam, not the mechanism."** Witness tally: **40 PARTIAL, 2 ABSENT, 4 NO-WITNESS**. Almost nothing is a clean ABSENT — fak repeatedly owns the *capability* the candidate serves but implements a different, usually coarser, mechanism on the exact axis:

- fak owns capability-narrowing + read-only allowlisting, but `gh`/`git` are just `Bash` → deny-by-regex on the raw command string, **no CLI-grammar positive allowlist and no attenuate-and-admit** (scope a query down rather than deny it whole). → #4046
- fak's reviewer distrust doctrine is pervasive and diff-witnessed, but the LLM reviewer emits **no per-location findings**, so it cannot tag `in_diff`/`off_diff`, cannot sort in-diff-first, and cannot impose an asymmetric off-diff evidence rule — the single largest reviewer FP class isn't even addressable. → #4049
- fak has the full deterministic **runtime** gate (regex/policy floor + severity) but not the **authoring-time** NL→regex+severity compile that produces it — a human is both analyzer and regex author. → #4054
- fak's skill-overlay dedup (`OverlayCache`) is *more* principled than claude-code's loaded-state flag but is **request-path-dead** — nothing in `internal/gateway` calls it, so a re-invoked skill's body is not deduped when the prompt is actually assembled. → #4057
- fak protects its own trust config and escalates *harder* (fail-closed DENY, not an advisory comment) but every seam is **polarity-blind** — a diff that narrows policy trips the floor identically to one that widens it; there is no additive-widening detector. → #4048

The high fileable rate is real but does **not** justify 46 issues — the candidates collapse onto a small set of fak seams, filed as **14 single-axis children under epic #4040**, each folding its sub-mechanisms as a checklist item.

## Filed this pass — epic #4040 (14 children)

**Least-privilege / self-guard (security)**
- #4046 — CLI-grammar read-only allowlist + attenuate-and-admit for gh/git (`internal/adjudicator/decide.go`; folds the NO-WITNESS gh-facade card)
- #4047 — closed-vocabulary label clamp at the issue-edit actuator (`cmd/fak/issue_edit.go`; drop-loud; folds per-run mutation cap)
- #4048 — polarity-aware diff-widening escalation for the guard's own trust config (`internal/dispatchtick/selfmodify.go`)

**LLM-reviewer precision/recall** _(companion #3846 crossaudit)_
- #4049 — soft in_diff/off_diff finding anchor + asymmetric off-diff evidence rule (`internal/modelroute/review.go`)
- #4050 — injection-safe DATA-ONLY fencing of untrusted diff/issue text in producer prompts (`cmd/fak/commit_poison_audit.go`, `internal/dispatchtick/prompt.go`)
- #4051 — decoupled adversarial refute stage: recall detector → precision emitter (`internal/modelroute/crossaudit.go`; folds low-yield re-investigate)
- #4052 — full-file-rewrite pre-existing-content filter, removed∩added → context (`cmd/fak/commit_poison_audit.go:268`)

**Model routing**
- #4053 — error-class model-chain fast-fail: 5xx falls through, 429 retries in place (`internal/agent/retry.go`; folds latency-hedge sibling)

**Hook authoring** _(companion #2396 hookbus)_
- #4054 — two-stage NL→gate compile: authoring-time LLM, runtime LLM-free deny_regex (`internal/policy/policy.go`)
- #4055 — prompt-hook `type` + event-admissibility validator, admissibility gate only / no model call (`cmd/fak/guard_toolproc_hooks.go`)

**Progressive disclosure / context budget** _(companion #2398 ctxpages)_
- #4056 — hard per-load-tier word budgets + no-duplication KPI for skills (`cmd/fak/skill_effectiveness.go`)
- #4057 — idempotent skill injection on the live request path (`internal/syspromptmmu/overlay.go`)
- #4058 — advisory "derivable-from-tree → trim" audit over CLAUDE.md/AGENTS.md (`internal/agentsindex/toc.go` → `cmd/fak/doctor.go`)

**Loop control**
- #4059 — filesystem-sentinel loop disarm: `rm armed-file` → clean exit between turns (`internal/loopmgr/governor.go`, `cmd/fak/loop_drive.go`; folds fixed-prompt re-injection + crash-safe counter siblings)

## Deduped into already-open issues

- **#3165** (worktree isolation) ← subagent commands pinned to their own worktree + a confirm gate before entering an out-of-sanctioned-dir worktree (`CHANGELOG.md:75`). fak's `workerworktree` isolation is default-OFF with no out-of-root entry gate.
- **#2753** (out-of-band operator control) ← background/notification turns assert "no human input" so ambient transcript prose can't be read as consent (`CHANGELOG.md:52`). Directly answers fak's forgeable in-transcript `_fak_confirm` token.

## Captured, not filed (NO-WITNESS — no clean fak seam)

- `scripts/gh.sh:29` — read-only `gh` verb+flag facade (umbrella framing folded into #4046).
- `scripts/issue-lifecycle.ts:3` — one declarative stage table (label + day-threshold + nudge) as the single source of truth for both the commenter and the sweeper; anti-drift pattern for fak's issue-lifecycle automation, but no single fak seam owns it today.

## Follow-ups

- Each child ships with a byte-checkable first step and a failing-test landing; none blocks another.
- The reviewer-precision cluster (#4049–#4052) is the highest-leverage set — it targets the guard-complaint over-block FP class already tracked by #2821 and the crossaudit epic #3846.

_All borrows inspire; `anthropics/claude-code` ships no OSI LICENSE, so every child is a clean-room Go reimplementation of a discipline, zero bytes vendored._
