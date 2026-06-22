---
title: "DOS kernel transfer playbook"
description: "How to run this repo under the DOS kernel's per-call hook adjudication and audit a transfer pass — the advisory audit command and the threshold for filing new process debt against tree_known=false admission cautions."
---

# DOS Kernel Transfer Playbook

This playbook covers driving the repo under the DOS kernel's per-call hook
adjudication (the `dos` plugin's `PreToolUse` / `PostToolUse` / `Stop` hooks) and
auditing what the hooks intervened on afterward.

A **transfer pass** is one window in which a fleet of agents (Claude Code and
Codex) drove the repo with the hooks live. The kernel records every adjudication
to `.dos/metrics/observations.jsonl` (gitignored, re-derivable host state). The
audit below folds a pass into the few numbers that decide whether to file new
process debt.

## Advisory audit (`tree_known=false` admission cautions)

The lane-admission rung proves a tool call's file-tree footprint is **disjoint**
from every held lane lease before allowing it. When the hook cannot derive a
footprint for a call — a free-form shell command, or an editor call whose edited
path the hook can't see — the footprint is *unknown* (`tree_known=false`). With a
lane held, an unknown footprint cannot be proven disjoint, so the rung emits an
**advisory caution** (`outcome=warn`, `rung=admission`): the call still proceeds,
nothing is refused. These cautions are the expected cost of proving disjointness
over opaque calls, not a failure.

### Audit command

Two lenses, both portable, run from the repo root:

```powershell
# Lens 1 — the folded enforcement summary: refused + advisory, by reason and tool.
dos helped

# Lens 2 — the raw observation grep: count the tree_known=false admission cautions.
Get-Content .dos\metrics\observations.jsonl |
  ForEach-Object { try { $_ | ConvertFrom-Json } catch {} } |
  Where-Object { $_.outcome -eq 'warn' -and $_.tree_known -eq $false } |
  Measure-Object
```

Cross-platform lens 2 (Python — `dos doctor` reports the interpreter it found):

```bash
python - <<'PY'
import json
n = 0
for line in open(".dos/metrics/observations.jsonl", encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    try:
        o = json.loads(line)
    except ValueError:
        continue
    if o.get("outcome") == "warn" and o.get("tree_known") is False:
        n += 1
print(n)
PY
```

The two lenses count over different windows — `dos helped` folds the lane journal
while the raw grep counts the per-call observation log — so they report two
nearby totals, not a disagreement (`dos helped` says as much in its own output).

### What a healthy pass looks like

Measured on a live pass (window `2026-06-21T20:05Z → 2026-06-22T08:01Z`, 21,156
adjudicated calls):

| metric | value |
|---|---|
| advisory cautions, all `reason=admission` | 192 |
| of those, `tree_known=false` | 177 (0.84% of all calls) |
| refused calls (held-lane collision) | 3 |
| dominant opaque tool | `Bash` |

Every `tree_known=false` caution traced to a genuinely opaque call — free-form
shell and editor calls the hook can't introspect — not to a call that *does*
carry a visible path. That is the accepted floor.

### Threshold for filing new process debt

File a new **dos-kernel footprint-derivation** issue (the hook change is out of
scope for this repo — see below) when a transfer pass crosses *either* line:

1. **Rate** — `tree_known=false` admission cautions exceed **2%** of adjudicated
   tool calls. (First transfer pass: 1.3%, `305 / 22730`. This pass: 0.84%. The
   2% line is ~1.5× the first-pass rate — a material widening of the footprint
   gap, not noise.)
2. **Shape** — the remaining cautions stop being dominated by genuinely opaque
   calls (free-form shell) and start hitting calls that carry a visible path
   (`Edit` / `Write` / `apply_patch`). That points at a footprint-extraction
   regression in the hook, not inherent opacity, and warrants a dedicated issue.

Below both lines the cautions are the known, accepted cost of proving lane
disjointness over opaque calls — do not file.

## What this repo can and cannot fix

The footprint derivation — teaching the hook to read a read-only or path-scoped
footprint out of a shell command or editor call — lives in the **dos-kernel
plugin**, not in this repo. The `dos.toml` workspace policy declares the lane
taxonomy and refusal vocabulary the hook *consumes*; it carries no knob for
per-call footprint extraction. So:

- Reducing the cautions at the source (deriving a footprint for common read-only
  shell and editor calls, and including an `apply_patch` edit's path set in the
  hook-visible footprint) is a dos-kernel hook change, tracked upstream.
- This repo's job is to **measure** each transfer pass with the audit command
  above and file process debt only when a threshold is crossed.
