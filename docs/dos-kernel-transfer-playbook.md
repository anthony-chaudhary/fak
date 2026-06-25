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

### Codex session witness

For a specific Codex thread, the dogfood witness folds the same stream and
observation evidence into its saved JSON under `checks.dos_session_audit`:

```powershell
python tools\codex_dogfood_witness.py `
  --session $env:USERPROFILE\.codex\sessions\YYYY\MM\DD\rollout-...jsonl `
  --thread-id <codex-thread-id> `
  --fak-bin .\fak.exe
```

That report records the per-session `.dos/streams/<thread>.jsonl` step count, the
timestamp-window fold over `.dos/metrics/observations.jsonl`, the
`tree_known=false` admission-warning rate, delegate count, stop blocks, and a short
recommendation list. It copies only timestamps, tool names, counts, latencies, and
hash digests; it does not copy prompts, tool arguments, tool results, diffs, or model
text. The observation join is explicitly a timestamp window, so concurrent sessions
in the same window can contribute rows.

For a multi-session Codex pass, use the rollup:

```powershell
python tools\codex_dos_recent_audit.py `
  --repo-root . `
  --codex-home $env:USERPROFILE\.codex `
  --limit 10 `
  --since-days 7 `
  --check-latest `
  --out experiments\agent-live\codex-dos-recent-audit.json
```

The rollup matches Codex session IDs to DOS stream files, reports aggregate
unknown-footprint and delegate rates across the recent sessions, and, with
`--check-latest`, compares the locally installed `dos-kernel` package with PyPI's
`https://pypi.org/pypi/dos-kernel/json` `info.version`.

Freshness is not the same as fast-path wiring. The same report also includes
`codex_hook_fast_path`: it reads the cached DOS hook manifest under the Codex home
and reports whether Codex-dialect hooks use the bundled native launcher or the
slower Python CLI hook path. It records only manifest-relative paths and mode
counts, not hook command bodies.

If that check is `WARN`, inspect the repair plan first:

```powershell
python tools\codex_dos_hook_doctor.py --codex-home $env:USERPROFILE\.codex
```

The dry-run prints both current hook modes and projected hook modes after apply.
Treat `projected_codex_command_modes: {"native_launcher": N}` with zero projected
`python_cli` hooks as the proof that the repair would clear the Codex fast-path
warning.

Then apply it explicitly:

```powershell
python tools\codex_dos_hook_doctor.py --codex-home $env:USERPROFILE\.codex --apply
```

The doctor rewrites cached DOS hook manifests to call the bundled native
PowerShell launcher first and leaves the Python CLI hook as the delegate fallback.
It writes a sibling backup before changing a manifest. Like the audit, its report
copies only manifest-relative paths and mode counts, never hook command bodies.

After a repair, the rollup adds two post-repair lenses:

- `post_repair_observations` folds only Codex hook observations after the repaired
  manifest timestamp, so pre-repair delegates do not keep masquerading as current
  fast-path failures.
- `post_repair_command_shapes` reads local Codex session arguments only for the
  same DOS-matched threads included in `sessions_audited`, classifies command
  shape, then drops the command bodies. `shell_no_write_target_detected` means the
  remaining DOS warning is opacity from shell-based read/inspect calls, not an
  observed write target. `shell_out_of_tree_write_target` is the category that
  needs immediate repo-guard review. If mutating shell families are present, the
  report includes a sanitized `mutating_shell_sessions` list with thread id,
  session filename, and category counts only.

Add `--fail-on-warn` when this should act as a local transfer gate instead of an
observational report. The default budget is the playbook's 2% unknown-footprint
line; override it with `--max-unknown-tree-rate` and cap native-hook fallbacks with
`--max-delegates`.

When you need a post-repair risk gate that keeps the residual shell-opacity debt
visible but does not fail on it, use:

```powershell
python tools\codex_dos_recent_audit.py `
  --repo-root . `
  --codex-home $env:USERPROFILE\.codex `
  --limit 10 `
  --since-days 7 `
  --fail-on-actionable-warn `
  --max-delegates 0
```

That gate still fails on an unrepaired fast path, post-repair delegates, stop
failures, unparseable shell commands, `shell_out_of_tree_write_target`, or
mutating opaque shell families such as `git_write`. It does not turn
`shell_no_write_target_detected` into a pass for the strict transfer gate; it
reports read/inspect opacity as `actionability.residual: ["HOST_SHELL_OPACITY",
...]`.

To make the remaining host-opacity debt issue-ready without copying commands or
prompts, emit the sanitized packet. It includes shell shape, family, and mutating
family categories, not raw command text:

```powershell
python tools\codex_dos_recent_audit.py `
  --repo-root . `
  --codex-home $env:USERPROFILE\.codex `
  --limit 10 `
  --since-days 7 `
  --check-latest `
  --out-debt experiments\agent-live\codex-dos-host-opacity-debt.md
```

After routing Git mutations through structured fak gates, include those gate reports
in the audit so historical `git_write` remains visible but does not masquerade as a
new post-mitigation failure:

```powershell
python tools\codex_dos_recent_audit.py `
  --repo-root . `
  --codex-home $env:USERPROFILE\.codex `
  --limit 10 `
  --since-days 7 `
  --gate-report experiments\agent-live\codex-fak-gate-git-add.json `
  --gate-report experiments\agent-live\codex-fak-gate-git-commit.json `
  --gate-report experiments\agent-live\codex-fak-gate-git-push.json `
  --fail-on-actionable-warn `
  --max-delegates 0
```

The gate reports must be expected-deny results for `git_add`, `git_commit`, and
`git_push`. When all three are proven, the audit adds a post-gate command-shape
lens; any `git_write` after that timestamp still fails actionability.

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
