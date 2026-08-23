---
name: refresh-cachedoc-numbers
description: Refresh the recent-operational cachevalue numbers in a guarded doc (e.g. docs/integrations/fable5-more-usage-for-free.md) when this-week's telemetry has moved on. Re-derives the frozen snapshots from live `fak cachevalue report`, reconciles the doc's rendered numbers + snapshot_date to the fresh capture, and re-runs the hermetic audit until it is clean. The audit (tools/cachedoc_numbers_audit.py, gated in `make cachedoc-numbers-lint`) binds every rendered number to a committed snapshot field and checks the arithmetic invariants the doc asserts — this skill is the maintenance loop that keeps that. Use when this named workflow matches the task.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, Edit, Grep, Glob
argument-hint: "[manifest-basename] (no args = every guarded doc; e.g. fable5-more-usage-for-free.json)"
---

# /refresh-cachedoc-numbers — re-capture a cachevalue doc's live numbers, honestly

A doc under this gate (see `tools/docnumbers/README.md`) quotes **recent
operational telemetry** — this week's fleet `fak cachevalue report`, the last few
days of dev sessions. Those windows advance, so the numbers go stale. This skill
re-captures them from live `fak`, reconciles the doc to the fresh capture, and
proves the result with the hermetic audit — instead of hand-editing prose and
hoping the arithmetic still closes.

**The distrust that matters:** the new numbers come from a fresh `--json`
capture, and every one of them must land back in a committed snapshot the audit
checks. You are not "updating the text" — you are re-deriving the evidence and
making the text match it.

## Steps

1. **See what's guarded and whether it's stale.**
   ```
   python3 tools/cachedoc_numbers_audit.py
   ```
   A `WARN [staleness]` line, or a `FAIL`, tells you which manifest needs a
   refresh. `Read` that manifest (`tools/docnumbers/<stem>.json`) and its doc.

2. **Re-derive the frozen snapshots from live `fak`.**
   ```
   python3 tools/cachedoc_numbers_audit.py --refresh   # add --manifest <stem>.json to scope it
   ```
   Run this **from the repo root**, where `fak` has its data context. It rewrites
   the snapshots under `snapshot_dir` for every source that has a `cmd`, and
   prints how many fields each got. Sources with `cmd: null` are
   machine-specific per-session captures — recapture those by hand with
   `fak cachevalue status --session <id> --json` if the doc still cites them.

3. **Read the fresh snapshots and reconcile the doc.** For each changed field,
   `Edit` the doc's rendered number to the correctly-rounded fresh value, and
   `Edit` the manifest's matching `expected` (and any `appears_as` string that
   embeds the number). Re-derive any invariant `expect`/`parts` and the
   `snapshot_date`. Keep the provenance labels (`WITNESSED`/`OBSERVED`/
   `ESTIMATED`) truthful — a value's provenance does not change just because it
   moved.

   > The audit will reject a doc number that is not a correct rounding of its
   > snapshot field, an `expected` that disagrees with the snapshot, or a
   > decomposition that no longer sums. Let it drive you: fix what it names.

4. **Prove it.**
   ```
   python3 tools/cachedoc_numbers_audit_test.py    # contract still holds
   python3 tools/cachedoc_numbers_audit.py         # the corpus is clean
   ```
   Optionally `--live` to confirm the cited fields still exist in fresh output.
   Both must exit 0 with no FAIL before you commit.

5. **Commit the doc, manifest, and snapshots together** — they are one artifact.
   Stamp the commit with the doc's lane (`fak <leaf>`; use
   `mcp__fak__fak_index_lane` if unsure).

## Guardrails

- Never edit a rendered number without also moving its snapshot + `expected`; a
  number the audit can't trace to a committed field is exactly what this gate
  exists to stop.
- Don't invent precision. Match the doc's existing rounding (a `≈` value gets 3%
  slack; a plain value must round correctly). If a field is now zero or missing
  after `--refresh`, you likely ran it outside the repo's `fak` data context —
  re-run from the repo root before trusting it.
- This skill only touches guarded docs. Committed benchmark claims belong to
  `docs/benchmarks/BENCHMARK-AUTHORITY.md`; the README's numbers to
  `tools/readme_freshness_audit.py`. Don't cross those streams.
