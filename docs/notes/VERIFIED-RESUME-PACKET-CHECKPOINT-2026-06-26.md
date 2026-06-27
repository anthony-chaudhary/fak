# Verified resume packet — sourcing the checkpoint from witnessed state (#636) (2026-06-26)

**Status:** design note (issue [#636](https://github.com/anthony-chaudhary/fak/issues/636), `docs` lane).
The schema and the consume path are specified here; the flag-gated code increment in
`tools/session_checkpoint.py` is the named follow-on (it lives in the `tools` lane, not `docs`).

## The crux

`tools/session_checkpoint.py` (the `fak-session-checkpoint/1` record) renders a **self-reported,
human-readable** status note from a raw `git` snapshot — branch, HEAD, dirty path *names*, and a
free-text `in_flight` line — then routes it off-host. That free-text line is the weak seam. It is
populated from `--in-flight` or, in Stop mode, from the Stop event's `goal` / `stop_hook_condition`
(`session_checkpoint.py:466`). In other words: *what the agent says it was doing*, never checked.

fak already records the strong form of that fact in two places, neither of which can be a
self-report:

- **`internal/trajectory`** — a `Recorder` is an `abi.Emitter`; the kernel fans every adjudication
  to it and it folds the stream into per-trace `Turn` rows (`trace_id`, `seq`, `query`, `tool`,
  `verdict`, `reason`, `taint`, `token_estimate`, `cache_hit`, `labels`). This is the **witnessed
  verdict trace** of what the kernel actually decided, exported as stable JSONL by `fak traj export`
  (`internal/trajectory/trajectory.go:24`, `:151`).
- **the `dos_status` digest** — one fail-closed `{schema, run_id, liveness, progress, region,
  resume}` fact. Its load-bearing property: **it has no `claimed` field by construction**, so a peer
  reading it structurally cannot pick up a worker's self-report. `progress` is built from the
  kernel's ledger-VERIFIED rung only; `region` is the held-lease tree; `resume` is the resume plan,
  null while the run is live and populated once it has stopped (`dos status RUN_ID --json`).

The adjustment in #636: make the checkpoint say *what was witnessed to have happened*, and make it
**resumable by a fresh or cloud agent**, not just readable by a human after a crash.

## What stays, what changes

Keep `session_checkpoint.py` as the dumb, always-on, off-host **safety net**. It works today, needs
nothing from the kernel, and survives a hard crash via the periodic writer (the only writer that
runs *before* the crash — a Stop hook never fires on a TDR kill; see the tool's header). The git
snapshot stays as the **floor**.

Add a **witnessed overlay** on top of the floor: when a run id (and, optionally, an exported
trajectory corpus) is available, fold the `dos_status` digest + the trajectory tail into the record
and let them *replace the self-reported `in_flight` line* with verified progress. Absent those
inputs, the record degrades cleanly to today's v1 — the floor is never weakened.

## The schema — `fak-session-checkpoint/2` (superset of v1)

v1 fields are unchanged (`schema`, `source`, `stamp`, `host`, `repo_root`, `branch`, `head`,
`head_subject`, `dirty_paths`, `dirty_count`, optional `transcript`). The v1 free-text `in_flight`
is **demoted to a fallback**: it is emitted only when the witnessed block below is absent. Two
additive blocks are introduced (both `omitempty`, so an old reader keeps parsing):

```json
{
  "schema": "fak-session-checkpoint/2",
  "source": "stop",
  "...": "all v1 git/host fields unchanged",

  "witnessed": {
    "run_id": "RID-…",
    "liveness": { "moving": true,  "forward_commits": 3, "stalled": false },
    "progress": { "phase": "<ledger-VERIFIED rung>", "verified": true },
    "region":   { "lane": "docs", "tree": ["docs/**"] },
    "trajectory_tail": [
      { "trace_id": "sess-a", "seq": 41, "tool": "search_kb",      "verdict": "ALLOW", "reason": "" },
      { "trace_id": "sess-a", "seq": 42, "tool": "refund_payment", "verdict": "DENY",  "reason": "POLICY_BLOCK" }
    ]
  },

  "resume": {
    "verdict": "<dos_status.resume, null while live>",
    "region": ["docs/**"],
    "last_witnessed_phase": "<= witnessed.progress.phase>",
    "context_pointer": {
      "kind": "transcript",
      "ref": "<host path to the active .jsonl>"
    }
  }
}
```

- **`witnessed`** is the verified replacement for `in_flight`. `liveness` / `progress` / `region`
  come verbatim from the `dos_status` digest (no `claimed` field is ever copied because the digest
  has none). `trajectory_tail` is the last *N* (default 5) `Turn` rows for the active trace, read
  from the corpus `fak traj export` writes — verdicts and reasons, **not** query/result bytes by
  default (see the leak note below).
- **`resume`** is the agent-facing packet. `verdict` and `region` are the `dos_status.resume`
  plan; `last_witnessed_phase` mirrors `witnessed.progress.phase`; `context_pointer` is the seam
  that makes "resume the actual context" real (next section).

## How `session_checkpoint` consumes it

A single flag turns the fold on; everything degrades fail-soft so the crash-survivor floor never
regresses:

```
python tools/session_checkpoint.py --hook --witnessed --run-id RID-… [--corpus <turns.jsonl>]
```

1. **Resolve the run id.** From `--run-id`, else `$DISPATCH_RUN_ID` / the run's intent ledger. No
   run id ⇒ skip the fold, emit v1. (No fabricated id is ever minted here.)
2. **Fold `dos_status`.** Shell `dos status <run_id> --json --workspace <repo>` and copy
   `liveness` / `progress` / `region` / `resume` into the record. A non-zero exit, a missing `dos`
   binary, or an `{error,…}` body ⇒ skip the witnessed block, emit v1. Fail-soft is mandatory: a
   checkpoint failure must never block a Stop or crash the scheduled task (the tool's standing
   contract).
3. **Fold the trajectory tail.** If `--corpus` is given (or discovered next to the transcript),
   read it as JSONL, select the rows whose `trace_id` matches the active trace, keep the last *N*,
   and project each to `{trace_id, seq, tool, verdict, reason}`. Absent/empty corpus ⇒ omit
   `trajectory_tail`.
4. **Demote `in_flight`.** When `witnessed` is present, `render_md` prints the verified progress +
   the verdict tail in place of the free-text line. The self-report only renders when nothing was
   witnessed — and it is then labelled as a self-report, not as progress.

**Leak gates already cover this.** The new fields are digests, lane names, verdicts, and reasons —
the same shape of host-tagged data the **private** route already permits (its gate is
secrets-only, `needle_hits(secrets_only=True)`). The **public** route still runs the full scrub
transform + re-audit and refuses on any surviving needle, so a `query`/`reason` string that names a
host or account can never reach a public surface. To stay conservative, `trajectory_tail` carries
**no `query` text by default** — only the structural verdict columns.

## What a fresh / cloud session needs to resume

This is the third acceptance bullet of #636, spelled out. To pick up where a dead session stopped,
a fresh worker needs exactly three things, all present in the `resume` block:

| need | field | source | status |
|---|---|---|---|
| **region** — which tree to (re)take the lease on, so it cannot collide with a peer | `resume.region` | `dos_status.region` (held lease) | available today |
| **last witnessed phase** — where verified progress actually stopped, never the dead worker's claim | `resume.last_witnessed_phase` + `witnessed.trajectory_tail` | `dos_status.progress` (ledger-VERIFIED rung) + the `Turn` tail | available today |
| **context / KV pointer** — enough to restore the *context*, not just the file list | `resume.context_pointer` | see below | transcript today; KV-span next |

The **context pointer** is the part that turns a human note into a real resume packet, and it has
two fidelities:

- **Today — the transcript pointer.** `session_checkpoint` already discovers and carries the active
  Claude Code `.jsonl` path (`discover_active_transcript`, `session_checkpoint.py:203`). A fresh
  session resumes by replaying that transcript. This is a *path*, not content — the same leak
  surface the route gates already handle.
- **Next — a digest-addressed KV span.** The KV-backend seam (`abi.RegisterKVBackend`,
  `internal/abi/kvbackend.go`) addresses a context span by **digest** and stages/restores it across
  the off-box tier, returning a typed `KVResidency` (`ok | MISS | FAULT`) so a failed restore is
  reported, never a silent recompute. When a run's KV span is staged off-box, `context_pointer`
  becomes `{ "kind": "kv_span", "ref": "<span digest>" }` and a *cloud* worker can `RestoreSpan`
  the actual cached context instead of re-reading the file list. This is the disaggregated-memory
  direction; the packet schema already has the slot for it, gated behind that seam being wired on
  the serving path.

## Increment ladder (what closes #636, and what is the named follow-on)

1. **This note** — the verified-resume-packet schema + the consume path + the resume contract.
   *(delivered here.)*
2. **The flag-gated fold** in `tools/session_checkpoint.py` — `--witnessed --run-id … [--corpus …]`
   bumps the record to `fak-session-checkpoint/2`, replacing the self-reported `in_flight` line with
   the `dos_status` digest + trajectory tail, fail-soft to v1. *Named follow-on, `tools` lane.* Note:
   per `AGENTS.md`, non-trivial work on a `tools/*.py` defaults to a Go port — the issue's own
   end-state is a first-class `fak checkpoint` subcommand with `session_checkpoint.py` as its
   reference *consumer* (the way `trajectory-garden` consumes the trajectory primitives), so the fold
   should land as `fak checkpoint` (Go) wiring + a thin Python consumer, not a deeper Python rewrite.
3. **The KV-span context pointer** — promote `context_pointer.kind` from `transcript` to `kv_span`
   once `abi.RegisterKVBackend` is wired on the live serve loop, so a cloud worker restores the real
   context. *Gated on the disaggregated-memory seam landing on the serving path.*

## Grounding

- `tools/session_checkpoint.py` — the v1 record, the routers, the two writers, `discover_active_transcript`.
- `internal/trajectory/trajectory.go` — the `Turn` schema (`:24`) and JSONL export (`:151`).
- `docs/observability/trajectory.md` — the trajectory data-plane / `fak traj` surface.
- `internal/abi/kvbackend.go` — the `KVBackend` seam, `KVResidency` (`ok | MISS | FAULT`).
- `dos_status` (external `dos` CLI) — the `{liveness, progress, region, resume}` digest, no `claimed` field.
