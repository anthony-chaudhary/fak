---
title: "Guard-session dataset: a durable, policy-attributable corpus for training + regression"
description: "The operating plan for turning every fak guarded session into one durable, redacted, schema-versioned dataset the fleet can learn from, regression-test policies against, and use to understand how the guard loop and its policies actually behave."
---

# Guard-session dataset — operating plan

**Goal.** Turn *every* `fak manage -- <agent>` session — now and for future
sessions — into one **durable, curated dataset** the fleet can (1) learn from and
train on, (2) run **regression tests** against (a policy change must not silently
flip a historical verdict), and (3) mine to **understand how the guard loop and
its policies behave**. The near-term forcing function is getting the `fak manage`
RSI loops *better used*: they already fold our own journals, but the raw journals
they read are host-local, gitignored, and pruned — so the signal evaporates before
it becomes a durable, shareable corpus.

This plan is spine-first (`docs/spine-first-defaults.md`): the minimal working
end-to-end surface is a pure fold + a hermetic test; the hardening backlog is filed
as issues under the epic at creation time.

---

## What exists today (grounding)

A guarded session already emits a **default-on, hash-chained decision journal** —
one JSONL row per kernel adjudication. This is the primary durable witness and the
raw material for the dataset. The pieces already in the tree:

| Layer | Where | What it gives us |
|---|---|---|
| **Decision journal** (raw rows) | `internal/journal/journal.go` — package doc `:1`, `Row` schema `:63`, append `:236`, `Verify` `:856` | Per-adjudication rows: `kind`, `tool`, `trace_id`, `verdict`, `reason`, `by` (deciding gate), `taint`, `args_label`, `witness`, `args_digest`/`result_digest`, `call_seq`; hash-chained + tamper-evident. Off by default; `fak manage` defaults it on to `.dispatch-runs/guard-audit/interactive-<pid>-<hash>.jsonl` (`cmd/fak/guard_support.go:461`, `:530`). |
| **Verdict RSI fold** | `internal/guardrsi/guardrsi.go` (`FoldRows` `:150`, `VerdictQuality` `:237`, `WorstBucket` `:248`) | Folds journals → `verdict_quality` + worst honesty hole (blank-reason-on-deny, unknown-verdict, witnessless-block, child-crash). This is "getting the guard loop used", scored by `fak guard-rsi-scorecard`. |
| **Cross-session usage rollup** | `internal/auditusage/auditusage.go` (`Fold` `:300`, `GuardRollup` `:113`) | Pure fold of the journal + 6 other sinks into one `fak audit usage` report, with a `witnessed`/`observed` basis tag per section. |
| **Durable central mirror** | `internal/logvault` (read via `auditusage.VaultInput`) | Hash-chained + mirror-verified backup of every journal — the durable *backup*, not a curated dataset. |
| **Journal reaper** | `internal/guardaudit/guardaudit.go` (`Plan` `:51`, `Apply` `:125`), `fak guard-audit prune` | Deletes journals older than 7d / beyond 1500 files, **only after** a logvault mirror witnesses their bytes. This is *why* the raw signal is ephemeral. |
| **Effective policy digest** | `cmd/fak/guard_support.go:513` (`guardEffectivePolicyDigest`), computed at startup `cmd/fak/guard_startup.go:252`, stamped into spawn metadata `cmd/fak/guard_child.go:708` | A normalized content digest of the *effective* policy = base floor + allow overlay + deny overlay. Distinguishes policy versions deterministically (`guard_effective_digest_test.go`). |

### The two gaps this plan closes

**Gap 1 — the dataset does not exist.** The journals are the raw data, but nothing
folds them into a **committed, schema-versioned, redacted, per-session dataset** that
survives the reaper, travels across hosts, and is diffable in git. `auditusage` gives
a *rollup* (counts), not per-session records; `logvault` gives a *backup* (raw bytes,
off-repo, unschematized); `guardrsi` gives a *score*, and reads the raw journals it
scores. There is no "one row per session, with its tool-call sequence and outcome"
artifact you can train on or replay. (`internal/advmodel/corpus.go` is an unrelated
adversarial-prompt corpus — the name is taken but the concept is new here.)

**Gap 2 — verdicts are not policy-attributable once aggregated.** A journal `Row`
records *which gate* decided (`by` = `secretgate` / `a2achan/gate` / `advmodel` /
`vdso` / …) but **not which policy version was active**. The effective policy digest
exists (`guardEffectivePolicyDigest`) but is stamped only into the **per-session
spawn-metadata sidecar** (`.dispatch-runs/*.startup.json`), keyed by pid/trace. Once
journals are mirrored to the vault or aggregated across hosts — away from their
sidecars — the join is lost, and you can no longer answer *"which policy produced
this verdict?"* That question is the whole point of "understanding how policies
work", and it is unanswerable in the aggregated data today. This is the concrete
**data element to add**.

---

## The dataset — schema and shape

The dataset is a **repo-tracked analytics ledger** (the `docs/nightrun/*` axis from
`docs/observability/durable-artifacts.md`): committed, grows, folded/cut to bound.
Two records, both schema-versioned, both derived by a **pure fold over journal rows**
(same rows in → same records out, no wall-clock, no RNG) so they are hermetically
testable and reproducible.

### 1. Session record (`fak-guard-session/1`)

One row per guarded session (keyed by `trace_id` + journal-segment identity):

```
{ "schema": "fak-guard-session/1",
  "trace_id": "...", "agent": "claude-code", "host_class": "desktop",
  "policy_digest": "sha256:...",        // Gap-2 fix: the effective policy in force
  "started_unix_nano": ..., "ended_unix_nano": ...,
  "tool_calls": N, "by_verdict": {"ALLOW":..,"DENY":..,"QUARANTINE":..},
  "by_reason": {"POLICY_BLOCK":..,"SECRET_DISCOVERED":..},
  "by_gate":  {"secretgate":..,"a2achan/gate":..,"vdso":..},
  "honesty_holes": {"blank_reason_on_deny":0,"unknown_verdict":0,"witnessless_block":0,"child_crash":0},
  "outcome": "CLEAN|CRASHED|RATE_LIMITED",
  "chain_verified": true }               // was the source chain intact?
```

### 2. Adjudication example (`fak-guard-example/1`)

One row per *interesting* adjudication — the replayable training/regression unit. Only
the **redacted, bounded** fields already deemed safe for the journal are carried
(`args_label`, never raw args; `witness` claim; digests) — the dataset inherits the
journal's redaction contract (`internal/journal/journal.go:552` `ArgsLabelForBytes`,
`secretish` `:729`), it never re-derives a looser one:

```
{ "schema": "fak-guard-example/1",
  "policy_digest": "sha256:...", "tool": "Bash", "args_label": "command=rm path=..",
  "verdict": "DENY", "reason": "POLICY_BLOCK", "by": "floor", "taint": "trusted",
  "witness": "<offending glob / arg bound>", "kind": "DENY" }
```

"Interesting" = every block/quarantine/witness/crash + a bounded sample of allows, so
the corpus is class-balanced and small enough to commit and diff.

---

## What each consumer gets

- **Training / learning.** `fak-guard-example/1` rows are labeled (`verdict`,
  `reason`, `by`, `policy_digest`) redacted tool-call shapes — the exact supervision
  a verdict-prediction or reason-classification model needs, without any raw payload
  leaving the redaction contract.
- **Regression tests (the load-bearing win).** Pin a golden slice of the dataset +
  the `policy_digest` it was produced under. A policy change re-runs the recorded
  `(tool, args_label-shape)` inputs through the kernel at the *new* digest and asserts
  the verdicts are unchanged except where the change intends to move them. A silent
  flip (a floor edit that starts denying a previously-allowed shape, or vice-versa) is
  caught as a diff against the golden dataset — today nothing catches this.
- **Policy understanding.** With `policy_digest` on every row, you can slice the
  corpus by policy version: *which reasons does this policy fire, how often, on which
  tools, and how did that distribution shift when the overlay changed?* The
  `by_gate` breakdown shows which adjudicator carries each policy's weight.

---

## Track A — the spine (build this session)

**A0 (data element, Gap 2).** Carry the effective `policy_digest` into the dataset
fold. The digest already exists per session (`guardChildSpawnMetadata.PolicyDigest`);
the spine joins it in at fold time from the session's startup sidecar, and the
hardening backlog (A4) stamps it durably into a journal **session-boundary anchor**
row so the attribution survives even when the sidecar is gone.

**A1 (fold core).** `internal/guardcorpus` — a pure package: `[]journal.Row`
(+ session metadata) → `SessionRecord` + `[]Example`. No I/O, no clock, deterministic.
Inherits the journal redaction contract; never emits a field the journal would redact.

**A2 (witness).** A hermetic test that plants a synthetic row sequence (allow, deny
with reason+witness, quarantine, a blank-reason deny, a child-crash) and asserts the
folded record's counts, honesty-holes, outcome, policy attribution, and example
redaction — same discipline as `guardrsi_test.go` and the `dispatch sessions` fold.

Later (fan-out): the CLI surface `fak manage corpus build|verify` that reads the real
`.dispatch-runs/guard-audit/*` (via `guardrsi.JournalPaths`) + logvault, writes the
committed dataset under `docs/nightrun/guard-corpus.jsonl`, and bounds it with a `Cut`.

### Spine result (this session)

The pure fold (`internal/guardcorpus`, tier 2) + its golden regression harness are
built and green. The fold was run over this host's **real** journals — **510
sessions, 37,647 tool calls, 332 denies, 2 child-crashes folded** — and the golden
round-trip test (`TestGoldenCorpusRoundTrip`) byte-locks a committed dataset fixture
so a fold change that silently flips a record fails CI. That is the A5 regression
mechanism in miniature, seeded.

**Finding surfaced by the real-data fold (A-advisory).** The dominant real verdict
is `ADVISORY` (~39 rows/session, `kind: TOOL_DEFINITION_PRUNED`, `by:
tool-definition-pruner`) — a legitimate non-blocking monitor verdict. But
`guardrsi.KnownVerdicts` (`internal/guardrsi/guardrsi.go:25`) OMITS `ADVISORY`, so
the guard-RSI **verdict-quality metric miscounts every advisory as an
`unknown_verdict` honesty hole** (~20k phantom holes across 510 sessions), pinning
its worst bucket on a false signal. `guardcorpus` does not inherit the bug
(`TestFoldAdvisoryIsNotAnUnknownVerdict`); the fix to `guardrsi` is filed as a
fan-out item because it moves a *ratcheted* scorecard number and needs its own
witness. This is exactly the "understand how policies/verdicts actually behave"
payoff the dataset promised — it fell out of the first real fold.

### Track A fan-out backlog (file under the epic)
- **A-advisory** — add `ADVISORY` to `guardrsi.KnownVerdicts` (+ its verdict-quality test), so the RSI loop stops counting monitor advisories as honesty holes; re-baseline the `guard_rsi_debt` scorecard under the corrected metric.
- **A3** — CLI `fak manage corpus build|verify|stats`; wire into `fak garden` tick so every session appends (mirrors the `guard-verdict-rsi route` auto-tick).
- **A4** — durable policy attribution: emit a `SESSION_BOUNDARY` journal row (or extend `Row`) carrying `policy_digest`, so aggregated/mirrored journals stay attributable with no sidecar. Keep it out of the hash-chain pre-image (like the crash/restart fields, `journal.go:106`+) so existing journals verify byte-for-byte.
- **A5 (regression harness)** — golden-dataset replay: re-adjudicate recorded inputs at a pinned policy digest, diff verdicts, fail on an unintended flip. The regression-test deliverable.
- **A6** — bound + commit: `Cut`-style compaction (`internal/gatewayusageledger/cut.go` pattern) so the committed corpus stays diffable; class-balanced sampling of allows.
- **A7** — cross-host aggregation: fold logvault-mirrored journals from all fleet hosts into one corpus (dedupe by `trace_id` + chain hash).
- **A8** — scorecard: a `guard_corpus` member (coverage = sessions captured / sessions run; freshness; policy-version count) on the control-pane ratchet, driven by an RSI skill — the "better used" loop closes on corpus coverage, not just verdict-quality.
- **A9 (docs)** — doctrine + doc-map linkage (this plan, `docs/observability/durable-artifacts.md`, `INDEX.md`, `llms.txt`).

---

## Sequencing

1. This plan doc (spine witness for the fan-out). ✅
2. File epic + Track A children (`fak issue create`, milestone + labels at creation).
3. Build + test the `internal/guardcorpus` fold spine (A1/A2), green `go test ./internal/guardcorpus`.
4. Note the spine commit on the fold child; land A0 policy-attribution join next.

## Read next

- [Durable artifacts inventory](observability/durable-artifacts.md) — where every guard/session sink lives and who may delete it.
- [Guard verdict-pattern RSI loop](fak/guard-verdict-rsi-loop.md) — the loop this dataset feeds; today it reads the raw journal, tomorrow it reads the corpus.
- [Dispatch session observability plan](DISPATCH-SESSION-OBSERVABILITY-PLAN.md) — the sibling plan that folds the *dispatch* plane; this one folds the *guard-adjudication* plane.
