---
title: "fak proof: the fail-closed guard audit"
description: "The enumerated ledger of every fak manage, adjudication rung, and hook, with its asserted failure mode, its witness test, and an honest finding for each path that fails open."
---

# D1 · fail-closed audit

This is the **coverage ledger** for fak's failure posture: every guard,
adjudication rung, and hook on the kernel's decision path, enumerated with the
mode it takes **when the check itself breaks** — not when it merely returns a
verdict.

The distinction the ledger exists to make precise:

- **fail-closed** — the check errors, cannot reach its evidence, or cannot
  decide, and the outcome is **DENY**. The unproven call does not run.
- **fail-open** — the check errors and is **skipped**. The call proceeds
  unchecked.

The audit's rule is that a fail-open entry is not automatically a defect, but it
is never allowed to be *silent*: it must appear in this ledger with a stated
rationale, or the CI gate below reds the build.

## The architectural rule this audit found

fak does not have one uniform posture, and claiming so would be the overclaim
this ledger is meant to prevent. The real rule, which held everywhere it was
checked:

> **Enforcement fails closed. Observation fails open.**

Every rung that *decides whether a call runs* denies on error. Every surface
that *watches, records, or advises* allows on error, so that a bug in the
telemetry path can never wedge a live fleet. Both halves are deliberate; the
ledger's job is to keep the boundary between them explicit and testable, so a
rung cannot drift from the first category into the second unnoticed.

This is the inverse of a fail-open-by-default posture, where a broken check
silently downgrades to no check at all on the enforcement path itself.

---

## THEOREM 1 — the adjudication core denies on every indecision path

**THEOREM.** `kernel.Fold` has no path that yields anything but a conclusive
verdict or `Deny`. Absence of policy, universal deferral, and residual
indeterminacy each fold to `ReasonDefaultDeny`.

**PROOF.** `Fold` (`internal/kernel/kernel.go:189`) has exactly three terminal
branches that are not a conclusive rung verdict, and all three deny:

- an **empty chain** — no policy loaded — returns
  `Deny/ReasonDefaultDeny, By:"empty-policy"` (`kernel.go:190-192`);
- a **residual indeterminate**, where a rung declined to commit and no later
  rung concluded, returns `Deny/ReasonDefaultDeny` with `Meta{"fold":
  "indeterminate"}` (`kernel.go:221-224`);
- **every rung deferring** returns `Deny/ReasonDefaultDeny, By:"all-defer"`
  (`kernel.go:225`).

Conflicting conclusive verdicts resolve by `abi.FoldRank`, most-restrictive-wins
(`kernel.go:210-216`), so an unrecognized verdict kind cannot outrank `Deny`.

Two adjacent defaults carry the same posture. The adjudicator's `Posture` zero
value is `PostureFailClosed` (`internal/adjudicator/decide.go:135`, `iota`), so
an unset or zero-valued posture is the strict one rather than the permissive
one. The require-witness gate stays closed with `ReasonUnwitnessed` when every
resolver abstains or none is registered (`internal/kernel/kernel.go:237`) — an
uncorroborated claim is not a corroborated one.

**WITNESS.**
```
go test ./internal/kernel/ ./internal/adjudicator/ -count=1 -timeout 120s \
  -run 'TestFoldResidualIndeterminateFailsClosed|TestFoldDefaultDenyEmptyPolicy|TestNeverAdmits'
```

**VERDICT.** **PROVEN** by construction and by the pinning tests above.

---

## THEOREM 2 — an unclassified repo-guard reason denies

**THEOREM.** `repoguard.DefaultSeverity` returns `SeverityDeny` for any reason
not present in its posture table.

**PROOF.** The default posture map is deliberately permissive for the reasons it
names — `OUT_OF_TREE_WRITE` and `LIVE_MONITOR_OUTPUT_READ` resolve to
`SeverityRecord` (silent journal row, allow) and the hint-bearing rungs to
`SeverityWarn` (`internal/repoguard/severity.go:77-85`). The *fallthrough*,
however, is strict: an unknown reason returns `SeverityDeny`
(`severity.go:91-95`), so a refusal-class reason added later denies until it is
explicitly softened in the table. `Severity.String` likewise renders an unknown
value as `"deny"` (`severity.go:53-55`). The permissiveness is enumerated; it is
not the default.

**WITNESS.**
```
go test ./internal/repoguard/ -count=1 -timeout 120s
```

**VERDICT.** **PROVEN**.

---

## The commit-boundary gate ledger

Every gate registered by `hooks.PreCommitGates()`
(`internal/hooks/hooks.go:76`), with its **default enforcement** when it detects
a violation. `block` refuses the commit; `warn` is advisory and prints without
refusing. Each gate additionally honors a `ModeEnv` to soften it and a one-shot
`EscapeEnv` to skip it once — deliberate operator overrides, not fail-open
paths.

The `Fail mode` column is the mode taken when the gate's own `Check` returns an
error. It is `fail-open` for every entry, structurally: `cmd/fak/hooks.go:355`
skips a gate whose check errored and runs the rest. Since #5299 that skip is
reported rather than silent — the gate is named on stderr and carried in the
`--json` output as `skipped_gates`/`skipped_count` — but it is still a skip, and
the fail-open posture recorded below is unchanged. See FINDING 1.

The table below is machine-read by the CI gate; the fenced region is parsed and
cross-checked against the live registry in code.

<!-- failclosed-ledger:begin surface=pre-commit-gate -->

| Entry | Default enforcement | Fail mode | Note |
|---|---|---|---|
| PUBLIC_LEAK | block | fail-open | staged content matched against redact-needles |
| SECRET_SHAPE | block | fail-open | credential-shaped literals in the staged diff |
| DOC_PLACEMENT | block | fail-open | keeps stray docs out of the repo root |
| BROKEN_LINK | block | fail-open | relative markdown links must resolve |
| FILE_ADMISSION | block | fail-open | private-only, junk, or oversized paths refused |
| INDEX_SYNC | block | fail-open | INDEX.md / llms.txt reciprocal-orphan check |
| CONCEPT_ADMISSION | block | fail-open | a new concept needs its glossary entry |
| CONCEPT_FRESHNESS | block | fail-open | concept docs must not go stale against code |
| PROVENANCE_LABEL | block | fail-open | witnessed / observed / modeled labelling |
| HARDWARE_TELL | block | fail-open | no local-hardware blocker as a terminal answer |
| NATIVE_FIRST | block | fail-open | commit-boundary native-inference substitution guard |
| BARE_COMMIT_SWEEP | warn | fail-open | advisory by design (#3615); closes the raw-git bypass |
| E2E_OVER_MOCKS | warn | fail-open | advisory by design (#2901); asks for a witnessed run |
| DESKTOP_POPUP_REGRESSION | block | fail-open | candidate-index Go/Python/PowerShell helpers must suppress background console windows |
| PARALLEL_FABRIC_NUDGE | warn | fail-open | advisory; new parallel-fanout language should name the bounded micro-context fabric |
| GIT_HYGIENE_BYPASS | warn | fail-open | advisory by design (#5588); flags ad-hoc git-lock removal / object maintenance outside the owning packages |
| TRUST_WIDENING | warn | fail-open | advisory; flags a widened trust boundary |
| PRIOR_ART | warn | fail-open | advisory; prints the SOTA reference to cite |
| MICROHARNESS_WITNESS | warn | fail-open | advisory; nudges bounded harness changes toward a staged test or typed receipt |
| UNTIERED_LEAF | warn | fail-open | advisory by design (#3614); staged twin of TIER_DECLARED |
| CART_BEFORE_HORSE | warn | fail-open | advisory by design (#2521); new leaves establish an applied spine before downstream proof/performance breadth |
| GOFMT | block | fail-open | blocking commit-boundary twin of make ci gofmt-check |
| DUPLICATION | warn | fail-open | advisory; in-process twin of fak dup guard --staged |
| COMMENT_QUALITY | warn | fail-open | advisory; changed-lines-only comments should explain durable why |

<!-- failclosed-ledger:end -->

## The repo-guard posture ledger

`internal/repoguard` decides the PreToolUse verdict for every reason it names,
and `defaultSeverity` (`internal/repoguard/severity.go`) is the per-reason
posture table behind that decision. `deny` refuses the call; `warn` returns a
hint and lets it run; `record` journals it and lets it run silently; `off` is not
evaluated at all. A reason with **no** entry in that map resolves to `deny` —
`DefaultSeverity`'s fail-safe fallthrough — so it is enumerated below as the
strict entry it actually is rather than vanishing from the count.

The `Fail mode` column is *derived* from the severity rather than asserted
separately: only a denied call fails closed, so an unannounced escalation to
`deny` reds the gate until this table says so.

This surface was enumerated by hand in the original audit, which is the exact
failure the audit exists to prevent — a complete-*looking* list stops the reader
checking. The fenced table below is parsed and cross-checked against the package
source in both directions by
`internal/hooks/failclosed_ledger_surfaces_test.go`: a new reason with no row
reds, and a row naming no live reason reds.

<!-- failclosed-ledger:begin surface=repoguard-severity -->

| Entry | Default severity | Fail mode | Note |
|---|---|---|---|
| BUILD_CACHE_CLEAN_RACE | deny | fail-closed | ambient Go build-cache deletion is blocked while peer builds may consume it |
| FOREGROUND_NETWORK_LOOP | warn | fail-open | hint returned, call proceeds |
| FOREGROUND_POWERSHELL_INVENTORY | warn | fail-open | hint returned, call proceeds |
| FOREGROUND_SLEEP | warn | fail-open | hint returned, call proceeds |
| INTERACTIVE_HANG | warn | fail-open | hint returned, call proceeds |
| LIVE_MONITOR_OUTPUT_READ | record | fail-open | journalled, call proceeds silently |
| OUT_OF_TREE_WRITE | record | fail-open | journalled, call proceeds silently |
| UNDECLARED_LEAF | warn | fail-open | hint returned, call proceeds |
| WORKSPACE_PATH_UNMAPPED | warn | fail-open | hint returned, call proceeds |

<!-- failclosed-ledger:end -->

## The DOS refusal-vocabulary ledger

`dos.toml`'s `[reasons.*]` blocks are the closed vocabulary a refusal may cite.
`Posture` is the block's own `refusal` flag: `refusal` means a caller may refuse
with it, `advisory` means it reports without refusing. `Floor` records whether
the block names an **enforcing floor** — a `Floor:` cite in its fix text, naming
the code or command that actually stops the behaviour — or declares none.

`floor-absent` is the load-bearing half of this table. A refusal reason with no
enforcing floor is a *name* for a behaviour nothing stops; the audit's point is
that such an entry must be an explicitly declared row, never an omission a reader
mistakes for completeness. The `Note` column is derived from the two columns
beside it on purpose: each block's own prose is the long form, and this ledger
binds the posture, not the wording.

Bound in both directions against `dos.toml` by the same test as the surface
above.

<!-- failclosed-ledger:begin surface=dos-reason -->

| Entry | Posture | Floor | Note |
|---|---|---|---|
| ARCH_LAYER_VIOLATION | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| ASSUMPTION_STALE | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| ASSUMPTION_UNVERIFIABLE | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| ASSUMPTION_VIOLATED | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| BARE_COMMIT_SWEEP | refusal | floor-absent | no enforcing floor declared — vocabulary-only |
| BLOCKED_BY_KNOWN_BAD | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| BLOCKED_BY_OPEN_PREREQ | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| BROADCAST_MALFORMED | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| BUILD_CACHE_CLEAN_RACE | refusal | floor-absent | no enforcing floor declared — vocabulary-only |
| CACHE_PREFIX_RESIDENT | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| DEPTH_NOT_CARRIED | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| CHECKER_TAMPERED | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| COLLISION_RISK | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| COMPACTION_THRASH | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| CONTROL_REV_STALE | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| CONTROL_SESSION_TERMINAL | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| CORE_SELF_MODIFY | refusal | floor-absent | no enforcing floor declared — vocabulary-only |
| CRASH_RESTART_EXHAUSTED | advisory | floor-absent | no enforcing floor declared — vocabulary-only |
| DISAMBIGUATION_TIMEOUT | refusal | floor-absent | no enforcing floor declared — bounded timeout refusal vocabulary |
| DOOM_LOOP | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| FILE_ADMISSION | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| FLEETBUS_APPLY_REFUSED | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| FLEETBUS_EXPIRED | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| FLEETBUS_MALFORMED | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| FLEETBUS_NO_TARGET | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| FLEETBUS_UNKNOWN_OP | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| FOCUS_WIP_SATURATED | advisory | floor-declared | `Floor:` cite in the dos.toml block |
| FOREGROUND_SLEEP | advisory | floor-declared | `Floor:` cite in the dos.toml block |
| FRESH_DELETION | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| FRONTIERSWE_SCORE_PARITY_FAILED | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| GATE_LATENCY_REGRESSION | refusal | floor-absent | no enforcing floor declared — vocabulary-only |
| GATE_PRESSURE | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| HOST_CHURN_BACKOFF | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| INDETERMINATE | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| INTERACTIVE_HANG | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| INVALID_OPTIONS | refusal | floor-absent | no enforcing floor declared — vocabulary-only |
| ISSUEFANOUT_CONTRACT_REFUSED | advisory | floor-absent | no enforcing floor declared — vocabulary-only |
| KNOWN_BAD_ALREADY_CLAIMED | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| KNOWN_BAD_EXPIRED_OR_REVOKED | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| KNOWN_BAD_NOT_WITNESSED | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| L3_CROSS_TENANT_SCOPE_DENIED | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| L3_PAGE_DIGEST_MISMATCH | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| L3_UNWITNESSED_FLEET_STAMP | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| LEDGER_WRITE_FAILED | refusal | floor-absent | no enforcing floor declared — vocabulary-only |
| LESSON_OVERCLAIMS | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| LIVE_MONITOR_OUTPUT_READ | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| LOCK_CLEANUP_FAILED | refusal | floor-absent | no enforcing floor declared — vocabulary-only |
| LOOP_DONE_UNWITNESSED | refusal | floor-absent | no enforcing floor declared — vocabulary-only |
| MAINTENANCE_INCIDENT | refusal | floor-absent | no enforcing floor declared — vocabulary-only |
| MESSAGE_RACE | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| MICROCONTEXT_LEDGER_REFUSED | refusal | floor-absent | no enforcing floor declared — vocabulary-only |
| MICROHARNESS_UNWITNESSED | advisory | floor-absent | advisory recovery vocabulary; no enforcing floor declared |
| MODEL_TOON_UNFIT | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| NET_TOKENS_NONPOSITIVE | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| NEVER_AMEND_SHARED | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| OBJECTIVE_SCORER_MISSING | advisory | floor-absent | no enforcing floor declared — vocabulary-only |
| OFF_TRUNK | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| OUTPUT_DIRECTION | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| OUT_OF_DIRECTION | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| OUT_OF_TREE_WRITE | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| OVERHEAD_BUDGET_EXCEEDED | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| OVERLAY_WOULD_GATE | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| PAYLOAD_TOO_SMALL | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| PIN_EVICT_REFUSED | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| PUBLIC_LEAK | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| RATE_LIMIT_BACKOFF | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| REDIRECT_MALFORMED | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| REDIRECT_NO_REDIRECTABLE_STATE | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| RELAY_IDLE_PARKED | advisory | floor-declared | `Floor:` cite in the dos.toml block |
| RELAY_NO_PROGRESS | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| RELAY_ORPHANED_FOLLOWON | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| REQUIRE_WITNESS | refusal | floor-absent | no enforcing floor declared — vocabulary-only |
| RESUME_COST_EXCEEDED | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| ROUNDTRIP_LOSSY | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| RUN_STATUS_CLAIMED_FIELD | refusal | floor-absent | no enforcing floor declared — vocabulary-only |
| SESSION_CEILING_SATURATED | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| SKILL_DESC_BUDGET_EXCEEDED | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| SKILL_DESC_BUDGET_STALE | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| STALE_BASE_DELETION | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| STALE_RECALL | refusal | floor-absent | no enforcing floor declared — vocabulary-only |
| STEER_NO_OWNED_LOOP | refusal | floor-absent | no enforcing floor declared — vocabulary-only |
| SYSTEM_COMMIT_HEADROOM | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| TABULAR_ELIGIBILITY_LOW | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| TICK_BUSY | advisory | floor-absent | no enforcing floor declared — vocabulary-only |
| TICK_LOCK_ERROR | refusal | floor-absent | no enforcing floor declared — vocabulary-only |
| UNAUTHENTICATED_OFF_HOST_BIND | refusal | floor-absent | no enforcing floor declared — vocabulary-only |
| UNTIERED_LEAF | refusal | floor-absent | no enforcing floor declared — vocabulary-only |
| VALUECHAIN_UNWITNESSED | advisory | floor-absent | no enforcing floor declared — vocabulary-only |
| VOLATILE_SPAN | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| WEBHOOK_URL_NOT_ALLOWLISTED | refusal | floor-declared | `Floor:` cite in the dos.toml block |
| WORKSPACE_PATH_UNMAPPED | advisory | floor-declared | `Floor:` cite in the dos.toml block |

<!-- failclosed-ledger:end -->

## The observation surfaces

These are enumerated for completeness. All fail open, and all are correct to do
so: none of them decides whether a tool call runs.

| Surface | Fail mode | Why it is not a finding |
|---|---|---|
| `cmd/repoguard` PreToolUse hook | fail-open | Documented at `cmd/repoguard/main.go:12`; always exits 0 and signals a deny through the JSON decision payload, so a guard bug cannot wedge a fleet. See FINDING 2. |
| Stop hook (`cmd/fak/guard_stophook.go`) | fail-open | Allows the stop when the gate cannot decide, and labels the refusal string `fail-open:` so the choice is visible in the ledger rather than silent. |
| Pre-compact hook (`cmd/fak/guard_precompact.go`) | fail-open | Context maintenance, not admission. |
| Tool-process hooks (`cmd/fak/guard_toolproc_hooks.go:22`) | fail-open | Instrumentation; always exits 0 by design. |
| Kernel observers (`internal/kernel`) | fail-open | A panicking observer is recovered per-observer and cannot change the verdict — a *tested* contract (`internal/kernel/emit_failopen_test.go`). |
| Guard launcher config path (`cmd/fak/guard.go`) | fail-closed | Setup, flag, and config errors exit 2; the session does not launch. |

---

## Findings

Per this issue's scope, findings are *recorded*, not fixed here. Fixes that
landed later under their own issues are noted inline, so this stays a live
ledger rather than a snapshot that quietly goes stale.

**FINDING 1 — a pre-commit gate that errors is skipped silently. RESOLVED (#5299).**
At the time of this audit `cmd/fak/hooks.go` continued past any gate whose
`Check` returned an error. The skip is deliberate and contractual
(`internal/hooks/hooks.go:36-39`, `ErrCouldNotRun`: "fail-open, never a block"),
and the rationale is sound — a broken checker must not wedge every commit on a
shared trunk. The defect was not the fail-open, it was the **silence**: no
stderr line, no counter, and no distinction in the exit code between "17 gates
ran clean" and "PUBLIC_LEAK errored and the other 16 ran clean". An operator
could not tell a green commit from a degraded one. The smallest fix is
observability, not enforcement: emit the skipped gate's name and count it.

**Resolution.** #5299 did that and no more. `cmd/fak/hooks.go:355-369` names the
gate on stderr and appends it to a ledger surfaced in `--json` as
`skipped_gates`/`skipped_count`. Both keys are emitted on a clean run too, as
`[]` and `0`, so a consumer can never read an absent key as "nothing was
skipped". The fail-open posture is unchanged and is now pinned by test — every
witness asserts the exit code stays 0, and a mutant that flips could-not-run to
blocking reds them. One deliberate restraint: the gate's error VALUE is never
rendered, only a fixed classification literal, because `PUBLIC_LEAK` and
`SECRET_SHAPE` scan the staged diff and their error text can carry the very
material they matched. Witness:

```
go test ./cmd/fak/ -count=1 -run 'PreCommit|EnabledGateNames'
```

**FINDING 2 — the PreToolUse hook swallows malformed input as "allow".**
`cmd/repoguard/main.go` fails open on any internal error, including a payload
unmarshal failure. A guard *bug* and a *malformed payload* are different failure
classes: the first argues for fail-open, the second is attacker-influenceable
input. Worth separating, though the hook is a heuristic floor and not a sandbox
by its own documentation.

Both findings were recorded here rather than fixed, per #2865's stated
out-of-scope boundary. FINDING 1 has since been fixed under its own issue
(#5299) and is marked resolved above; FINDING 2 remains open.

## The CI gate

`internal/hooks/failclosed_ledger_test.go` parses the fenced table above and
enforces four properties:

1. **Bidirectional coverage** — every gate in `hooks.PreCommitGates()` has
   exactly one ledger row, and every ledger row names a live gate. A new guard
   landing without a ledger row reds the build, which is what makes this ledger
   an enumeration rather than a snapshot.
2. **Declared enforcement matches code** — each row's `Default enforcement`
   must equal the gate's real `DefaultMode`. Quietly downgrading a `block` gate
   to `warn` reds the build.
3. **Closed fail-mode vocabulary** — each row's `Fail mode` must be exactly
   `fail-closed` or `fail-open`. An undeclared or invented mode reds the build.
4. **Fails closed on a parse of nothing** — zero parsed rows is a failure, not a
   pass, so moving or renaming this file cannot read as green.

**WITNESS.**
```
go test ./internal/hooks/ -count=1 -timeout 120s -run TestFailClosedLedger
```

## Assumptions to recheck

- The ledger's authority is the pre-commit registry plus the hand-enumerated
  observation surfaces. Only the **first** is mechanically bound; the
  observation table is prose and can drift. Binding `repoguard`'s severity table
  and the `dos.toml` reason vocabulary into the same gate is the natural next
  increment.
- "Enforcement fails closed, observation fails open" held everywhere it was
  checked, but the `cmd/fak/guard_*.go` family is large and was sampled, not
  exhaustively read. A counterexample there would invalidate the rule as stated
  and should demote it from a rule to a tendency.
- Findings 1 and 2 assume the current fail-open rationale (never wedge a fleet)
  stays correct. If commit gates ever become the sole barrier for a
  security-critical property, Finding 1 escalates from observability to
  enforcement.
