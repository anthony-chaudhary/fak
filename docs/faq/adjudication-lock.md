---
title: "fak FAQ — The lock — how adjudication works"
description: "How fak adjudicates tool calls with default-deny policy, closed refusal reasons, capability manifests, caching, and explicit human approvals."
---

# The lock — how adjudication works

Part of the [fak FAQ](../FAQ.md) — the essentials and every other theme are
indexed there.

Memory engineering is the discipline of deciding what an AI agent remembers, where that memory lives, how a memory is re-verified before it is trusted again, and when it is provably forgotten — each decided by an inspectable mechanism rather than by the model's in-context judgment. Prompt engineering shaped one completion and context engineering shapes one session's window; memory engineering governs what an agent carries across sessions. `fak` implements the four decisions at the same kernel boundary that adjudicates tool calls: a truth-duration write gate that defaults ephemeral facts to expire rather than persist, a structured promotion record so `fak memory explain-promotion` answers "why is this fact in memory" from a ledger captured at write time rather than a model's story, verified recall (`fak memory recall`) that re-checks a note's claims against ground truth at read time and withholds what fails, ECC-style syndrome and scrub integrity over persisted memory images, and bit-exact span eviction so a removed memory is provably gone (`max|Δ| = 0`). The one-line test: if a memory system cannot answer "why is this fact in memory", "is it still true", and "can you prove it is gone", it has memory features, not memory engineering. The definitional page is docs/explainers/memory-engineering.md.

## The lock — how adjudication works

The capability floor, end to end: the path a proposed tool call takes through the kernel, the closed refusal vocabulary it answers with, and exactly what the floor does and does not bound.

## What exact path does a proposed tool call take through the kernel?

A proposed tool call hits the in-process vDSO fast-path first; on a miss the kernel folds the adjudicator chain to one verdict, and only an allowed call is ever enqueued. There is no spawned hook and no inter-process call on the decide path. `Submit` consults the vDSO, and a hit returns `Allow by=vdso` with no adjudication and no engine call. On a miss, `decide()` folds the registered chain to a single verdict and routes it, and a denied call is never enqueued for execution. Reaping a result runs the separate result-side admission chain.

## What does "default-deny" actually mean in fak's adjudicator?

Default-deny means any tool you did not explicitly allow-list is refused, regardless of context or injected text. A zero (empty) policy is the fail-closed floor: nothing is allowed, so every call returns `DEFAULT_DENY`. The fold reinforces this structurally — an empty chain folds to `Deny/DEFAULT_DENY by="empty-policy"`, and a chain where every rung defers folds to `Deny/DEFAULT_DENY by="all-defer"`. The default-deny-on-empty-policy guarantee is pinned by the `TestFoldDefaultDenyEmptyPolicy` witness.

## What is the closed refusal vocabulary, and what are the exact reason codes?

`fak` refuses only with one of 17 codes from a closed vocabulary, never free text: `DEFAULT_DENY`, `POLICY_BLOCK`, `SELF_MODIFY`, `LEASE_HELD`, `TRUST_VIOLATION`, `MALFORMED`, `MISROUTE`, `RATE_LIMITED`, `SECRET_EXFIL`, `UNWITNESSED`, `OVERSIZE`, `UNKNOWN_TOOL`, `RESULT_SECRET_DISCOVERED`, `SECRET_REDACTED`, `SHELL_DIALECT`, `PII_REDACTED`, and `PII_EXFIL` (plus `NONE`, which is not a refusal). The set is the source of truth in `internal/abi/reasons.go` and is the same vocabulary the policy loader validates against. It is forward-compatible: an unknown code renders as `REASON_<n>` rather than panicking, so a newer rung can add a code without breaking an older reader.

## How do allow, allow_prefix, and deny work in a policy manifest?

`allow` is an exact tool-name match, `allow_prefix` matches a tool name by prefix, and `deny` is a provable refusal by name whose value is a closed-vocabulary reason code. In the manifest these are the fields `allow`, `allow_prefix`, and `deny` (a map of tool name to reason name), and the default `allow_prefix` family is the read-only set `read_ get_ search_ list_ lookup_ find_ calc`. A loaded manifest replaces the floor rather than merging into a built-in default, so the manifest you load is the whole floor.

```bash
fak policy --dump > floor.json   # emit the full default to edit
fak policy --check floor.json    # validate + print the admitted floor
```

## What is the difference between fail_closed and admit_and_log posture?

`fail_closed` (the default, zero value) refuses anything not allow-listed, while `admit_and_log` downgrades only a LOW-RISK, READ-SHAPED default-deny to an allow while recording what it would have denied. Under `admit_and_log` a downgraded call carries `Meta{posture:"admit_and_log", would_deny:"DEFAULT_DENY"}` so the would-be refusal is still auditable. It is not a blanket open door: explicit denies, self-modify, arg-rule violations, and any write-shaped default-deny still fail closed. The read-shaped test is name-based and conservative, and caller-supplied metadata cannot widen authority.

## Why is a policy refusal an HTTP 200 instead of a 4xx error?

A refusal is a successful turn carried as a verdict value, so `fak serve` returns `200 OK` with the verdict in the response body and never a non-2xx for a policy refusal. Over the gateway, `adjudicateProposed` keeps ALLOW and TRANSFORM calls, drops the rest, and records each decision in the `fak` response extension as a per-call `ToolAdjudication`/`WireVerdict`; for clients that do not read that extension, a deny summary is also written in-band. HTTP error statuses are reserved for malformed requests, auth failures, and upstream faults, so a client never treats "the kernel said no" as an exception.

## What does "deny is a value, not an error" mean inside the kernel loop?

When the kernel denies a call it produces a structured Result the next loop turn consumes in-band, rather than raising an error. The `DenyResult` carries `Status=StatusError, Outcome=OutcomeCommitted` plus `Meta{verdict:"deny", reason, disposition, by}` and a bounded witness containing only the offending set. The disposition tells the loop what to do next: malformed and misroute denies are `RETRYABLE`, rate-limit and lease denies are `WAIT`, self-modify and trust denies are `ESCALATE`, and everything else is `TERMINAL`.

## Does the adjudication floor bound a tool's arguments, or only its name?

The capability floor bounds tool *names* structurally; it does not bound the resolved *effect* of an allow-listed tool's arguments. An allow-listed `send_email` with attacker-chosen recipients is not stopped by the floor itself, so the guidance is to keep exfil-shaped tools off the allow-list entirely. `fak` does add arg-level predicates (issue #9) that can restrict an allowed tool by inspecting one decoded argument string, but those inspect a single value, not the resolved effect, and a satisfied predicate never *grants* an allow. Argument-scoped capabilities (path, host, amount as first-class constraints) are roadmap, not shipped.

## How do arg-level predicates restrict an allow-listed tool?

Arg-level predicates (issue #9) are RESTRICT-ONLY rules keyed on a tool name plus an argument value, evaluated after name-deny and self-modify but before the affirmative allow, so an allow-listed tool with a malicious argument is refused at the floor instead of being waved through to detection. There are three kinds: `allow_glob` (positive — the value must be a non-escaping path under a glob, and a missing arg or `../` escape fails closed), `deny_regex` (negative RE2 match), and `max_bytes` (a string over N bytes is denied). A violation denies with the rule's reason (default `POLICY_BLOCK`) and a bounded witness of the bound that was violated, never the argument value itself.

## How does fak handle a malformed or wrongly-shaped tool call?

Malformed calls are routed by two early rungs: grammar repair can rewrite a repairable call into a `Transform`, and an unrepairable one is denied with `MISROUTE` (a retryable disposition). The grammar rung defers well-formed calls, repairs malformed-but-repairable ones (a positional-to-named zip when arity matches, or an alias rename), and fails *open* with a `Defer` when no grammar exists for the tool so it never over-refuses. Below it, the preflight ladder does a static JSON parse (rung-0) and a schema required-fields and types check (rung-1); a failure there denies with `MALFORMED`.

## How does the adjudicator chain combine multiple rungs into one verdict?

The chain folds to the single most-restrictive verdict, so a stricter rung can only tighten the outcome, never loosen it. Each verdict kind has a fold rank — Allow=0, Defer=1, Transform=2, Quarantine=3, RequireWitness=4, Deny=100 — and the highest non-defer rank wins; an unknown registered kind folds to 100, which is fail-closed. The default rungs are grammar repair, the preflight ladder, and the authoritative adjudicator monitor. Because the fold is order-independent, a rung's rank only orders the work, not the result.

## In what order does the adjudicator monitor decide a single call?

Inside the authoritative monitor the decision walks a fixed order: explicit name-deny first, then self-modify on a path argument, then self-modify on a shell or command string, then arg-level predicates, then redaction transforms, then the affirmative allow or allow_prefix, and finally the default-deny catch-all. This ordering is why a malicious argument on an allowed tool is refused at the floor rather than reaching detection: the arg predicates run before the affirmative allow. The affirmative allow is the last thing consulted before the default-deny, so anything not explicitly permitted falls through to a refusal.

## Why does fak deny a write-shaped shell command that touches a guarded path?

`fak` refuses a write-shaped command that targets a guarded glob with a `SELF_MODIFY` denial, because an agent editing its own policy or harness is the self-grading-homework failure the rung exists to stop. The shell-path form fires only when a command contains a guarded glob *and* a write verb or redirect; the write detection is a deliberately over-broad substring floor — covering `sed -i`, `tee`, `cp`/`mv`, `git apply/checkout/restore`, interpreter eval flags, `>`/`>>`, and many more — not a real shell parser. A plain read of a guarded file stays allowed, and the bias is intentional: a false refusal is cheap, while a false allow here is the failure mode the rung exists to stop.

## What happens if my policy manifest has a typo or an unknown field?

`fak` fails loud on a bad manifest rather than silently falling back to a more permissive default. The loader uses strict field decoding, so a typo like `allows` for `allow` is a hard error (`json: unknown field "allows"`); an unknown deny reason errors with the list of offenders plus the full valid vocabulary; and an unknown posture, bad regex, or malformed arg rule each hard-error. On startup `fak serve` propagates that error as a fatal failure, so there is no silent fallback to a more permissive floor. A round-trip is exact: `--dump` piped into `--check` validates unchanged.

## How do I check what verdict a single tool call gets without running a server?

`fak preflight` is the per-call oracle: it runs the adjudication rungs over one tool call and prints `verdict=… reason=… by=…` with no dispatch and no server. Pass the tool name, its arguments as JSON, and optionally a policy file; `--explain` or `--json` dumps the per-rung decision trace. This is the offline way to prove a policy refuses what you expect before you wire anything live.

```bash
fak preflight --tool refund_payment --args '{}' --policy floor.json --explain
```

## Does the vDSO fast-path skip the security check on a cache hit?

No, a vDSO hit is sound by construction: a cache hit is defined to equal a fresh call, so serving it without re-adjudicating does not loosen the floor. The fast-path serves only repeat decisions that are pure functions of their inputs or are bound to the current world-version, and the write-shape veto is name-based and re-checked rather than trusted from an annotation. A write-shaped completion bumps the world-version so stale entries cannot be served. The kernel counts `VDSOHits` separately, so the hit ratio is observable on `/metrics`.

## What does the kernel do when a policy injects its own per-kernel adjudicator chain?

By default the kernel folds the process-global adjudicator registry, but `WithAdjudicators` lets you inject a per-kernel chain so concurrent kernels can run independent policies. An empty or nil injected chain is a no-op fallback to the global registry; it never silently installs a default-deny-all in place of your real policy. The fold semantics are identical either way — most-restrictive-wins over whatever chain is in effect — so independent policies coexist without one kernel's floor leaking into another's.

## Why is running the adjudication check in-process load-bearing rather than just fast?

