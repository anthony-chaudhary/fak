---
title: "Core soft locks: coherence locks for fast-moving agent code"
description: "Explores a DOS-shaped lock doctrine for fak: hard locks protect self-grading machinery, soft locks make coherence-bearing edits explicit and witnessed, and ordinary leaf work stays fast."
date: 2026-06-30
---

# Core soft locks

Status: concept note plus shipped Phase 3 note. The staged doctrine below was
written before the gate shipped. As of 2026-07-01, `fak commit` enforces only the
narrow `hard-self` class: a pathspec that touches declared hard-self machinery is
refused with `CORE_SELF_MODIFY` unless an independent maintenance witness claim
is confirmed. Softer classes remain advisory.

## The problem

fak is now a fast-moving, multi-agent tree. The architecture deliberately keeps
ordinary feature work in disjoint leaves, but several small surfaces act as
shared truth for the rest of the repo:

- the ABI and registry seams that every leaf compiles against;
- the adjudicator and policy floor that decide what an agent may do;
- the witness machinery that grades whether work really shipped;
- `dos.toml`, the lane/reason/stamp taxonomy that dispatch and verification read;
- the tier table and arch tests that keep the graph coherent;
- generated issue, route, release, and scorecard contracts that steer workers.

When these surfaces drift, the failure is not "one package has a bug." The rest
of the tree starts coordinating against a false map. That is the kind of
coherence loss a lock should catch.

The balance is hard. Lock too little and agents can self-modify the machinery
that grades them. Lock too much and the project stops evolving, or agents learn
to route around a heavy process. The target is therefore not "make core
immutable." The target is:

> Lock the authority to change coherence-bearing surfaces, not ordinary code.

## Vocabulary

`core` is not a prestige label. A file is core only when other work relies on it
as authority:

- it decides admission, refusal, evidence, or release;
- it defines a stable interface many leaves import;
- it names lanes, reasons, stamps, tiers, maturity rungs, or route classes;
- it can erase or weaken the witness that would catch its own bad edit.

`hard lock` means an in-path floor refuses a write unless the caller is outside
the locked trust loop. fak already has these for the self-grading machinery:
`SelfModifyGlobs`, shell-write self-modify detection, architest self-witness
coverage, git hooks, and exclusive DOS lanes.

`soft lock` means a write is possible, but it must carry an explicit reason,
serializes when needed, and ships with a witness that proves the coherence
contract still holds. A soft lock is closer to a staged gate than a mutex: it
does not prevent change; it prevents unobserved change.

`lock class` is data. A conforming lock should be declared in a small manifest
or `dos.toml` table, not scattered through prose or a one-off script.

## Lessons from DOS

The useful lessons are not "add more process." They are the shapes that kept DOS
from becoming trust-by-comment:

1. **Closed vocabulary.** A refusal names a token with a summary and fix. The
   agent can recover from `COLLISION_RISK` or `STALE_RECALL`; it cannot reliably
   recover from a paragraph of vibes.
2. **Evidence, not claims.** `dos verify`, `dos commit-audit`, and witnessed
   status do not believe a worker's return string. Core-lock exits should use the
   same rule: a lock clears on read-back, not on "I checked it."
3. **Data, not code.** `dos.toml` declares lanes and reasons. The mechanism lives
   in DOS/fak; repo policy is declarative data. Core locks should follow that
   pattern so new protected surfaces do not require new ad hoc tooling.
4. **Disjointness before fan-out.** `dos arbitrate` works because it prices
   collision by file tree before workers launch. Soft locks should be known
   before an agent starts mutating a shared taxonomy, not discovered at commit.
5. **Advisory before enforcement when uncertainty is high.** The transfer
   playbook treats `tree_known=false` as a measured warning with a threshold. A
   new core-lock class should start as warn/report until false positives are
   understood.
6. **No spontaneous refusal from vocabulary alone.** Adding a reason token or
   lock class should not block work by itself. A named floor or explicit check
   must opt into enforcement.
7. **Both lenses.** A good lock is also an optimization: fewer merge collisions,
   fewer stale route decisions, fewer red-trunk repairs, less operator time spent
   finding which shared contract moved.

## Lock classes

An initial ladder can stay small:

| Class | Meaning | Default action | Examples |
|---|---|---|---|
| `hard-self` | Editing this surface can weaken the machinery that catches the edit. | Deny in-agent writes; require external/human or a privileged release path. | `internal/adjudicator/**`, `internal/architest/**`, `internal/shipgate/**`, `dos.toml` witness sections |
| `serial-core` | The surface is valid to edit, but concurrent edits are incoherent. | Acquire an exclusive lane/lock and run the named witness before commit. | `internal/abi/**`, `internal/kernel/**`, `internal/registrations/**`, release metadata |
| `soft-contract` | The file defines a taxonomy or generated contract other agents consume. | Warn or require a contract check when changed. | architest tier table, issue contract schema, route/status schemas, scorecard rungs |
| `shadow-learn` | Suspected core surface whose false-positive rate is unknown. | Log and report only; promote after measured clean history. | newly introduced catalogs, operator-facing control panes |
| `open-leaf` | Ordinary implementation surface. | Existing lane/test rules only. | most `internal/<leaf>/**` code |

The ladder is intentionally asymmetric. Promotion to a stricter class needs
evidence that drift in the surface caused coordination cost. Demotion needs
evidence that the surface no longer acts as authority.

## Candidate protected surfaces

The current repo already has a de facto core set:

| Surface | Current guard | Proposed class | Why |
|---|---|---|---|
| `internal/adjudicator/**` | self-modify glob, architest wiring checks | `hard-self` | The policy decision point can weaken its own floor. |
| `internal/architest/**` | self-modify glob, CI | `hard-self` | It is the outside witness for the architecture contract. |
| `internal/shipgate/**` | self-modify glob, keep-bit tests | `hard-self` | It grades whether a candidate may ship. |
| `dos.toml` | exclusive DOS lane, self-modify glob coverage | `hard-self` for reasons/stamps; `serial-core` for lane additions | It names the coordination and refusal vocabulary. |
| `internal/abi/**` | exclusive lane, root import tests | `serial-core` | It is the root every leaf imports; changes are valid but high-blast-radius. |
| `internal/kernel/**` | policy self-modify coverage, import-purity tests | `serial-core` | It walks the registries and folds verdicts. |
| `internal/registrations/**` | architest self-registration checks | `serial-core` | It wires built-in leaves into the process. |
| policy manifests and loader | policy check, self-modify globs | `soft-contract` to `serial-core` | A bad policy shape can silently widen or narrow authority. |
| issue/dispatch schemas | issue-contract tests, route tests | `soft-contract` | Vague generated issues turn into vague work. |
| scorecard/maturity taxonomies | scorecard checks | `shadow-learn` to `soft-contract` | They steer prioritization and worker routing. |
| system-prompt/base-context spine | design only today | future `hard-self` for spine, `soft-contract` for overlay | The agent must not rewrite the rules that govern its own edits. |

The table is a starting point, not a blanket. A file should not become core just
because it is important. It becomes core when other automated decisions consume
it as truth.

## The soft-lock protocol

A soft-locked edit should have a cheap, predictable path:

1. **Declare the affected lock.** The check derives it from changed paths and
   prints the lock id, class, reason token, and witness requirement.
2. **Acquire or serialize if needed.** `serial-core` edits take an exclusive DOS
   lane or a short TTL lease. `soft-contract` edits usually do not.
3. **State intent in the artifact.** The commit or handoff names why the contract
   changed. This is not a replacement for evidence; it is the human-readable
   index into the evidence.
4. **Run the named witness.** Examples: `go test ./internal/architest`, `fak
   policy --check`, `fak issue contract`, `fak dispatch route` fixture tests,
   `dos commit-audit`, or a docs-specific freshness check.
5. **Commit by path with the right stamp.** The existing commit discipline is
   already the right landing path.
6. **Read back the effect.** A core-lock check should be able to say "the lock
   cleared because witness X passed on diff Y," not merely "the agent said it
   passed."

For `hard-self`, the protocol intentionally stops earlier: the in-agent write is
refused, and the unlock path is a human/release path or a separately witnessed
maintenance path.

## What not to lock

Do not lock ordinary leaf implementation because it feels central this week.
The leaf model is the concurrency engine of the repo. A soft lock over every
interesting package would erase the DOS disjointness win.

Do not lock exploratory notes, benchmark scratch, or experiments unless a
generated route or claim system consumes them as authority. Exploration needs
room to be wrong.

Do not make a lock depend on the editing agent's self-report. That recreates the
self-grading problem with nicer words.

Do not let a lock become a permanent tax without measurement. A lock should have
a false-positive budget, an observed benefit, or a demotion path.

## Implementation sketch

This can be staged without destabilizing the tree:

### Phase 0: document and audit only

Add a read-only report that folds changed paths into candidate lock classes and
prints required witnesses. It exits zero. This is the `tree_known=false` lesson:
measure the shape before enforcing it.

Possible command shape:

```bash
fak core-locks check --since origin/main --json
```

Output should be a closed schema: lock id, class, changed paths, required
witnesses, verdict `ok|warn|refuse`, and reason token.

### Phase 1: declare the lock data

Add a small data table, likely in `dos.toml` because this is coordination policy:

```toml
[[core_locks]]
id = "self-witness-floor"
class = "hard-self"
paths = ["internal/adjudicator/**", "internal/architest/**", "internal/shipgate/**"]
reason = "CORE_SELF_MODIFY"
witness = ["go test ./internal/architest"]

[[core_locks]]
id = "abi-spine"
class = "serial-core"
paths = ["internal/abi/**"]
reason = "CORE_SERIAL_REQUIRED"
witness = ["go test ./internal/abi/...", "go test ./internal/architest"]
```

New reasons would be data-only until a named floor consumes them:

- `CORE_SELF_MODIFY`: edit targets self-grading machinery.
- `CORE_SERIAL_REQUIRED`: edit targets a high-blast-radius shared spine and needs
  an exclusive lane plus witness.
- `CORE_CONTRACT_WITNESS_MISSING`: edit changed a consumed contract without its
  declared read-back witness.
- `CORE_LOCK_UNCLASSIFIED`: changed a lock-like path with no declared owner.

### Phase 2: warn at commit/guard boundaries

Wire the report into `fak hygiene` or the commit preview path first. A warning
should name the exact witness needed. It should not block until the false-positive
rate is known.

### Phase 3: enforce only the small hard set

The first enforcement target should be narrow: surfaces that can weaken their
own witness. Most other classes should remain warn or serial-only until measured.

Shipped maintenance path (2026-07-01): `fak commit` refuses hard-self pathsets
before staging. A privileged maintenance flow may pass
`--core-lock-maintenance-witness <claim>`; the witness resolver must confirm that
claim, and a verified accepted commit records a `corelock-maintenance` decision
with the claim on `refs/notes/fak/decisions`. A self-report that the witness ran
is not enough, and ordinary leaf commits do not enter this path.

### Phase 4: promote by evidence

Promote a `shadow-learn` surface to `soft-contract` only after repeated drift
causes coordination repair, red CI, bad issue routing, or stale verification.
Demote when a surface stops acting as authority.

## Success criteria

A core-lock system is working only if all of these are true:

- ordinary leaf edits are no slower and no more serial than today;
- edits to witness/admission/taxonomy surfaces are explicit, witnessed, and
  path-scoped;
- agents get repairable closed-vocabulary feedback, not vague process prose;
- the operator can see which locks are firing, false-positive rate, and time cost;
- new locks are promoted by evidence and can be demoted;
- no agent can edit the rule that would have caught that edit and then grade the
  result itself.

## Open questions

- Should the source of truth be `dos.toml [core_locks]`, a separate
  `core-locks.toml`, or generated from existing `lanes`, `reasons`, and
  architest tables?
- Should `serial-core` be enforced by DOS lease acquisition, by commit preview,
  or both?
- What is the minimum witness set for a policy-manifest edit: `fak policy
  --check`, round-trip tests, self-modify demo, or all three?
- How should docs that define normative standards be handled? A standards page
  can be more authoritative than code for future agents, but locking all docs
  would be counterproductive.
- What metric proves the lock pays for itself: fewer stale-base deletions, fewer
  issue-routing repairs, fewer architest red trunks, or lower operator review
  time?

## The doctrine

Hard locks protect the parts of fak that would let an agent grade its own
homework. Soft locks protect shared truth without freezing the project. The
default remains leaf velocity; the exception is a coherence-bearing surface whose
change would move the map underneath other agents.

That is the balance: make dangerous authority changes visible and witnessed, but
keep ordinary implementation work cheap enough that nobody has a reason to route
around the system.
