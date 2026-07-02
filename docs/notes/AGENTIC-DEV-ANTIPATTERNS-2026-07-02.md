# Agentic-dev anti-patterns: the registry, the unbounded score, and the gardening loops - 2026-07-02

This note establishes the initial set of agentic-development anti-patterns observed
in this workspace, a scoring design that is deliberately unbounded, and the gardening
loops that keep both alive. It is the research half of the `internal/antipattern`
spine; the epic tracks the build-out.

Method: three parallel surveys grounded every claim here in this repo's own history —
(1) the scorecard-family mechanics (`tools/scorecard_control_pane.py`, `pkg/scorecard`,
the ratchet), (2) the gardening-loop skeleton (~13 live loops across skills, scheduled
tasks, and `fak garden`), and (3) a sweep of the session-memory store plus `docs/notes/`
audits for observed failure mechanisms (~70 distinct, all with in-tree evidence).
Nothing in the taxonomy is hypothetical.

## 1. What counts as an anti-pattern here

An agentic-dev anti-pattern is a *recurring failure mechanism of agents doing
development work* — not a code smell (the slop/quality cards own those) and not a
one-off bug. Each entry names a mechanism that (a) happened in this workspace at
least once, (b) can recur for any agent under the same pressure, and (c) admits a
guard, a detector, or at minimum a written fence.

The unit of the registry is a **class**: a kebab-case id, a group, a 1-2 line
mechanism, a severity weight, evidence refs, the guards that exist, and a detection
**rung** (§3). Instances are occurrences of a class (a specific commit, a leaked
worktree, a stale ref).

## 2. The initial taxonomy — 10 groups, 43 classes

Severity mirrors the control pane's `GRADE_DEBT` weights: 2 = annoyance/tax,
4 = corrupts a decision or wastes a session, 8 = corrupts trunk, trust, or privacy.
Evidence refs abbreviate: `mem/<x>` = session-memory file, `notes/<x>` = docs/notes.

### A. False completion (the self-report family)

| id | sev | mechanism | existing guard |
|---|---|---|---|
| unwitnessed-claim-commit | 8 | commit subject claims work the diff doesn't back (`test(...)` touching no test file; `fix(...)` on a docs-only diff) | FLEET_REVIEW_GUARD, `dos commit-audit` (advisory seams) |
| false-closed-issue | 8 | issue closed/marked shipped with no commit witness behind it | `tools/issue_closure_audit.py` (TRUE_RESOLVED vs CLAIMED_CLOSED) |
| code-exists-not-acceptance | 4 | "code landed" read as "acceptance met"; checklist items silently unwired | witness-gated epic close-out (EPIC-CLOSEOUT-METHOD) |
| silent-stop-read-as-done | 4 | a worker going quiet treated as success | dispatch_status content sweep (#1276) |
| green-exit-no-work | 4 | exit 0 / LastResult=0 read as evidence of work (18h-dead scheduled task reported 0) | none general; reproduce-exact-invocation discipline |
| hallucinated-improvement-map | 4 | fan-out "next improvements" maps wrong on file/mechanism/magnitude, shipped unverified | adversarial re-verify discipline |
| unstamped-ship-claim | 2 | ship commit without the `(fak <leaf>)` trailer — claim can't bind to a lane, stays NOT_SHIPPED | commitstamp / `dos verify` |

### B. Shared-tree collisions

| id | sev | mechanism | existing guard |
|---|---|---|---|
| peer-sweep-commit | 8 | path-scoped commit sweeps a peer's uncommitted co-resident files under your SHA | diff-stat-before-ship discipline; safecommit (partial) |
| history-rewrite-shared | 8 | amend/rebase/force-push moves or clobbers a peer's commit | gitgate NO_REWRITE (argv floor) |
| off-trunk-escape | 8 | side branch/worktree created to route around a dirty trunk | trunk guard OFF_TRUNK |
| stale-base-deletion | 8 | pathspec commit from a stale base silently drops a peer's pushed block | safecommit checkStaleBaseDeletion |
| shell-laundered-hazard | 8 | git hazard hidden in `bash -c`/`eval`/`$(...)` evades the argv prefilter | gitgate unwrap pass (#823) |
| refactor-split-sweep | 8 | deletion half (tracked) and addition half (untracked) separated by a peer sweep; trunk references undefined symbols | `git add` new files immediately; clean-checkout build |
| unregistered-lane | 4 | leaf ships without a `dos.toml` lane/tree row, so the arbiter can't see collisions and stamps bind to a phantom | `dos lint` (CI) |

### C. Zombies and cleanup debt

| id | sev | mechanism | existing guard |
|---|---|---|---|
| leaked-build-worktree | 2 | killed tick skips `defer cleanup()`; build worktrees accumulate (57 observed) | selfinstall.ReapStaleBuilds; `fak tree-doctor` (unwired) |
| unbounded-ephemera | 2 | nothing prunes the clone's own gitignored scratch (174 MB in 8 days) | `tools/stale_work_watchdog.py` (dry-run cadence) |
| wedged-oneshot-recovery | 4 | recovery task fails after the gate but before its done-marker; retries the same items forever | none — operator review |
| build-dark-gate | 4 | a broken build stops a CI gate (e.g. `-race`) from running; violations accumulate unseen | local `-race` run discipline |
| worktree-green-checkout-red | 4 | untracked sibling file lets local build pass while a clean checkout fails | `git archive HEAD` build check |

### D. Metric gaming and vanity signals

| id | sev | mechanism | existing guard |
|---|---|---|---|
| vanity-metric-steering | 8 | optimizing a self-fulfilling number (cache-hit-rate rises because the prefix grows; "saved tokens" ≈ session length) | steer on verified-progress-per-token; provenance labels |
| provenance-conflation | 8 | upstream/provider value reported as a fak claim without an OBSERVED qualifier | `internal/conflationscore` floor-is-zero test (in `make ci`) |
| scratch-corpus-pollution | 4 | scorecard walks gitignored checkout mirrors; debt inflates ~6x phantom | `GO_EXCLUDE_DIRS` + git-tracked-only corpus |
| silent-snapshot-drift | 4 | committed scorecard snapshot with no `--check` cadence silently diverges from a fresh run | ratchet family (#1278 for the rest) |
| readiness-as-realized | 4 | a readiness multiplier read as a realized gain | cachevalueledger abstain-below-MinGateTurns |
| bounded-metric-chase | 2 | chasing "3x" on a bounded average that cannot 3x; faking a structural probe | honesty; not shipping marginal churn |

### E. Slop-adjacent process waste

| id | sev | mechanism | existing guard |
|---|---|---|---|
| noop-worker-spawn | 4 | backend spawns a real pid that exits after a banner (zero turns), burning slots | dispatch_status stub sweep (#1275 open) |
| dormant-gate | 4 | component complete but wired into no default gate; protection exists only on paper | dormant-gate audit; `dos-lint` target |
| dead-kpi-field | 2 | declared KPI field never populated — advertises a number it doesn't produce | none (compute or delete) |
| churn-for-progress | 4 | shipping trivial/wrong changes to appear productive when no clean lever exists | loop-index honesty; don't ship |

### F. Guard friction and process bypass

| id | sev | mechanism | existing guard |
|---|---|---|---|
| invisible-false-deny | 4 | a wrong refusal is byte-identical to a right one in the journal; only the refused agent knows | guard-appeal / `fak complain` (engine built, verb not wired) |
| silent-workaround | 4 | agent routes around a wall and records nothing; the next agent pays the same tax | `fak blockers post`; refusal vocabulary |
| ungradeable-commit-verb | 2 | subject verb outside the closed set; the audit ABSTAINs | FLEET_MSG_GUARD |
| secret-fixture-pushwall | 4 | detector's own fake fixtures trip push-protection and wedge every trunk push | defang-by-splitting discipline (#1241) |
| guard-blocks-recovery | 2 | the floor refuses the very recipe that repairs its failure mode | documented alternate recipes |

### G. Memory and context rot

| id | sev | mechanism | existing guard |
|---|---|---|---|
| stale-crossref | 2 | `#NNN` / SHA / path refs go stale (pre-migration ids, squash-erased SHAs, flattened prefixes) | BROKEN_LINK gate (md links only); verify-before-cite |
| forked-memory-store | 4 | two divergent stores for one workspace; recall misaims | explicit `store=`; unification proposal |
| frozen-selfreport-memory | 2 | memories are point-in-time self-reports; "shipped" claims rot | re-verify against git before asserting |
| dangling-doc-pointer | 2 | catalogs/skills cite a runbook that is not on disk (`docs/repo-gardening-loops.md`) | none for non-md-link citations |
| compaction-context-loss | 4 | a second uncoordinated manager (harness compaction) rewrites what the first preserved | compactcohere PrefixEvent (detector only, #1133 unwired) |

### H. Liveness and babysitting

| id | sev | mechanism | existing guard |
|---|---|---|---|
| stranded-recovery-queue | 4 | watchdog runs but the queue doesn't drain; zero-counts read as health | R4 drain witness (#2273, not built) |
| false-healthy-absence | 4 | "no signal" presented as "healthy" (silent_workers counted only 0-byte logs) | widened silent threshold (#1276) |
| thundering-herd-fanout | 2 | wide fan-out fires the whole stage at once and trips rate limits | waves-of-~3 discipline |
| load-bearing-dashboard | 4 | a status UI an operator must watch to stay safe — polling institutionalized | no-babysitting doctrine (poll→interrupt) |

### I. Privacy and boundary leaks

| id | sev | mechanism | existing guard |
|---|---|---|---|
| private-leak-public | 8 | private concept/hardware/path crosses into the public tree | scrub_public_copy + PUBLIC_LEAK gate |

### J. Semantic conflation

| id | sev | mechanism | existing guard |
|---|---|---|---|
| term-overload | 2 | one term names different concepts across surfaces (preflight, in-flight) | disambiguation program |
| mechanism-conflation | 2 | adjacent mechanisms equated (tool-side dedup ≠ context-token cut; decision rate ≠ detection quality) | docs; rate+reason-histogram pairing |

## 3. The rung ladder — how a class matures

Every class sits on a five-rung detection ladder. The gardening loop's whole job is
to move classes up it.

- **R0 NAMED** — the class has an id and a mechanism in the registry.
- **R1 EVIDENCED** — ≥1 concrete evidence ref (memory, note, issue, incident).
- **R2 GUARDED** — a gate refuses at least one seam (commit hook, gitgate rule,
  CI floor). The class can still occur through other seams.
- **R3 MEASURED** — a deterministic detector counts live instances; the count is a
  scorecard signal.
- **R4 RATCHETED** — the instance count rides a pinned baseline: hold or fall,
  never rise (the conflation card's floor-is-zero test is the exemplar).

All 43 initial classes enter at R1 (evidence is the admission bar). Roughly half are
already R2 via existing guards; a handful are R3/R4 today through cards that predate
this registry (provenance-conflation is R4; false-closed-issue and noop-worker-spawn
are R3 via the closure/dispatch audits).

## 4. The unbounded scoring system

The scorecard family's own doctrine (agent-readiness's paired numbers, the control
pane's heterogeneous-units critique) gives the design directly. Three axes, two of
them deliberately unbounded, one ratchet-compatible:

- **`antipattern_pressure`** (unbounded, lower is better) — Σ over classes of
  severity × live instance count, from R3+ detectors. No ceiling and no 0-100 clamp:
  a fleet that leaks 500 worktrees should read 500-worktrees bad, not "F". This is
  the headline trend number, the analog of `heaviness_pressure`.
- **`coverage_frontier`** (unbounded, higher is better) — Σ over classes of
  severity × rung. It grows when a new class is admitted (naming a real failure is
  progress) and when a class climbs a rung. Like `experience_frontier`, it is a
  never-done number: the registry is expected to grow forever, so no completion
  percentage exists.
- **`antipattern_debt`** (floored at 0, the ratchet axis) — the count of HARD,
  in-tree-mendable defects only: registry-integrity violations plus tree-state
  detector findings. This is what the control pane folds and the CI ratchet holds.

The load-bearing design rule that falls out of the shared-trunk reality: **HARD debt
must be mendable in the working tree.** Immutable-history counts (e.g. "unwitnessed
test( claims in the last 200 commits") can never be HARD debt — a commit on a
no-rewrite trunk cannot be unshipped, so such a defect would be unfixable and would
red `make ci` for every peer with no mend available. History-window counts are
emitted as OBSERVED soft signals (they feed pressure and trend), never as defects.
This is the registry protecting itself from becoming an instance of
vanity-metric-steering and peer-churn-refill.

Envelope: the card emits the standard `pkg/scorecard` payload (KPI per group, bounded
per-group rung-coverage score for the legacy grade) with `antipattern_debt` as the
control-pane debt key and pressure/frontier as first-class corpus keys.

## 5. The gardening loops

Three loops, all conforming to the six-move skeleton every legit loop here follows
(named + catalogued entry point; pure-read detect; idempotent gated mend;
ground-truth witness; durable ledger; cadence + brake). 18 of 19 catalogued loops
are detect-only today — these stay propose-first too.

1. **registry-garden** (the meta loop) — detect: `fak antipattern-scorecard --json`
   emits a worklist of classes sorted by severity × rung-gap (the biggest unguarded
   severity first). Mend: propose-only — file/refresh the issue for the next rung
   climb, admit newly evidenced classes, retire classes whose mechanism is
   structurally impossible now. Witness: `coverage_frontier` delta between runs.
   Cadence: weekly, via a `control_pane.loops.json` row.
2. **instance-garden** — for R3+ classes, mend live instances from the detector
   worklist (reap the leaked worktrees, fix the dangling pointers, defang the
   fixture). Mechanical/reversible mends only, explicit-path commits; judgment mends
   stay propose-only. Witness: `antipattern_pressure` falls or holds.
3. **intake-garden** — new-class admission from live friction surfaces: wave-harvest
   buckets (QUIET_INCOMPLETE, STRANDED), guard-appeal / `fak complain` records,
   dispatch refusal storms. Each admitted class must arrive with evidence (R1 bar),
   keeping the registry grounded rather than speculative.

## 6. Spine v1 (shipping with this note)

- `internal/antipattern`: the typed registry (43 classes as data), the rung ladder,
  the three-axis fold on `pkg/scorecard`, registry-integrity checks (id shape,
  severity ∈ {2,4,8}, evidence refs resolve on disk, rung consistent with guards),
  a dangling-doc-pointer detector (catalog-cited runbooks must exist), and the
  worklist emission.
- `cmd/fak antipattern-scorecard` shim (`--json`/`--markdown`), control-pane
  registration (`key: antipattern`, `debt: antipattern_debt`), baseline re-pin in
  the same commit (the sota-card precedent), architest tier row, `dos.toml` lane.

Not yet (tracked in the epic): history-window observers (soft signals), host-state
detectors (worktree/ephemera counts via the watchdog), the three loop registrations,
per-class R4 baselines, and the intake wiring.

## 7. Fences

- The registry scores the *development process*, not the code (slop/quality/doc
  cards own content) and not the runtime (guard/gateway cards own the wire).
- A class without evidence is not admitted; a guard listed in the table is a claim
  about a seam, not blanket coverage; rungs are per-class, not per-group.
- `coverage_frontier` rising is not "the fleet got worse" — it usually means the
  registry learned. Pressure is the health trend; frontier is the knowledge trend;
  only debt gates.
