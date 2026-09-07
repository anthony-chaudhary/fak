---
name: ci-repair
description: Diagnose, isolate, and repair broken CI/CD workflows and red trunk gates across GitHub Actions, spine-invariance, ci-fast, architest DAG rules, structural policies, scorecard freshness, and build breakages. Uses parallel subagents for failure triage, isolated leaf implementation, on-device witness verification, and atomic commit landing. Call when CI turns red, when `fak sync push` or `fak commit --push` is blocked by `COMMITTED_RED`/`CI_BASE_RED`, or when dual-repo synchronization is wedged.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, Write, Edit, Grep, Glob, Task
argument-hint: "[--run <run_id>] [--workflow <name>] [--diagnose-only] [--push]"
metadata:
  opencode: agent-permission
---

# /ci-repair — Triage, Isolate & Repair Red CI/CD Workflows on Trunk

Deterministic, evidence-backed protocol for rapidly converting red CI/CD workflows and broken trunk gates into verified green runs across both public **`fak`** and private **`fak-private`**. Decomposes compound CI failures into atomic root causes, arbitrates disjoint write sets, dispatches parallel subagents for implementation, verifies on-device test witnesses, and lands explicit-path commits without peer collision.

---

## When to Invoke

1. **GitHub Actions Red Alert**:
   Any core CI workflow (`spine-invariance`, `ci-fast`, `ci`, `garden`, `build-matrix`, `security-audit`) fails on `main` or in an active PR.
2. **Push & Synchronization Blockers**:
   - `fak commit --push` or `fak sync push` fails with `COMMITTED_RED`, `CI_BASE_RED`, `TRUNK_CLOSED`, or pre-push hook rejection.
   - `fak sync drain` becomes stuck in exponential backoff because the upstream trunk has a broken commit.
3. **Dual-Repository Wedges (`fak-private`)**:
   `fak-sync repo` fails fast-forward verification because public `fak` does not compile, has broken tests, or fails boundary checks (`fak-boundary check`).
4. **Automated Loop Escalation**:
   Headless super-loops (`super-loop`, `run-it-all-night`, `fleet-wave`) encounter an unrecoverable red trunk and need to heal the mainline before continuing unattended waves.

---

## Non-Negotiable Invariants

1. **One Root Cause, One Issue, One Commit**:
   Never batch unrelated CI fixes into a kitchen-sink commit. Decompose compound failures into distinct concerns (e.g., an architest upward import vs. an adjudicator structural policy vs. a stale scorecard README) and commit each under its own issue and leaf stamp `(fak <leaf>)`.
2. **Witness-First Verification**:
   Never assume a fix works because "the code looks clean." Reproduce the failure first using the exact command CI executes, implement the minimal remedy, verify on-device exit code 0, and cross-validate with an independent subagent before committing.
3. **Tree-Disjoint Worker Boundaries**:
   When dispatching concurrent repair workers, assign each worker a mutually disjoint file set. No two workers may concurrently edit the same package or files on the shared trunk.
4. **Preserve Architectural Invariants**:
   - Never loosen the architest tier table to resolve an upward import; invert the dependency via a registration seam (`Register...`) or push the shared type down a layer.
   - Never suppress policy rules without structural AST deciders; ensure remedy text does not self-refute by matching its own deny regex.
   - Never silently delete non-secret env reads without updating `admittedPostFreeze` or moving them to the config surface.
5. **Safe Synchronization**:
   Always land commits using `fak commit --path` or `fak sweep --apply`. Always push using `fak sync push` (which safely retries transient non-fast-forward races and refuses destructive merges).

---

## Systematic Failure Taxonomy & Remedies

When diagnosing failed runs from `gh run list` and `gh run view <id> --log-failed`, classify errors into one of six canonical categories:

### Category A: Architest & Layered-DAG Violations (`internal/architest`)
- **Upward Import (`TestNoUpwardImports`)**:
  - *Symptom*: `layered-DAG import rule violated: <pkgA> -> <pkgB>. Rule: tier(A) >= tier(B)`.
  - *Remedy*: Invert the dependency. Define an interface or function type in the lower tier (e.g. `type ContentScreener func(body []byte) bool` in `internal/mcpbroker`), provide a `Register...` hook, and wire it from a higher-tier package (e.g. `internal/headroom/admit.go`).
- **Unregistered Engine ID (`TestSingleEngineDriverPerID`)**:
  - *Symptom*: `engine id "X" is registered by [P] in non-test code but is not a declared id in engineDriverRole`.
  - *Remedy*: Add `"X": {"P": "role description"}` to `engineDriverRole` in `internal/architest/architest_test.go`.
- **Off-List Self-Registration (`TestRequestPathLeavesRegistered`)**:
  - *Symptom*: `leaf P self-registers (init() -> abi.Register*) but is neither blank-imported in internal/registrations nor on regOffList`.
  - *Remedy*: Add `"P": true` to `regOffList` in `internal/architest/architest_test.go`.
- **Non-Literal Subprocess Exec (`TestRequestPathInterpreterFree`)**:
  - *Symptom*: `request-path package P calls exec with a non-literal program argument...`.
  - *Remedy*: Add `"P": "justification"` to `interpreterExecAllow` in `internal/architest/architest_test.go`.
- **Missing Third-Party Tool Inventory (`TestThirdPartyEffectsRegisterCoversTree`)**:
  - *Symptom*: `the tree pulls in subprocess_tool "X" and docs/sbom/third-party-effects.json has neither a row nor an exclusion`.
  - *Remedy*: Add `"X"` under `class: "subprocess_tool"` in alphabetical order in `docs/sbom/third-party-effects.json` with `"pin": "host_provided"`.
- **Declared Reason Missing Code Emitter (`TestEveryDeclaredReasonHasAnEmitter`)**:
  - *Symptom*: `dos.toml declares [reasons.X] but no non-test source contains the token`.
  - *Remedy*: Define `const ReasonX = "X"` in the declaring package and wire it into the producer's refusal/status path.
- **Brittle Test Sleeps (`TestNoBrittleSleepsInAuditedPackages`)**:
  - *Symptom*: `brittle time.Sleep synchronization in test; use channels, sync.WaitGroup, or bounded retry/polling loops`.
  - *Remedy*: Replace `time.Sleep(...)` with channel synchronization or `select { case <-time.After(...): }`.

### Category B: Policy & Structural Surface Rules (`internal/adjudicator`, `internal/policy`)
- **Unrecognised Structural Rule (`TestEveryShippedStructuralRuleIsRecognised`)**:
  - *Symptom*: `tool "X" arg "command" ships a deny_regex with no structural decider and no inventory entry`.
  - *Remedy*: Implement a structural AST/command parser in `internal/adjudicator/<rule>.go` (e.g. `isGitPushArgRule`, `commandExecutesGitPush`), wire it in `decide_argpredicates.go`, and register it in `policy_structural_surface_test.go` under `families`.
- **Self-Refuting Remedy Text (`TestEveryShippedDenyRuleNamesARemedy`)**:
  - *Symptom*: `the fix text for regex MATCHES its own deny_regex, and this rule has NO structural decider`.
  - *Remedy*: Add the structural decider to the `decided` boolean in `policy_structural_surface_test.go` so inert mentions in commit messages, greps, or quoted text are admitted without re-tripping the rule.

### Category C: Generated Artifact & Scorecard Freshness (`concept freshness`, `docs/`)
- **Disambiguation Scorecard Drift**:
  - *Symptom*: `concept generated-artifact freshness` fails with `{"verdict":"stale","stale_paths":["docs/concept-disambiguation-scorecard/README.md"]}`.
  - *Remedy*: Regenerate via `go run ./cmd/fak concept generate`, then verify with `go run ./cmd/fak concept freshness --check --json`. Commit under lane `(fak docs)`.

### Category D: Uncommitted Companion WIP / Compiler Breakages (`COMMITTED_RED`)
- **Missing Struct Field / Method**:
  - *Symptom*: `X undefined (type T has no field or method X)`.
  - *Remedy*: Check the working tree for uncommitted companion edits from peer sessions, or implement the missing exported field/method with appropriate doc comments and tests.

### Category E: Ratchet & Debt Ledgers (`internal/envconfiglint`, `CONFIG_NOT_ENV`)
- **Unrecorded Non-Secret Env Read**:
  - *Symptom*: `env read X is not a declared secret; move it to the config surface (CONFIG_NOT_ENV)`.
  - *Remedy*: If the read landed during an unmonitored window, record `"X"` in `admittedPostFreeze` in `internal/envconfiglint/admitted.go` and verify with `go test ./internal/envconfiglint`.

### Category F: Formatting & Line Hygiene (`gofmt`)
- **Unformatted Go Source**:
  - *Symptom*: `gofmt: N file(s) not formatted (run 'gofmt -w .' from fak/)`.
  - *Remedy*: Format the specific affected files with `gofmt -w <paths>`, verify with `gofmt -l <paths>`, and commit via `fak commit --path`.

---

## Step-by-Step Execution Workflow

### Phase 1: Triage & Failure Capture
1. **List recent workflow runs**:
   ```bash
   gh run list --limit 15
   ```
2. **Inspect failed workflow jobs**:
   ```bash
   gh run view <run_id> --log-failed
   ```
   Or examine specific failed job logs:
   ```bash
   gh run view --job=<job_id> --log
   ```
3. **Partition failures**: Group failures into distinct, non-overlapping concerns (Category A through F).

### Phase 2: Advance Ticketing & Arbitration
1. **Create dedicated GitHub issues**:
   For each distinct failure category:
   ```bash
   gh issue create --title "fix(<subsystem>): <concise summary>" --body "Problem: ... Intended outcome: ... Witness: ..."
   ```
2. **Arbitrate lane leases**:
   Verify lane availability with `dos arbitrate`:
   ```bash
   dos arbitrate --lane <lane> --mode exclusive
   ```

### Phase 3: Parallel Subagent Delegation
1. **Assign disjoint write sets**:
   Launch concurrent `worker` subagents for independent failure components.
   - Worker 1: e.g. `internal/mcpbroker/...`, `internal/headroom/admit.go`
   - Worker 2: e.g. `internal/adjudicator/git_push.go`, `internal/adjudicator/...`
   - Worker 3: e.g. `docs/concept-disambiguation-scorecard/README.md`
2. **Execute on-device witness commands**:
   Each worker must execute and return clean witness command output (e.g. `go test -v ./internal/architest -run TestNoUpwardImports`).

### Phase 4: Independent Cross-Validation
1. Dispatch a `cross-validator` or `tester` subagent to execute the full affected test matrix independently before landing.
2. Confirm zero regressions on related packages.

### Phase 5: Atomic Staging, Commit & Push
1. **Lint subject before commit**:
   ```bash
   fak commit --preview --path <p1> --path <p2> -m "fix(<lane>): <msg> #<issue> (fak <lane>)"
   ```
2. **Commit explicit paths**:
   ```bash
   fak commit --path <p1> --path <p2> -m "fix(<lane>): <msg> #<issue> (fak <lane>)"
   ```
3. **Safe sync and push**:
   ```bash
   fak sync push
   ```
   If racing against peer commits, `fak sync push` automatically retries fast-forward rebases up to the budget.

### Phase 6: Post-Push CI Verification
1. Monitor new workflow runs triggered by the push:
   ```bash
   gh run list --limit 5
   ```
2. Watch until `spine-invariance`, `ci-fast`, and `ci` complete green.

---

## Integration with Automated Processes & Loops

To keep unattended fleets moving when CI breaks, other processes interact with `/ci-repair` as follows:

### 1. The `fak sync drain` Recovery Hook
`fak sync drain` queues commits stranded when the upstream trunk is red. Instead of passive exponential backoff that deadlocks on a permanent trunk break:
- On attempt 2 (`attempts >= 2`), `fak sync drain` inspects `gh run list --workflow ci-fast`.
- If the failure is a known deterministic break (e.g., stale concept scorecard or unformatted files), it automatically triggers the deterministic healing pass (`fak concept generate` / `gofmt -w`), commits the fix, and flushes the drain queue.

### 2. `fak recover COMMITTED_RED`
When `fak commit` or preflight encounters `COMMITTED_RED`:
- `fak recover COMMITTED_RED` provides executable recovery:
  ```bash
  fak recover COMMITTED_RED --execute
  ```
  which invokes the `/ci-repair` diagnostics and suggests the exact remediation command.

### 3. Dual-Repo Sync (`fak-sync repo` in `fak-private`)
When `fak-sync repo` is wedged by public trunk compilation errors:
- `fak-sync repo` automatically runs boundary and compilation probes (`fak validate --mine`, `go work sync`).
- If public `fak` is red, it suggests or triggers `/ci-repair` on the public checkout before attempting to fast-forward the private mirror.

### 4. Headless Super-Loops (`run-it-all-night`, `fleet-wave`, `super-loop`)
When a headless coordinator catches an `EXIT_TRUNK_RED` error:
- It checks `dos arbitrate --lane ci`.
- If free, it acquires the `ci` lane lease, executes the bounded `/ci-repair` packet, verifies trunk health, and resumes the overnight issue wave without terminating.
