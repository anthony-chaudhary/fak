# fak vNext (targeting v0.49.0): Work in Progress

This document tracks in-flight work on `main` targeting the upcoming `v0.49.0` release.
It is updated as commits land so that release notes are maintained proactively rather than scrambled at cut time.

- **Projected version:** `0.49.0` (`minor` bump)
- **Base release tag:** `v0.48.0`
- **Commits in flight:** 75

## What changed

- Wire KVStore, CompactDenySummary, and execution verification.
- Wire GoalAnchor and StabilizePromptPrefix into loop and upstream.
- Wire MMU compactor sliding window and positive compaction.
- Wire TieredEvaluator convenience methods onto Manifest.
- Wire lifecycle FSM, recovery audit, and AST write extractors.
- Add fak self alias and harden repo discovery with version fallback.
- Restore living release draft tracking and fresh NEXT.md.
- Add host-wide build concurrency governor.
- Debounce headroom exhaustion and park goals on host commit floor.
- Add sweep auto-archive, gpt-6-astra codex defaults, and terminal headroom recovery.
- Support WIP fence tag isolation in buildcheck.
- Support extended retention for gpt-6 and astra models.
- Add EnvRunner interface and RunWithEnv on GitRunner.
- Wire GPT-6 Astra aliases into default account roster.
- Classify GPT-6 Astra and aliases as Tier 1 frontier.
- Support GPT-6 Astra flagship default and alias normalization.
- Implement reversibility classification and shell ast parser.
- Implement process suspend/resume and commit hooks.
- Improve preflight checks and link verification.
- Add check_docs_links wrapper for front-door and notes link auditing.
- Enforce production database connection fence.
- Support working set trimming and physical memory reclamation.
- Infer ownership when scope empty and lane active.
- Enforce scratch file quota and reap session scratch.
- Refuse peer wip collision and allow quarantined orphans.
- Support functions.exec_command surface and verify posture transitions.

## Reliability and correctness

- Consolidate duplicate release notes items by scope.
- Ignore NEXT.md sync commits in release_next filter.
- Enforce conflict banner and silent drop merge gates (#11306).
- Bound launcher and orgdebt subprocesses with context cancellation.
- Preserve active posture during floor reload.
- Reject conflict banners and silent drop merges (#11306).
- Update active benchmark tasks from Qwen3.6 to Qwen3.8 (#11226).
- Bound helper subprocess execution with timeout and procguard kill.
- Fix hook test branch and recover refusal constant.
- PAUSE and YIELD on SYSTEM_COMMIT_HEADROOM instead of crashing (#11304).
- Delegate blockClone release to gitWorktree release.
- Normalize slashes in session ID validation for cross-platform scratch hygiene.
- Add substantive benchmark suite and retire debt.

## Engineering quality and evidence

- Add 13-feature-island wiring audit test (#11309).
- Add production benchmarks for cryptographic hash chain and row verification.
- Add benchmark suite and prune comment density.
- Add benchmark suite and clean comment bloat.
- Add substantive benchmark suite for plan adjudication.
- Add test hygiene scanner against hardcoded ports and brittle sleeps (#11307).
- Replace timing sleeps with os.Chtimes for mtime invalidation (#11307).
- Replace hardcoded ports and drain sleeps (#11307).
- Replace brittle sleep polling with bounded pollUntil helper (#11307).
- Use dynamic port for mock base URL (#11307).
- Replace sleep polling with reader onActive callback notification (#11307).
- Replace sleep with deterministic watermark stats polling (#11307).
- Replace timing sleeps with barrier channels and wait hooks (#11307).
- Replace static port bindings with dynamic ephemeral ports (#11307).
- Use dynamic port in gateway plan resolution test (#11307).
- Use dynamic port allocation in benchmark options (#11307).
- Eliminate hardcoded redis port and replace sleep with pollUntil (#11307).
- Replace time.Sleep with runtime.Gosched in concurrency test (#11307).
- Replace time.Sleep with channel and waitgroup synchronization (#11307).
- Use cancellable wait loop and dynamic port in opencode test (#11307).
- Isolate Windows fault domain job object binding in subprocess.
- Make headroom debounce test synchronization deterministic.
- Verify conflict templates, conflict markers, and silent drop merges (#11306).
- Add substantive benchmarks for milestone doc operations.
- Add substantive benchmarks for recovery planning.
- Reconcile Qwen3.8 GPU Direct overflow provenance and changelog (#11226).
- Add substantive benchmarks for process reaping.
- Add substantive benchmarks and clean comments.
- Update documentation guides and scorecards.
- Update examples readme.
- Verify test compilation binary containment in native serve loop.
- Ratcheted description budget ceiling to 34155 bytes.
- Tighten curl pipe to shell regex in command boundary smoke policy.
- Update synthetic producer inventory for loop wire prompt components.
- Disable core.autocrlf in drill test repo setup.
- Refresh scorecards, skill budgets, and workflow branch audit.

## Upgrade and breaking changes

- No manual migration required unless specified above.
