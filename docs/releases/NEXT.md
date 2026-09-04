# fak vNext (targeting v0.49.0): Work in Progress

This document tracks in-flight work on `main` targeting the upcoming `v0.49.0` release.
It is updated as commits land so that release notes are maintained proactively rather than scrambled at cut time.

- **Projected version:** `0.49.0` (`minor` bump)
- **Base release tag:** `v0.48.0`
- **Commits in flight:** 39

## What changed

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

- Update active benchmark tasks from Qwen3.6 to Qwen3.8 (#11226).
- Bound helper subprocess execution with timeout and procguard kill.
- Fix hook test branch and recover refusal constant.
- PAUSE and YIELD on SYSTEM_COMMIT_HEADROOM instead of crashing (#11304).
- Delegate blockClone release to gitWorktree release.
- Normalize slashes in session ID validation for cross-platform scratch hygiene.
- Add substantive benchmark suite and retire debt.

## Engineering quality and evidence

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
