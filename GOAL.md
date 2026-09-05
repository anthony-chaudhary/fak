---
loop: goal
witness: "go test -v ./internal/hooks -run TestE2EOverMocks && go test -v ./cmd/fak -run TestValidatePhaseOrderIncludesSmoke && go test -v ./internal/devcmd -run TestCIPreflightRenderSmokeFailure"
budget: { max_iters: 25 }
lane: cmd
---
# Objective
Add real-world smoke testing earlier in the development process and increase testing volume and frequency across multiple development stages (validation, pre-commit, CI preflight, git hooks, and runtime maturity proofs).

# Non-Goals
- Do not edit frozen ABI (`internal/abi`).
- Do not add network dependencies or external model calls to smoke tests (keep them hermetic, offline, <5s).
- Do not create shell/PowerShell scripts (`.sh`, `.ps1`) - implement new tooling in Go.
- Do not commit peer WIP or use `git add -A`.
- Do not branch or force push (`main` only).

# Plan
- [x] 1. Pin objective, non-goals, and plan in GOAL.md and todowrite
- [x] 2. Stage 1: Add `--smoke` phase to `fak validate` (`cmd/fak/validate.go`, `cmd/fak/validate_helpers.go`, `cmd/fak/validate_smoke.go`)
- [x] 3. Stage 2: Wire real-world CLI smoke execution and `dogfood-test` into `Makefile` (`smoke-exec`, `smoke`, `test-fast`, `ci`) and `scripts/ci.ps1`
- [x] 4. Stage 3: Expand `internal/hooks/gate_e2eovermocks.go` with runtime surfaces and smoke trailers
- [x] 5. Stage 4: Add `--smoke` execution to `fak-dev ci-preflight` (`internal/devcmd/ci_preflight.go`)
- [x] 6. Stage 5: Expand `internal/maturity/runtime-proofs.json` with hermetic CLI smoke proofs for core operational lanes
- [x] 7. Stage 6: Update documentation and skills (`AGENTS.md`, `CONTRIBUTING.md`, `.claude/skills/verify/SKILL.md`)
- [x] 8. Verify all stages with tests and witness gates
- [ ] 9. Commit and push cleanly by explicit path with DCO and `(fak cmd)` trailer

# Verification Evidence
- **Stage 1**: `TestValidatePhaseOrderIncludesSmoke` in `cmd/fak/validate_smoke_test.go` PASS (0.00s). `fak validate --help` surfaces `-smoke`.
- **Stage 2**: `Makefile` wired with `smoke-exec`, `smoke`, `test-fast`, and `ci` targets. `scripts/ci.ps1` executes real binary CLI smoke tests + `dogfood-test`.
- **Stage 3**: `TestE2EOverMocks` in `internal/hooks/gate_e2eovermocks_test.go` PASS (0.00s). Recognizes `cmd/fak/`, `internal/engine/`, `internal/dogfood/` and smoke witness trailers.
- **Stage 4**: `TestCIPreflightRenderSmokeFailure` in `internal/devcmd/ci_preflight_test.go` PASS (0.00s). `fak-dev ci-preflight --smoke` executes tip binary smoke check.
- **Stage 5**: `TestRealRuntimeWitnessRegistryEveryRowMeetsContract` in `internal/maturity/maturity_test.go` PASS (879/879 subtests). `gateway` wired to real `fak serve --help` CLI execution.
- **Stage 6**: `AGENTS.md` and `.claude/skills/verify/SKILL.md` updated with real-world smoke test earlier in process.

# Scratch / last-refusal
All implementation and verification stages completed cleanly.
