---
parent_goal: goals/GOAL-agent-centric-mlx-mac-observability.md
sub_step: 3_cli_macobs_tooling
witness: "go test -v ./cmd/fak -run TestRunMacObs && go test -v ./internal/devindex -run 'TestVerbTierCoverageIsTotal|TestVerbManifestCoversEveryDispatcherVerb'"
target_files:
  - cmd/fak/macobs.go
  - cmd/fak/macobs_test.go
  - cmd/fak/main.go
  - internal/devindex/tiers.go
  - internal/devindex/verbs.go
  - internal/devindex/devreuse.go
---
# Sub-Goal Objective
Implement the `fak macobs` CLI verb and wire it into the repository dispatch tables and index registries.
Requirements:
1. `cmd/fak/macobs.go`:
   - `cmdMacObs(argv []string)` and `runMacObs(stdout, stderr io.Writer, argv []string) int`
   - Flags:
     - `--json`: emit canonical `fak.macobs.v1` JSON envelope
     - `--check-headroom`: emit concise agent-focused admission report (recommended agents, headroom MB, verdict, remediation)
     - `--agents <N>`: target concurrency (default 4)
     - `--prefix-tokens <N>`: shared prefix tokens (default 4096)
     - `--tail-tokens <N>`: private agent turn tokens (default 2048)
     - `--mlx-endpoint <URL>`: override MLX server / metrics URL
     - `--watch`: watch mode with `--interval`
   - Formatted human-readable output highlighting:
     - Apple Silicon memory (physical, wired limit, Metal allocated, swap)
     - MLX serving telemetry (requests running/queued, KV cache %, TTFT, tok/s)
     - Subagent concurrency headroom (shared prefix vs isolated, concurrency advantage)
     - Verdict and remediation
2. Register `macobs` in:
   - `cmd/fak/main.go`
   - `internal/devindex/tiers.go` (`TierDev`)
   - `internal/devindex/verbs.go` (`canonicalVerbs`)
   - `internal/devindex/devreuse.go` (`devSpecificCommands`)
3. `cmd/fak/macobs_test.go`:
   - Unit tests verifying `--json`, `--check-headroom`, help, and invalid arguments.
