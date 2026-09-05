---
parent_goal: goals/GOAL-agent-centric-mlx-mac-observability.md
sub_step: 2_macobs_package_implementation
witness: "go test -v ./internal/macobs/..."
target_files:
  - internal/macobs/doc.go
  - internal/macobs/types.go
  - internal/macobs/hardware_darwin.go
  - internal/macobs/hardware_other.go
  - internal/macobs/mlx_metrics.go
  - internal/macobs/headroom.go
  - internal/macobs/analyzer.go
  - internal/macobs/macobs.go
---
# Sub-Goal Objective
Implement the `internal/macobs` leaf package according to the architectural specification from Sub-Goal 1.
Requirements:
1. `doc.go` and `types.go`:
   - Schema `fak.macobs.v1`
   - Closed sets for `ActionVerdict`, `BottleneckType`, `ThermalState`, `PowerSource`
   - Telemetry structs: `HardwareTelemetry`, `MLXServingTelemetry`, `HeadroomTelemetry`, `PrefixCacheTelemetry`, `AnalysisReport`, and composite `Snapshot`
2. `hardware_darwin.go` & `hardware_other.go`:
   - Fail-soft execution with timeout isolation for `/usr/sbin/ioreg`, `/usr/sbin/sysctl`, `/usr/bin/vm_stat`, and `/usr/bin/pmset`
   - XML plist parsing for `IOAccelerator` (`Alloc system memory`, `In use system memory`, `Device Utilization %`, `Renderer Utilization %`, `recoveryCount`)
   - Sysctl parsing for `hw.memsize`, `iogpu.wired_limit_mb` (or fallback `iogpu.wired_mem_limit`), `vm.swapusage`, `kern.memorystatus_level`
   - Paging from `vm_stat` (compressed bytes, wired, free)
   - Thermal and power parsing from `pmset`
   - Portable non-Darwin stubs returning `UNAVAILABLE` provenance and safe zero/nil values
3. `mlx_metrics.go`:
   - Parse MLX Prometheus metrics (supporting both vllm-mlx and mlx-lm)
   - Fallback and direct calculation for TTFT, ITL, KV cache usage, throughput
4. `headroom.go`:
   - Calculate KV bytes per token given model geometry
   - Calculate available KV budget under wired limit and OS reserve
   - Compute `MaxSharedAgents`, `MaxIsolatedAgents`, and `ConcurrencyAdvantage`
5. `analyzer.go`:
   - Implement `Diagnose(...)` returning closed ActionVerdict, primary bottleneck, and concrete remediation
6. `macobs.go`:
   - `Collector` with options (custom endpoints, runner injections for testing)
   - `Observe(ctx)` method returning the complete `Snapshot`
