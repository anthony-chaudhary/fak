package gateway

import (
	"time"

	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/metrics"
	"github.com/anthony-chaudhary/fak/internal/stepobs"
)

// stepobs_wiring.go — the PRODUCTION attachment for the fak_engine_kernel_* and
// fak_engine_planner_step_* families.
//
// This file exists because a family with writers and no attachment is a band that can
// never light up (docs/tickets/observability/TICKET-11-*, fak_sched_preempt_*). Every
// seam below is one a real `fak serve` reaches:
//
//   - computetrace.Record       — called unconditionally by the CPU-reference MatMul
//     (internal/compute/cpuref.go) on every GEMM, and by the device backends while
//     compute tracing is enabled. internal/compute is a 339-package blast hub and is
//     deliberately NOT modified: the fan-out lives in the trace package the backends
//     already call.
//   - enginestep phase/decode  — the native serving loop, next to where
//     enginestep.Default is already fed.
//   - metrics.MicroSpanScope   — the microagent host, one terminal leg at a time.
//
// The two bridges live HERE rather than in internal/stepobs on purpose: internal/metrics
// sits below internal/agent, agent reaches stepobs, so stepobs importing metrics would
// close an import cycle in metrics' own test build. Gateway already imports both, and
// it is the package that renders the families, so it is the honest owner of the bridge.

// attachStepObservation wires the sub-kernel and sub-planner step observers. It is
// idempotent: every call installs the same process-level sinks, so repeated server
// construction (and the test binary that constructs many servers) converges on one
// attachment rather than stacking them.
func attachStepObservation() {
	stepobs.Attach()
	enginestep.SetStepObserver(func(_ string, _ int, d time.Duration) {
		// One decode forward IS one model turn step — the leg the closed vocabulary
		// calls "step". enginestep's own PhaseDecode is the request's whole decode
		// loop, so this is the sub-planner detail the phase family cannot name.
		stepobs.Default.ObservePlannerStep(stepobs.StepKindStep, d)
	})
	metrics.SetSpanObserver(func(kind metrics.MicroSpanKind, d time.Duration) {
		stepobs.Default.ObserveEngineStep(string(kind), d)
	})
}
