package opttarget

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/rsiloop"
)

// Measurer resolves the two IMPURE seams a target needs — the baseline probe and
// the per-candidate measure — for a given OptTarget. It is the one part of a
// target that cannot be pure data: measuring forks an isolated worktree and runs
// a real probe/suite/truth check. A registry binds a target's declared Measurer
// name to an implementation (Phase 1); tests and the worktree adapter bind one
// directly. Compile wires everything else (metric name, direction, the candidate
// grammar) from the declaration alone.
type Measurer interface {
	// Baseline measures the target's KPI on its BaselineRef and returns the
	// resolved ref every candidate must fork from.
	Baseline(t OptTarget) (metric float64, baselineRef string, err error)
	// Measure applies one candidate in isolation and returns its DERIVED witness
	// (metric + suite-green + truth-clean), every field measured, never supplied.
	Measure(t OptTarget, c rsiloop.Candidate) (rsiloop.Measurement, error)
}

// Compile lowers a DECLARED OptTarget into a rsiloop.Harness driven by m. The
// harness's MetricName/LowerBetter/BaselineRefName and its Candidates come PURELY
// from the declaration; only the Baseline/Measure seams come from the measurer.
// The returned harness drives rsiloop.Run with the unchanged non-forgeable
// keep-bit — a target is declared, not coded, and earns the same gate. A
// malformed target (Validate) or a nil measurer is refused.
func Compile(t OptTarget, m Measurer) (rsiloop.Harness, error) {
	if err := t.Validate(); err != nil {
		return rsiloop.Harness{}, err
	}
	if m == nil {
		return rsiloop.Harness{}, fmt.Errorf("opttarget %q: nil Measurer", t.Name)
	}
	return rsiloop.Harness{
		MetricName:      t.Metric,
		LowerBetter:     t.lowerBetter(),
		BaselineRefName: t.BaselineRef,
		BaselineMetric:  func() (float64, string, error) { return m.Baseline(t) },
		Candidates:      t.candidates,
		Measure:         func(c rsiloop.Candidate) (rsiloop.Measurement, error) { return m.Measure(t, c) },
	}, nil
}

// HarnessMeasurer adapts an existing rsiloop.Harness (e.g. one from
// rsiloop.NewWorktreeHarness — the REAL worktree/probe/suite/truth seams) into a
// Measurer, so a DECLARED OptTarget drives the same live loop the hand-wired
// harness does. Baseline and Measure delegate to the wrapped harness's seams; the
// candidate GRAMMAR still comes from the OptTarget. This is the bridge from "a
// target is data" to "a target runs the unchanged keep-bit": the candidates are
// declared, the measurement is the production run.
//
// Scope (Phase 0): a wrapped worktree harness rewrites its OWN configured tunable
// literal, so a HarnessMeasurer is faithful for a target whose Site IS that
// tunable (the cache-size demo). Lowering an ARBITRARY Site's (path, const)
// rewrite into the worktree seam is the named Phase 0.1 follow-on; until then a
// general Site is carried as declared metadata, not yet rewritten.
type HarnessMeasurer struct {
	H rsiloop.Harness
}

// Baseline delegates to the wrapped harness's baseline probe.
func (h HarnessMeasurer) Baseline(OptTarget) (float64, string, error) {
	return h.H.BaselineMetric()
}

// Measure delegates to the wrapped harness's per-candidate measurement.
func (h HarnessMeasurer) Measure(_ OptTarget, c rsiloop.Candidate) (rsiloop.Measurement, error) {
	return h.H.Measure(c)
}
