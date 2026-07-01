package memorystability

import (
	"math"
	"testing"
)

func cycle(i int, divergence float64, stale, poison, probes int) map[string]any {
	return map[string]any{
		"manifest":       "m" + string(rune('0'+i)),
		"divergence":     divergence,
		"stale_reuse":    stale,
		"poison_leakage": poison,
		"probes":         probes,
	}
}

func benign(n int) []map[string]any {
	noise := []float64{0, 0.01, 0, 0.01, 0, 0.01, 0, 0.01}
	out := make([]map[string]any, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, cycle(i, noise[i%len(noise)], 0, 0, 100))
	}
	return out
}

func injectedDrift(n int) []map[string]any {
	out := []map[string]any{cycle(0, 0, 0, 0, 100)}
	for i := 1; i < n; i++ {
		out = append(out, cycle(i, 0.03*float64(i), 0, 1, 100))
	}
	return out
}

func TestMemoryStabilityGovernor(t *testing.T) {
	p := BuildPayload("/repo", benign(8), "", DefaultTau, DefaultBudget, DefaultFloor, 0.05, "")
	if p.Metrics.Frozen || !p.OK || p.Verdict != Stable {
		t.Fatalf("benign report = %+v", p)
	}
	benignSlope := p.Metrics.DriftSlope
	if math.Abs(benignSlope) > 0.01 || p.Metrics.CumulativeDrift > p.Metrics.Budget || p.Metrics.RollbackTo != nil {
		t.Fatalf("bad benign metrics = %+v", p.Metrics)
	}

	inj := injectedDrift(8)
	p = BuildPayload("/repo", inj, "", DefaultTau, DefaultBudget, DefaultFloor, 0.05, "")
	if !p.Metrics.Frozen || p.OK || p.Verdict != Frozen {
		t.Fatalf("injected should freeze: %+v", p)
	}
	if p.Metrics.DriftSlope <= 0.01 || p.Metrics.DriftSlope <= benignSlope+0.01 {
		t.Fatalf("slope did not move enough: injected=%v benign=%v", p.Metrics.DriftSlope, benignSlope)
	}
	if p.Metrics.CumulativeDrift <= p.Metrics.Budget || p.Metrics.RollbackTo == nil || p.Metrics.FreezeCycle == nil {
		t.Fatalf("freeze metrics incomplete: %+v", p.Metrics)
	}
	for _, c := range p.Cycles[1 : *p.Metrics.FreezeCycle+1] {
		if !c.BelowTau {
			t.Fatalf("breaching cycle was not below tau: %+v", c)
		}
	}
	if !p.Metrics.SubThresholdBreach {
		t.Fatalf("sub-threshold breach not flagged")
	}
	if got, want := *p.Metrics.RollbackTo, inj[*p.Metrics.FreezeCycle-1]["manifest"]; got != want {
		t.Fatalf("rollback = %v, want %v", got, want)
	}
	if p.Cycles[*p.Metrics.FreezeCycle].Admitted {
		t.Fatalf("breaching cycle admitted")
	}

	s := StabilityScore(cycle(1, 0, 0, 80, 100))
	if s.Reliability > 0.2 || s.Overall != math.Min(s.Reliability, math.Min(s.Recency, s.Consistency)) {
		t.Fatalf("bad stability score: %+v", s)
	}
	low := []map[string]any{cycle(0, 0, 0, 0, 100), cycle(1, 0, 0, 70, 100)}
	p = BuildPayload("/repo", low, "", DefaultTau, 10, DefaultFloor, 0.05, "")
	if len(p.Forget) == 0 || p.Cycles[1].Admitted {
		t.Fatalf("low-stability cycle not surfaced/refused: %+v", p)
	}
	if DriftSlope([]float64{0, 0.1, 0.2, 0.3}) <= 0 || math.Abs(DriftSlope([]float64{0.1, 0.1, 0.1, 0.1})) >= 1e-9 || DriftSlope([]float64{0.3}) != 0 {
		t.Fatalf("bad slope behavior")
	}
	if empty := BuildPayload("/repo", nil, "", DefaultTau, DefaultBudget, DefaultFloor, 0.05, ""); !empty.OK || empty.Verdict != Stable {
		t.Fatalf("empty trajectory should pass: %+v", empty)
	}
	steep := []map[string]any{cycle(0, 0, 0, 0, 100), cycle(1, 0.10, 0, 0, 100), cycle(2, 0.20, 0, 0, 100), cycle(3, 0.30, 0, 0, 100)}
	drifting := BuildPayload("/repo", steep, "", DefaultTau, 10, DefaultFloor, 0.05, "")
	if drifting.Verdict != Stable || drifting.Metrics.Frozen || drifting.OK || ExitCode(drifting) != 0 {
		t.Fatalf("drifting-in-budget exit contract broken: %+v", drifting)
	}
	if ExitCode(BuildPayload("/repo", injectedDrift(8), "", DefaultTau, DefaultBudget, DefaultFloor, 0.05, "")) != 1 {
		t.Fatalf("frozen store should exit 1")
	}
	if ExitCode(BuildPayload("/repo", nil, "", DefaultTau, DefaultBudget, DefaultFloor, 0.05, "boom")) != 1 {
		t.Fatalf("source error should exit 1")
	}
}
