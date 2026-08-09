package scorecard

import (
	"testing"
	"time"
)

// flowHealthyFacts is a seeded fixture where every flow family clears the pass line, so the
// fold is clean (debt 0, ok) and the headline is the mean of the five family healths.
// mean(0.90,0.80,0.70,0.95,0.85) = 0.84
func flowHealthyFacts() FlowFacts {
	return FlowFacts{
		StartLatency:    hp(0.90),
		CycleTime:       hp(0.80),
		WIPHeadroom:     hp(0.70),
		WIPVisibility:   hp(0.95),
		CommitAtomicity: hp(0.85),
	}
}

// TestFlowNumberMovesWhenAFamilyDegrades is the core witness: the single 0..1 number moves
// when a family degrades in a seeded fixture, so the headline actually discriminates.
func TestFlowNumberMovesWhenAFamilyDegrades(t *testing.T) {
	base, basePresent, _ := Flow(flowHealthyFacts())
	if basePresent != 5 {
		t.Fatalf("present = %d, want 5 (all families seeded)", basePresent)
	}
	if base < 0.839 || base > 0.841 {
		t.Fatalf("baseline delivery-flow = %.4f, want ~0.84", base)
	}

	degraded := flowHealthyFacts()
	degraded.CycleTime = hp(0.20) // one family degrades below the pass line
	got, present, _ := Flow(degraded)
	if present != 5 {
		t.Fatalf("present after degrade = %d, want 5", present)
	}
	if got >= base {
		t.Fatalf("delivery-flow did not move down: baseline %.4f, degraded %.4f", base, got)
	}
	// mean(0.90,0.20,0.70,0.95,0.85) = 0.72
	if got < 0.719 || got > 0.721 {
		t.Fatalf("degraded delivery-flow = %.4f, want ~0.72", got)
	}
}

// TestFlowWorklistIsWorstFirst pins the worklist ordering: every scored family, lowest health
// first, canonical FlowComponents order breaking ties.
func TestFlowWorklistIsWorstFirst(t *testing.T) {
	f := flowHealthyFacts()
	f.WIPHeadroom = hp(0.10) // the worst
	f.CycleTime = hp(0.40)   // second worst (also below the 0.5 pass line)
	_, _, worklist := Flow(f)
	if len(worklist) != 5 {
		t.Fatalf("worklist len = %d, want 5 (every scored family)", len(worklist))
	}
	want := []string{FlowWIPHeadroom, FlowCycleTime, FlowCommitAtomicity, FlowStartLatency, FlowWIPVisibility}
	for i, w := range want {
		if worklist[i].Component != w {
			t.Fatalf("worklist[%d] = %q, want %q (lowest health first); full order %v", i, worklist[i].Component, w, componentOrder(worklist))
		}
	}
	if !worklist[0].InDebt || !worklist[1].InDebt {
		t.Fatalf("the two below-pass-line rows must be InDebt: %+v %+v", worklist[0], worklist[1])
	}
	if worklist[2].InDebt {
		t.Fatalf("a family above the pass line must not be InDebt: %+v", worklist[2])
	}
}

// TestFlowWorklistTieBreaksOnCanonicalOrder pins that equal healths fall back to the declared
// FlowComponents order, so the render is deterministic rather than map-order dependent.
func TestFlowWorklistTieBreaksOnCanonicalOrder(t *testing.T) {
	f := FlowFacts{StartLatency: hp(0.30), CycleTime: hp(0.30)}
	_, present, worklist := Flow(f)
	if present != 2 {
		t.Fatalf("present = %d, want 2 (only the two seeded families)", present)
	}
	if worklist[0].Component != FlowStartLatency || worklist[1].Component != FlowCycleTime {
		t.Fatalf("tie order = %v, want canonical [start_latency cycle_time]", componentOrder(worklist))
	}
}

// TestFlowNilFamilyIsExcludedNotScoredZero is the load-bearing nil contract: a family with no
// evidence must not drag the headline down as if it were a measured 0.0.
func TestFlowNilFamilyIsExcludedNotScoredZero(t *testing.T) {
	f := FlowFacts{StartLatency: hp(1.0), CycleTime: hp(1.0)} // three families nil
	got, present, worklist := Flow(f)
	if present != 2 {
		t.Fatalf("present = %d, want 2 (nil families excluded)", present)
	}
	if len(worklist) != 2 {
		t.Fatalf("worklist len = %d, want 2 (only families with evidence)", len(worklist))
	}
	if got < 0.999 || got > 1.001 {
		t.Fatalf("delivery-flow = %.4f, want 1.0 (nil must not score as 0)", got)
	}
}

// TestFlowNoEvidenceFoldsToOne pins the INSUFFICIENT path: nothing known-unhealthy reads 1.0
// and a single healthy placeholder KPI, never a spurious 0/F from an empty slice.
func TestFlowNoEvidenceFoldsToOne(t *testing.T) {
	got, present, worklist := Flow(FlowFacts{})
	if present != 0 || len(worklist) != 0 {
		t.Fatalf("present/worklist = %d/%d, want 0/0", present, len(worklist))
	}
	if got != 1 {
		t.Fatalf("delivery-flow with no evidence = %.4f, want 1.0", got)
	}
	p := ComposeFlow(FlowFacts{})
	if !p.OK || p.Verdict != "OK" {
		t.Fatalf("no-evidence payload must be OK: ok=%v verdict=%q", p.OK, p.Verdict)
	}
	if len(p.KPIs) != 1 || p.KPIs[0].Key != "delivery_flow_evidence" {
		t.Fatalf("want one INSUFFICIENT placeholder KPI, got %d: %+v", len(p.KPIs), p.KPIs)
	}
}

// --- mapper helpers ---

func TestDurationPercentileNearestRank(t *testing.T) {
	h := time.Hour
	samples := []time.Duration{5 * h, 1 * h, 4 * h, 2 * h, 3 * h} // deliberately unsorted
	cases := []struct {
		p    float64
		want time.Duration
	}{
		{0.2, 1 * h},
		{0.4, 2 * h},
		{0.5, 3 * h},
		{0.85, 5 * h},
		{-1, 1 * h}, // clamped to the fastest
		{2, 5 * h},  // clamped to the slowest
	}
	for _, c := range cases {
		if got := DurationPercentile(samples, c.p); got != c.want {
			t.Fatalf("DurationPercentile(p=%v) = %v, want %v", c.p, got, c.want)
		}
	}
	if got := DurationPercentile(nil, 0.5); got != 0 {
		t.Fatalf("empty sample = %v, want 0", got)
	}
	// The input must not be mutated -- a caller reuses one sample across percentiles.
	if samples[0] != 5*h {
		t.Fatalf("DurationPercentile mutated its input: samples[0] = %v, want 5h", samples[0])
	}
}

func TestDurationHealth(t *testing.T) {
	h := time.Hour
	if got := DurationHealth(10*h, 10*h); got == nil || *got != 1 {
		t.Fatalf("hitting the target must read 1.0, got %v", got)
	}
	if got := DurationHealth(20*h, 10*h); got == nil || *got < 0.499 || *got > 0.501 {
		t.Fatalf("twice the target must read ~0.5, got %v", got)
	}
	if got := DurationHealth(5*h, 10*h); got == nil || *got != 1 {
		t.Fatalf("beating the target must clamp to 1.0, got %v", got)
	}
	if got := DurationHealth(0, 10*h); got != nil {
		t.Fatalf("a zero observation is no evidence, got %v", *got)
	}
	if got := DurationHealth(10*h, 0); got != nil {
		t.Fatalf("no target is no evidence, got %v", *got)
	}
}

func TestWIPHeadroomHealth(t *testing.T) {
	if got := WIPHeadroomHealth(3, 5); got == nil || *got != 1 {
		t.Fatalf("under the cap must read 1.0, got %v", got)
	}
	if got := WIPHeadroomHealth(5, 5); got == nil || *got != 1 {
		t.Fatalf("at the cap must read 1.0, got %v", got)
	}
	if got := WIPHeadroomHealth(10, 5); got == nil || *got < 0.499 || *got > 0.501 {
		t.Fatalf("twice the cap must read ~0.5, got %v", got)
	}
	if got := WIPHeadroomHealth(0, 5); got == nil || *got != 1 {
		t.Fatalf("nothing in flight must read 1.0, got %v", got)
	}
	if got := WIPHeadroomHealth(3, 0); got != nil {
		t.Fatalf("no declared cap is no evidence, got %v", *got)
	}
	if got := WIPHeadroomHealth(-1, 5); got != nil {
		t.Fatalf("a negative in-flight count is no evidence, got %v", *got)
	}
}

func TestVisibilityHealth(t *testing.T) {
	if got := VisibilityHealth(8, 2); got == nil || *got < 0.799 || *got > 0.801 {
		t.Fatalf("8 of 10 committed must read ~0.8, got %v", got)
	}
	if got := VisibilityHealth(0, 5); got == nil || *got != 0 {
		t.Fatalf("all-uncommitted in-flight work must read 0.0 (fully invisible), got %v", got)
	}
	if got := VisibilityHealth(0, 0); got != nil {
		t.Fatalf("no in-flight work is no evidence, got %v", *got)
	}
	if got := VisibilityHealth(-1, 1); got != nil {
		t.Fatalf("a negative count is no evidence, got %v", *got)
	}
}

func TestAtomicityHealth(t *testing.T) {
	if got := AtomicityHealth(7, 10); got == nil || *got < 0.699 || *got > 0.701 {
		t.Fatalf("7 of 10 single-commit closures must read ~0.7, got %v", got)
	}
	if got := AtomicityHealth(0, 10); got == nil || *got != 0 {
		t.Fatalf("no single-commit closure must read 0.0, got %v", got)
	}
	if got := AtomicityHealth(0, 0); got != nil {
		t.Fatalf("no closures is no evidence, got %v", *got)
	}
	if got := AtomicityHealth(11, 10); got != nil {
		t.Fatalf("more single-commit closures than closures is incoherent, got %v", *got)
	}
	if got := AtomicityHealth(-1, 10); got != nil {
		t.Fatalf("a negative count is no evidence, got %v", *got)
	}
}

// --- the folded payload ---

// TestComposeFlowCleanPayload pins the clean envelope: debt 0, ok, and the corpus keys the
// control-pane fold reads (value/grade/<debtKey>) plus this card's own headline.
func TestComposeFlowCleanPayload(t *testing.T) {
	p := ComposeFlow(flowHealthyFacts())
	if p.Schema != FlowSchema {
		t.Fatalf("schema = %q, want %q", p.Schema, FlowSchema)
	}
	if !p.OK || p.Verdict != "OK" {
		t.Fatalf("healthy fixture must be OK: ok=%v verdict=%q reason=%q", p.OK, p.Verdict, p.Reason)
	}
	if debt, ok := p.Corpus[FlowDebtKey]; !ok || debt != 0 {
		t.Fatalf("corpus[%q] = %v (present=%v), want 0", FlowDebtKey, debt, ok)
	}
	// The headline is the 0..1 number and equals corpus.value by construction.
	if got := p.Corpus["delivery_flow"]; got != Round3(0.84) {
		t.Fatalf("corpus[delivery_flow] = %v, want %v", got, Round3(0.84))
	}
	if p.Corpus["delivery_flow"] != p.Corpus["value"] {
		t.Fatalf("headline %v must equal corpus.value %v", p.Corpus["delivery_flow"], p.Corpus["value"])
	}
	if got := p.Corpus["components_present"]; got != 5 {
		t.Fatalf("components_present = %v, want 5", got)
	}
	if got := p.Corpus["components_total"]; got != len(FlowComponents) {
		t.Fatalf("components_total = %v, want %d", got, len(FlowComponents))
	}
	// Defects/Soft must marshal as [] not null on every KPI.
	for _, k := range p.KPIs {
		if k.Defects == nil || k.Soft == nil {
			t.Fatalf("KPI %q has a nil Defects/Soft slice: %+v", k.Key, k)
		}
	}
}

// TestComposeFlowBooksOneDefectPerBelowPassLineFamily is the debt contract: debt is the count
// of families below the pass line, ok flips, and each defect is prefixed with its component id
// so per-row debt stays recoverable from the joined reason.
func TestComposeFlowBooksOneDefectPerBelowPassLineFamily(t *testing.T) {
	f := flowHealthyFacts()
	f.WIPVisibility = hp(0.10)
	f.CycleTime = hp(0.25)
	p := ComposeFlow(f)
	if p.OK || p.Verdict != "ACTION" {
		t.Fatalf("two below-pass-line families must red the card: ok=%v verdict=%q", p.OK, p.Verdict)
	}
	if debt := p.Corpus[FlowDebtKey]; debt != 2 {
		t.Fatalf("corpus[%q] = %v, want 2", FlowDebtKey, debt)
	}
	byKey := map[string]KPI{}
	for _, k := range p.KPIs {
		byKey[k.Key] = k
	}
	for _, key := range []string{FlowWIPVisibility, FlowCycleTime} {
		k := byKey[key]
		if len(k.Defects) != 1 {
			t.Fatalf("KPI %q defects = %v, want exactly 1", key, k.Defects)
		}
		if got := k.Defects[0]; len(got) < len(key) || got[:len(key)] != key {
			t.Fatalf("defect for %q must be prefixed with the component id, got %q", key, got)
		}
	}
	if k := byKey[FlowStartLatency]; len(k.Defects) != 0 {
		t.Fatalf("a family above the pass line must book no defect, got %v", k.Defects)
	}
	// The score is 100*health per family, so the composite tracks the headline exactly.
	if k := byKey[FlowWIPVisibility]; k.Score < 9.99 || k.Score > 10.01 {
		t.Fatalf("KPI score = %.4f, want 100*0.10", k.Score)
	}
}

// componentOrder is a test helper for readable failure messages.
func componentOrder(rows []FlowRow) []string {
	out := make([]string, len(rows))
	for i, r := range rows {
		out[i] = r.Component
	}
	return out
}
