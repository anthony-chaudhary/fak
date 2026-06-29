package modelroute

import (
	"reflect"
	"testing"
	"time"
)

// manifestForOutcomes is a small two-rule manifest used by the outcome tests: a
// tool_call rule and a high-complexity rule, with a fail-closed default. Routing
// distinct subjects through it produces decisions on distinct (aspect,rule) keys.
func manifestForOutcomes() Manifest {
	return Manifest{
		Version: Version,
		Default: Plan{Members: []Member{{Model: "default", Role: "primary"}}},
		Rules: []Rule{
			{
				Name:  "tool-writes",
				Match: Match{Aspect: AspectToolCall, Tool: "write_file"},
				Plan:  Plan{Members: []Member{{Model: "small"}}},
			},
			{
				Name:  "hard",
				Match: Match{MinComplexity: ComplexityHigh},
				Plan:  Plan{Members: []Member{{Model: "large"}}},
			},
		},
	}
}

// TestOutcomeAggregatePerAspectRule witnesses the core deliverable: outcomes
// recorded against decisions aggregate per (aspect,rule) into a mean
// cost/latency/quality, the means are over the recorded samples only, and the
// (aspect,rule) buckets stay distinct.
func TestOutcomeAggregatePerAspectRule(t *testing.T) {
	m := manifestForOutcomes()

	tool := m.Route(Subject{Aspect: AspectToolCall, Tool: "write_file"})
	hard := m.Route(Subject{Aspect: AspectStep, Complexity: ComplexityHigh})

	if tool.RuleName != "tool-writes" || hard.RuleName != "hard" {
		t.Fatalf("setup: routes did not match expected rules: tool=%q hard=%q", tool.RuleName, hard.RuleName)
	}

	var j OutcomeJournal
	// Two outcomes on the tool route: costs 0.10 and 0.30 (mean 0.20), latencies
	// 100ms and 300ms (mean 200ms), quality 0.8 and 1.0 (mean 0.9).
	j.Record(m.Version, tool, Outcome{Cost: 0.10, Latency: 100 * time.Millisecond, Quality: 0.8})
	j.Record(m.Version, tool, Outcome{Cost: 0.30, Latency: 300 * time.Millisecond, Quality: 1.0})
	// One outcome on the hard route: cost 2.0, latency 1s, quality 0.5.
	j.Record(m.Version, hard, Outcome{Cost: 2.0, Latency: time.Second, Quality: 0.5})

	agg := j.Aggregate()

	if agg.Total != 3 {
		t.Fatalf("Total: got %d, want 3", agg.Total)
	}
	if len(agg.ByKey) != 2 {
		t.Fatalf("ByKey buckets: got %d, want 2 (one per aspect,rule)", len(agg.ByKey))
	}

	toolKey := AspectRuleKey{Aspect: AspectToolCall, Rule: "tool-writes"}
	ts, ok := agg.ByKey[toolKey]
	if !ok {
		t.Fatalf("tool bucket %+v missing from aggregate", toolKey)
	}
	if ts.Count != 2 {
		t.Errorf("tool Count: got %d, want 2", ts.Count)
	}
	if ts.MeanCost != 0.20 {
		t.Errorf("tool MeanCost: got %v, want 0.20", ts.MeanCost)
	}
	if ts.MeanLatency != 200*time.Millisecond {
		t.Errorf("tool MeanLatency: got %v, want 200ms", ts.MeanLatency)
	}
	if ts.MeanQuality != 0.9 {
		t.Errorf("tool MeanQuality: got %v, want 0.9", ts.MeanQuality)
	}

	hardKey := AspectRuleKey{Aspect: AspectStep, Rule: "hard"}
	hs, ok := agg.ByKey[hardKey]
	if !ok {
		t.Fatalf("hard bucket %+v missing from aggregate", hardKey)
	}
	if hs.Count != 1 {
		t.Errorf("hard Count: got %d, want 1", hs.Count)
	}
	if hs.MeanCost != 2.0 || hs.MeanLatency != time.Second || hs.MeanQuality != 0.5 {
		t.Errorf("hard stats: got cost=%v lat=%v q=%v, want 2.0 / 1s / 0.5", hs.MeanCost, hs.MeanLatency, hs.MeanQuality)
	}

	// The two routes must NOT bleed into each other's bucket.
	if reflect.DeepEqual(toolKey, hardKey) {
		t.Fatal("test bug: the two keys are equal")
	}
}

// TestOutcomeAggregateDeterministicFold witnesses that the aggregate is a pure
// fold: the SAME recorded outcomes, journaled in ANY order, fold to byte-identical
// per-key stats. A learned policy that stands on this corpus needs the fold to be
// reproducible, not order-dependent.
func TestOutcomeAggregateDeterministicFold(t *testing.T) {
	m := manifestForOutcomes()
	tool := m.Route(Subject{Aspect: AspectToolCall, Tool: "write_file"})
	hard := m.Route(Subject{Aspect: AspectStep, Complexity: ComplexityHigh})

	outs := []struct {
		d Decision
		o Outcome
	}{
		{tool, Outcome{Cost: 0.10, Latency: 100 * time.Millisecond, Quality: 0.8}},
		{hard, Outcome{Cost: 2.0, Latency: time.Second, Quality: 0.5}},
		{tool, Outcome{Cost: 0.30, Latency: 300 * time.Millisecond, Quality: 1.0}},
	}

	var jA OutcomeJournal
	for _, e := range outs {
		jA.Record(m.Version, e.d, e.o)
	}
	// Same records, reversed insertion order.
	var jB OutcomeJournal
	for i := len(outs) - 1; i >= 0; i-- {
		jB.Record(m.Version, outs[i].d, outs[i].o)
	}

	aggA := jA.Aggregate()
	aggB := jB.Aggregate()

	if !reflect.DeepEqual(aggA, aggB) {
		t.Fatalf("fold is order-dependent:\n A=%+v\n B=%+v", aggA, aggB)
	}
	// And folding the same journal twice yields an equal result (pure).
	if !reflect.DeepEqual(aggA, jA.Aggregate()) {
		t.Fatal("Aggregate is not pure: two folds of one journal differ")
	}
	// SortedKeys is the stable order the deterministic fold exposes.
	keys := aggA.SortedKeys()
	if len(keys) != 2 {
		t.Fatalf("SortedKeys: got %d keys, want 2", len(keys))
	}
	// AspectStep ("step") < AspectToolCall ("tool_call") so hard sorts first.
	if keys[0].Rule != "hard" || keys[1].Rule != "tool-writes" {
		t.Errorf("SortedKeys order: got %q,%q, want hard,tool-writes", keys[0].Rule, keys[1].Rule)
	}
}

// TestDecisionWithNoOutcomeContributesNothing witnesses the honesty boundary: a
// decision that was routed but never served (no recorded outcome) contributes
// NOTHING — it does not create a bucket and it does not drag a mean toward zero.
// A measured zero-cost/zero-quality outcome, by contrast, DOES count.
func TestDecisionWithNoOutcomeContributesNothing(t *testing.T) {
	m := manifestForOutcomes()
	served := m.Route(Subject{Aspect: AspectToolCall, Tool: "write_file"})
	unserved := m.Route(Subject{Aspect: AspectStep, Complexity: ComplexityHigh})

	var j OutcomeJournal
	// Record an outcome ONLY for the served decision. The unserved decision is
	// routed (we computed it) but no outcome is journaled for it.
	j.Record(m.Version, served, Outcome{Cost: 0.5, Latency: 50 * time.Millisecond, Quality: 1.0})
	_ = unserved // routed, deliberately NOT recorded

	agg := j.Aggregate()

	if agg.Total != 1 {
		t.Fatalf("Total: got %d, want 1 (only the served decision)", agg.Total)
	}
	if len(agg.ByKey) != 1 {
		t.Fatalf("ByKey: got %d buckets, want 1 (unserved route must not appear)", len(agg.ByKey))
	}
	unservedKey := AspectRuleKey{Aspect: AspectStep, Rule: "hard"}
	if _, ok := agg.ByKey[unservedKey]; ok {
		t.Errorf("unserved route %+v leaked a bucket into the aggregate", unservedKey)
	}

	servedKey := AspectRuleKey{Aspect: AspectToolCall, Rule: "tool-writes"}
	s := agg.ByKey[servedKey]
	// The served bucket's mean is its single sample, NOT halved by a phantom zero
	// from the unserved route.
	if s.Count != 1 || s.MeanQuality != 1.0 || s.MeanCost != 0.5 {
		t.Errorf("served bucket dragged by an absent outcome: got count=%d cost=%v q=%v, want 1 / 0.5 / 1.0", s.Count, s.MeanCost, s.MeanQuality)
	}

	// Counter-case: a MEASURED zero outcome DOES count (absence != measured zero).
	var j2 OutcomeJournal
	j2.Record(m.Version, served, Outcome{Cost: 1.0, Quality: 1.0})
	j2.Record(m.Version, served, Outcome{Cost: 0.0, Quality: 0.0}) // a real, measured zero
	s2 := j2.Aggregate().ByKey[servedKey]
	if s2.Count != 2 {
		t.Fatalf("measured-zero Count: got %d, want 2", s2.Count)
	}
	if s2.MeanCost != 0.5 || s2.MeanQuality != 0.5 {
		t.Errorf("a measured zero must pull the mean: got cost=%v q=%v, want 0.5 / 0.5", s2.MeanCost, s2.MeanQuality)
	}
}

// TestRecordOutcomeBindsDigest witnesses that an OutcomeRecord binds to the exact
// route it grades: its key is the decision's (aspect,rule) and its digest is the
// decision's content-address, so the corpus stays auditable/replayable (#615).
func TestRecordOutcomeBindsDigest(t *testing.T) {
	m := manifestForOutcomes()
	d := m.Route(Subject{Aspect: AspectToolCall, Tool: "write_file"})

	rec := RecordOutcome(m.Version, d, Outcome{Cost: 0.1, Quality: 0.9})
	if rec.Key != (AspectRuleKey{Aspect: AspectToolCall, Rule: "tool-writes"}) {
		t.Errorf("record key: got %+v", rec.Key)
	}
	if rec.Digest == "" || rec.Digest != d.Digest(m.Version) {
		t.Errorf("record digest does not bind to the decision: got %q, want %q", rec.Digest, d.Digest(m.Version))
	}
}
