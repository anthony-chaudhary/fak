package rsiloop

import "testing"

// TestFrozenRuleSynthCorpus_Mines proves the committed fixture is actually mined from
// the real floor (not a hand-built []NearMiss) and is non-empty — a corpus that mined
// to zero would give the loop nothing to drive, silently making the rulesynth harness a
// no-op.
func TestFrozenRuleSynthCorpus_Mines(t *testing.T) {
	corpus, benign := FrozenRuleSynthCorpus()
	if len(corpus) == 0 {
		t.Fatal("frozen corpus mined zero near-misses — the rulesynth harness would drive nothing")
	}
	if len(benign) == 0 {
		t.Fatal("frozen benign corpus is empty — a kept rule's no-regression claim would be vacuous")
	}
	// Every mined row must carry the command it was mined from and a guarded glob it
	// reached — the args/command-bearing capture that makes it Propose-able.
	for i, nm := range corpus {
		if nm.Command == "" {
			t.Errorf("corpus[%d] has empty command", i)
		}
		if nm.GuardedGlob == "" {
			t.Errorf("corpus[%d] (%q) names no guarded glob", i, nm.Command)
		}
	}
}

// TestFrozenRuleSynthCorpus_DrivesAKeep is the end-to-end witness for #586: the frozen
// corpus, run through the SAME engine and keep-bit cmd/rsiloop drives, must KEEP at
// least one synthesized rule — a strict, gated gain over the zero-catch floor. This is
// what makes NewRuleSynthHarness a REAL second subsystem (a kept gain), not a wired-but-
// inert harness.
func TestFrozenRuleSynthCorpus_DrivesAKeep(t *testing.T) {
	corpus, benign := FrozenRuleSynthCorpus()
	h := NewRuleSynthHarness(corpus, benign)

	res, err := Run(h, nil, 0, 0)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Kept < 1 {
		t.Fatalf("expected >=1 kept rule from the frozen corpus, got %d (rows: %+v)", res.Kept, res.Rows)
	}
	if res.FinalBaseline < 1 {
		t.Fatalf("kept rule should catch >=1 near-miss, final baseline = %v", res.FinalBaseline)
	}
	// The kept row must be a real, gated KEEP — measured, suite-green (no benign
	// regression), and truth-clean (caught its whole cluster).
	var kept *Row
	for i := range res.Rows {
		if res.Rows[i].Kept {
			kept = &res.Rows[i]
			break
		}
	}
	if kept == nil {
		t.Fatal("no kept row found")
	}
	if !kept.SuiteGreen {
		t.Errorf("kept rule regressed a benign call (suite not green): %+v", kept)
	}
	if !kept.TruthClean {
		t.Errorf("kept rule did not catch its whole cluster (not truth-clean): %+v", kept)
	}
	if kept.MetricName != RuleSynthMetricName {
		t.Errorf("metric name = %q, want %q", kept.MetricName, RuleSynthMetricName)
	}
}
