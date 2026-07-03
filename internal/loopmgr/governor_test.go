package loopmgr

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestSummarizeTracksConsecutiveRefusals(t *testing.T) {
	now := fixedClock()
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	// refuse, refuse, admit (resets), refuse: streak should end at 1.
	seq := []Event{
		{LoopID: "x", Kind: EventAdmit, Status: StatusRefused},
		{LoopID: "x", Kind: EventAdmit, Status: StatusRefused},
		{LoopID: "x", Kind: EventAdmit, RunID: "r1", Status: StatusAdmitted},
		{LoopID: "x", Kind: EventAdmit, Status: StatusRefused},
	}
	for _, ev := range seq {
		if _, err := Append(path, ev, WithClock(now)); err != nil {
			t.Fatalf("append: %v", err)
		}
	}
	events, err := Load(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	st := Summarize(events, time.Unix(0, 9000).UTC())
	if len(st.Loops) != 1 {
		t.Fatalf("loops = %d", len(st.Loops))
	}
	loop := st.Loops[0]
	if loop.Refused != 3 || loop.Admitted != 1 || loop.ConsecutiveRefusals != 1 {
		t.Fatalf("refused=%d admitted=%d consecutive=%d", loop.Refused, loop.Admitted, loop.ConsecutiveRefusals)
	}
}

func TestAdmitZeroPolicyAlwaysAdmits(t *testing.T) {
	loop := LoopSnapshot{LoopID: "x", Refused: 9, ConsecutiveRefusals: 9, Ended: 100, Witnessed: 0}
	d := Admit(loop, Policy{}, time.Unix(0, 0).UTC())
	if !d.Admit || d.Reason != ReasonAdmitted {
		t.Fatalf("zero policy must admit, got admit=%v reason=%q", d.Admit, d.Reason)
	}
}

func TestAdmitPausedAndDisabled(t *testing.T) {
	loop := LoopSnapshot{LoopID: "x"}
	if d := Admit(loop, Policy{Paused: true}, time.Unix(0, 0).UTC()); d.Admit || d.Reason != ReasonLoopPaused {
		t.Fatalf("paused: admit=%v reason=%q", d.Admit, d.Reason)
	}
	if d := Admit(loop, Policy{Disabled: true}, time.Unix(0, 0).UTC()); d.Admit || d.Reason != ReasonLoopDisabled {
		t.Fatalf("disabled: admit=%v reason=%q", d.Admit, d.Reason)
	}
	// Disabled wins over paused (checked first).
	if d := Admit(loop, Policy{Paused: true, Disabled: true}, time.Unix(0, 0).UTC()); d.Reason != ReasonLoopDisabled {
		t.Fatalf("disabled should win, got %q", d.Reason)
	}
}

func TestAdmitConcurrencyBudget(t *testing.T) {
	pol := Policy{MaxConcurrent: 3}

	// Not configured ⇒ admit even when many runs are in flight.
	saturated := LoopSnapshot{LoopID: "x", Started: 10, Ended: 0}
	if d := Admit(saturated, Policy{}, time.Unix(0, 0).UTC()); !d.Admit {
		t.Fatalf("budget unset must admit, got reason=%q", d.Reason)
	}

	// Under budget (2 in flight, cap 3) ⇒ admit.
	under := LoopSnapshot{LoopID: "x", Started: 5, Ended: 3} // 2 in flight
	if d := Admit(under, pol, time.Unix(0, 0).UTC()); !d.Admit {
		t.Fatalf("under budget must admit, got admit=%v reason=%q", d.Admit, d.Reason)
	}

	// At budget (3 in flight, cap 3) ⇒ refuse with BUDGET_SPENT.
	at := LoopSnapshot{LoopID: "x", Started: 3, Ended: 0} // 3 in flight
	if d := Admit(at, pol, time.Unix(0, 0).UTC()); d.Admit || d.Reason != ReasonBudgetSpent {
		t.Fatalf("at budget must refuse with BUDGET_SPENT, got admit=%v reason=%q", d.Admit, d.Reason)
	}

	// Over budget (5 in flight, cap 3) ⇒ refuse with BUDGET_SPENT.
	over := LoopSnapshot{LoopID: "x", Started: 6, Ended: 1} // 5 in flight
	if d := Admit(over, pol, time.Unix(0, 0).UTC()); d.Admit || d.Reason != ReasonBudgetSpent {
		t.Fatalf("over budget must refuse with BUDGET_SPENT, got admit=%v reason=%q", d.Admit, d.Reason)
	}

	// Defensive: Ended >= Started yields zero in-flight ⇒ admit.
	drained := LoopSnapshot{LoopID: "x", Started: 3, Ended: 3}
	if d := Admit(drained, pol, time.Unix(0, 0).UTC()); !d.Admit {
		t.Fatalf("fully drained loop must admit, got reason=%q", d.Reason)
	}
}

func TestAdmitBudgetGateOrder(t *testing.T) {
	// A loop saturated past its budget. Disabled (and Paused) are hard refuses
	// checked before the budget gate, so they win deterministically.
	saturated := LoopSnapshot{LoopID: "x", Started: 9, Ended: 0}
	if d := Admit(saturated, Policy{MaxConcurrent: 3, Disabled: true}, time.Unix(0, 0).UTC()); d.Reason != ReasonLoopDisabled {
		t.Fatalf("disabled must win over budget, got %q", d.Reason)
	}
	if d := Admit(saturated, Policy{MaxConcurrent: 3, Paused: true}, time.Unix(0, 0).UTC()); d.Reason != ReasonLoopPaused {
		t.Fatalf("paused must win over budget, got %q", d.Reason)
	}
	// Budget is checked before the cadence floor: a loop both inside its cadence
	// floor AND over budget refuses with BUDGET_SPENT, not CADENCE_FLOOR.
	last := time.Unix(1000, 0).UTC()
	tight := LoopSnapshot{LoopID: "x", Started: 4, Ended: 0, LastEventUnixNano: last.UnixNano()}
	pol := Policy{MaxConcurrent: 3, MinIntervalSeconds: 300}
	if d := Admit(tight, pol, last.Add(10*time.Second)); d.Admit || d.Reason != ReasonBudgetSpent {
		t.Fatalf("budget must be checked before cadence, got admit=%v reason=%q", d.Admit, d.Reason)
	}
}

func TestAdmitCadenceFloor(t *testing.T) {
	last := time.Unix(1000, 0).UTC()
	loop := LoopSnapshot{LoopID: "x", LastEventUnixNano: last.UnixNano()}
	pol := Policy{MinIntervalSeconds: 300}

	if d := Admit(loop, pol, last.Add(100*time.Second)); d.Admit || d.Reason != ReasonCadenceFloor {
		t.Fatalf("inside floor: admit=%v reason=%q", d.Admit, d.Reason)
	}
	if d := Admit(loop, pol, last.Add(300*time.Second)); !d.Admit {
		t.Fatalf("at floor must admit, got reason=%q", d.Reason)
	}
	if d := Admit(LoopSnapshot{LoopID: "x"}, pol, last); !d.Admit {
		t.Fatalf("no prior event must admit, got reason=%q", d.Reason)
	}
}

func TestAdmitRefusalStorm(t *testing.T) {
	pol := Policy{MaxConsecutiveRefusals: 3}
	if d := Admit(LoopSnapshot{LoopID: "x", ConsecutiveRefusals: 2}, pol, time.Unix(0, 0).UTC()); !d.Admit {
		t.Fatalf("under cap must admit, got reason=%q", d.Reason)
	}
	if d := Admit(LoopSnapshot{LoopID: "x", ConsecutiveRefusals: 3}, pol, time.Unix(0, 0).UTC()); d.Admit || d.Reason != ReasonRefusalStorm {
		t.Fatalf("at cap must refuse, got admit=%v reason=%q", d.Admit, d.Reason)
	}
}

func TestAdmitWitnessCollapse(t *testing.T) {
	pol := Policy{MinWitnessRate: 0.5, MinRunsForWitnessGate: 4}

	collapsed := LoopSnapshot{LoopID: "x", Ended: 10, Witnessed: 1}
	if d := Admit(collapsed, pol, time.Unix(0, 0).UTC()); d.Admit || d.Reason != ReasonWitnessCollapse {
		t.Fatalf("collapse: admit=%v reason=%q", d.Admit, d.Reason)
	}
	healthy := LoopSnapshot{LoopID: "x", Ended: 10, Witnessed: 8}
	if d := Admit(healthy, pol, time.Unix(0, 0).UTC()); !d.Admit {
		t.Fatalf("healthy must admit, got reason=%q", d.Reason)
	}
	young := LoopSnapshot{LoopID: "x", Ended: 2, Witnessed: 0}
	if d := Admit(young, pol, time.Unix(0, 0).UTC()); !d.Admit {
		t.Fatalf("young loop must not be witness-gated, got reason=%q", d.Reason)
	}
}

func TestPolicyForPrefersOverride(t *testing.T) {
	ps := Policies{
		Default: Policy{MaxConsecutiveRefusals: 5},
		Loops:   map[string]Policy{"special": {Paused: true}},
	}
	if got := ps.PolicyFor("special"); !got.Paused {
		t.Fatalf("override not used: %+v", got)
	}
	if got := ps.PolicyFor("other"); got.MaxConsecutiveRefusals != 5 {
		t.Fatalf("default not used: %+v", got)
	}
}

func TestLoadPoliciesMissingFileIsPermissive(t *testing.T) {
	ps, err := LoadPolicies(filepath.Join(t.TempDir(), "nope.json"))
	if err != nil {
		t.Fatalf("missing file must not error: %v", err)
	}
	if ps.Default.Paused || len(ps.Loops) != 0 {
		t.Fatalf("missing file must be empty/permissive: %+v", ps)
	}
}

func TestLoadPoliciesRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "policy.json")
	doc := `{"schema":"fak.loop-policy.v1","default":{"min_interval_seconds":300,"max_consecutive_refusals":4},"loops":{"issue-dispatch/default":{"paused":true}}}`
	if err := os.WriteFile(path, []byte(doc), 0o644); err != nil {
		t.Fatal(err)
	}
	ps, err := LoadPolicies(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if ps.Default.MinIntervalSeconds != 300 || ps.Default.MaxConsecutiveRefusals != 4 {
		t.Fatalf("default wrong: %+v", ps.Default)
	}
	if !ps.PolicyFor("issue-dispatch/default").Paused {
		t.Fatalf("per-loop override not loaded")
	}
}

func TestLoadPoliciesRejectsBadSchemaAndJSON(t *testing.T) {
	dir := t.TempDir()
	bad := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(bad, []byte(`{"schema":"wrong.v9"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadPolicies(bad); err == nil {
		t.Fatal("wrong schema must error")
	}
	broken := filepath.Join(dir, "broken.json")
	if err := os.WriteFile(broken, []byte(`{not json`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadPolicies(broken); err == nil {
		t.Fatal("malformed json must error")
	}
}

func TestAdmitAllStableOrder(t *testing.T) {
	st := Status{Loops: []LoopSnapshot{
		{LoopID: "zeta"},
		{LoopID: "alpha", ConsecutiveRefusals: 9},
	}}
	ps := Policies{Default: Policy{MaxConsecutiveRefusals: 3}}
	ds := AdmitAll(st, ps, time.Unix(0, 0).UTC())
	if len(ds) != 2 || ds[0].LoopID != "alpha" || ds[1].LoopID != "zeta" {
		t.Fatalf("not stable-sorted: %+v", ds)
	}
	if ds[0].Admit {
		t.Fatalf("alpha should be refused (storm)")
	}
}

// TestDefaultPoliciesShipsSaneBackpressure pins the embedded default's actual
// gate values: a global cadence floor + refusal-storm cap on every unnamed loop,
// and the garden bundle's 12h floor. This is the shipped answer to "what
// backpressure does a background loop get before an operator writes a policy?".
func TestDefaultPoliciesShipsSaneBackpressure(t *testing.T) {
	dp := DefaultPolicies()
	if dp.Schema != SchemaPolicies {
		t.Fatalf("embedded default schema = %q, want %q", dp.Schema, SchemaPolicies)
	}
	// Global default: a cadence floor and a refusal-storm cap, both non-zero so
	// an unnamed loop is never admit-always.
	if dp.Default.MinIntervalSeconds <= 0 {
		t.Fatalf("global default must carry a cadence floor, got %+v", dp.Default)
	}
	if dp.Default.MaxConsecutiveRefusals == 0 {
		t.Fatalf("global default must carry a refusal-storm cap, got %+v", dp.Default)
	}
	// A never-configured loop inherits the global brake: a storm past the cap is
	// refused, not admitted.
	stormed := LoopSnapshot{LoopID: "some-new-loop", ConsecutiveRefusals: dp.Default.MaxConsecutiveRefusals}
	if d := Admit(stormed, dp.PolicyFor("some-new-loop"), time.Unix(0, 0).UTC()); d.Admit {
		t.Fatalf("a storming unnamed loop must be refused by the default cap, got %+v", d)
	}
	if d := Admit(stormed, dp.PolicyFor("some-new-loop"), time.Unix(0, 0).UTC()); d.Reason != ReasonRefusalStorm {
		t.Fatalf("expected %s, got %q", ReasonRefusalStorm, d.Reason)
	}
	// The garden override still carries a storm cap (an override replaces the
	// default whole, so it must restate the global brake it wants to keep).
	garden := dp.PolicyFor("garden/default")
	if garden.MaxConsecutiveRefusals == 0 {
		t.Fatalf("garden override dropped the refusal-storm cap: %+v", garden)
	}
	if garden.MinIntervalSeconds <= dp.Default.MinIntervalSeconds {
		t.Fatalf("garden override should floor cadence tighter than the global default: %+v", garden)
	}
}

// TestLoadPoliciesOrDefaultUsesEmbeddedWhenAbsent proves the background governors'
// loader falls back to the embedded sane default when no operator policy exists,
// while a present file still wins whole.
func TestLoadPoliciesOrDefaultUsesEmbeddedWhenAbsent(t *testing.T) {
	// Absent file -> embedded default (braked), not the empty permissive set.
	ps, err := LoadPoliciesOrDefault(filepath.Join(t.TempDir(), "nope.json"))
	if err != nil {
		t.Fatalf("absent file must not error: %v", err)
	}
	if ps.Default.MaxConsecutiveRefusals == 0 || ps.Default.MinIntervalSeconds <= 0 {
		t.Fatalf("absent file must yield the braked default, got %+v", ps.Default)
	}
	// Empty path -> embedded default too.
	if ps, err = LoadPoliciesOrDefault(""); err != nil || ps.Default.MaxConsecutiveRefusals == 0 {
		t.Fatalf("empty path must yield the braked default, got %+v (err %v)", ps.Default, err)
	}
	// A present operator file wins whole (not merged with the default).
	path := filepath.Join(t.TempDir(), "policy.json")
	if err := os.WriteFile(path, []byte(`{"schema":"fak.loop-policy.v1","default":{"paused":true}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	ps, err = LoadPoliciesOrDefault(path)
	if err != nil {
		t.Fatalf("present file load: %v", err)
	}
	if !ps.Default.Paused {
		t.Fatalf("present file must win whole, got %+v", ps.Default)
	}
	if ps.Default.MaxConsecutiveRefusals != 0 {
		t.Fatalf("present file must NOT be merged with the embedded default, got %+v", ps.Default)
	}
	// A present-but-malformed file is still a loud error (not silently defaulted).
	bad := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(bad, []byte(`{not json`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadPoliciesOrDefault(bad); err == nil {
		t.Fatal("malformed present file must error, not silently default")
	}
}
