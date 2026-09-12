package turncost_test

import (
	"os"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/turncost"
)

// TestTurnCostRecordPhasePin injects a distinct fake delay into each of the six
// phases and pins the recorded duration to it. This is the phase-attribution
// correctness proof the ticket's done condition names: if a hook ever mislabels
// which phase a duration belongs to, or a phase is silently dropped, one of
// these assertions fires.
func TestTurnCostRecordPhasePin(t *testing.T) {
	cases := []struct {
		phase turncost.Phase
		delay time.Duration
	}{
		{turncost.PhaseAdmission, 11 * time.Millisecond},
		{turncost.PhasePrefill, 22 * time.Millisecond},
		{turncost.PhaseDecode, 33 * time.Millisecond},
		{turncost.PhaseStream, 44 * time.Millisecond},
		{turncost.PhaseProxyHop, 55 * time.Millisecond},
		{turncost.PhaseLedger, 66 * time.Millisecond},
	}
	if len(cases) != 6 {
		t.Fatalf("phase pin must cover all six phases, got %d", len(cases))
	}

	var rec turncost.TurnCostRecord
	for _, tc := range cases {
		start := time.Now()
		// Simulate the instrumented seam: a fake delay, then record it exactly
		// as a timing hook would.
		time.Sleep(tc.delay)
		rec.Add(tc.phase, time.Since(start))
	}

	if got := len(rec.MeasuredPhases()); got != 6 {
		t.Fatalf("measured phases = %d, want 6 (%v)", got, rec.MeasuredPhases())
	}
	for _, tc := range cases {
		got, ok := rec.Phase(tc.phase)
		if !ok {
			t.Fatalf("phase %q not measured", tc.phase)
		}
		want := tc.delay.Seconds()
		// Sleep granularity means the observed value is >= the requested delay;
		// allow generous upper slack while still pinning attribution.
		if got < want {
			t.Fatalf("phase %q = %.6fs, want >= %.6fs", tc.phase, got, want)
		}
		if got > want*4+0.05 {
			t.Fatalf("phase %q = %.6fs, wildly above %.6fs", tc.phase, got, want)
		}
	}
}

// TestTurnCostRecordPhaseIsolation proves a duration lands in exactly one phase:
// adding to prefill must not perturb decode or any other bucket.
func TestTurnCostRecordPhaseIsolation(t *testing.T) {
	var rec turncost.TurnCostRecord
	rec.Add(turncost.PhasePrefill, 7*time.Millisecond)
	if _, ok := rec.Phase(turncost.PhaseDecode); ok {
		t.Fatal("decode was measured after only prefill was added")
	}
	if v, _ := rec.Phase(turncost.PhasePrefill); v <= 0 {
		t.Fatal("prefill not recorded")
	}
	// Absent phases must not be fabricated as zero-valued present phases.
	if rec.Phases[turncost.PhaseProxyHop] != 0 {
		t.Fatal("unmeasured phase acquired a non-zero value")
	}
	if _, present := rec.Phases[turncost.PhaseProxyHop]; present {
		t.Fatal("unmeasured phase was fabricated as present")
	}
}

// TestTurnCostRecordIgnoresNonPositive keeps the zero-fabrication invariant: a
// seam that measured no time (or a negative clock skew) contributes nothing.
func TestTurnCostRecordIgnoresNonPositive(t *testing.T) {
	var rec turncost.TurnCostRecord
	rec.Add(turncost.PhaseDecode, 0)
	rec.Add(turncost.PhaseDecode, -time.Second)
	if len(rec.MeasuredPhases()) != 0 {
		t.Fatalf("non-positive durations fabricated phases: %v", rec.MeasuredPhases())
	}
	rec.Add(turncost.Phase("not-a-phase"), time.Second)
	if len(rec.MeasuredPhases()) != 0 {
		t.Fatal("unknown phase was accepted")
	}
}

// TestCollectorPrometheus pins the exposition: the family appears with the
// surface/phase label set, sums accumulate, samples count, and an absent phase
// emits NO series (rather than a zero row).
func TestCollectorPrometheus(t *testing.T) {
	c := turncost.NewCollector()

	stream := &turncost.TurnCostRecord{Streaming: true, Trace: "t-1", Model: "m"}
	stream.Add(turncost.PhaseAdmission, 3*time.Millisecond)
	stream.Add(turncost.PhasePrefill, 10*time.Millisecond)
	stream.Add(turncost.PhaseDecode, 20*time.Millisecond)
	stream.Add(turncost.PhaseStream, 5*time.Millisecond)
	stream.Add(turncost.PhaseProxyHop, 40*time.Millisecond)
	stream.Add(turncost.PhaseLedger, time.Millisecond)
	c.Observe(stream)

	// A second streamed turn accumulates onto the same series, but never
	// observes the ledger phase.
	stream2 := &turncost.TurnCostRecord{Streaming: true}
	stream2.Add(turncost.PhaseDecode, 4*time.Millisecond)
	c.Observe(stream2)

	buffered := &turncost.TurnCostRecord{}
	buffered.Add(turncost.PhasePrefill, 8*time.Millisecond)
	c.Observe(buffered)

	out := c.Prometheus()

	for _, want := range []string{
		`# TYPE fak_turn_cost_turns_total counter`,
		`fak_turn_cost_turns_total{surface="stream"} 2`,
		`fak_turn_cost_turns_total{surface="buffered"} 1`,
		`# TYPE fak_turn_cost_seconds_total counter`,
		`fak_turn_cost_seconds_total{surface="stream",phase="admission"} 0.003`,
		`fak_turn_cost_seconds_total{surface="stream",phase="prefill"} 0.01`,
		`fak_turn_cost_seconds_total{surface="stream",phase="decode"} 0.024`,
		`fak_turn_cost_seconds_total{surface="stream",phase="stream"} 0.005`,
		`fak_turn_cost_seconds_total{surface="stream",phase="proxy_hop"} 0.04`,
		`fak_turn_cost_seconds_total{surface="stream",phase="ledger"} 0.001`,
		`fak_turn_cost_seconds_total{surface="buffered",phase="prefill"} 0.008`,
		`fak_turn_cost_phase_samples_total{surface="stream",phase="decode"} 2`,
		`fak_turn_cost_phase_samples_total{surface="stream",phase="ledger"} 1`,
		`fak_turn_cost_phase_samples_total{surface="buffered",phase="prefill"} 1`,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("exposition missing %q\n--- got ---\n%s", want, out)
		}
	}

	// The buffered surface never measured decode; that phase must have no row.
	if strings.Contains(out, `fak_turn_cost_seconds_total{surface="buffered",phase="decode"}`) {
		t.Fatalf("absent phase emitted a fabricated series\n%s", out)
	}
	// Trace/model must never become labels (bounded-cardinality invariant).
	for _, forbidden := range []string{"t-1", `model=`} {
		if strings.Contains(out, forbidden) {
			t.Fatalf("forbidden high-cardinality label leaked into exposition: %q\n%s", forbidden, out)
		}
	}
}

// TestCollectorNilSafe proves the fail-open contract: a nil collector and a nil
// record are both no-ops, so a caller may wire the surface unconditionally.
func TestCollectorNilSafe(t *testing.T) {
	var zero turncost.Collector
	zero.Observe(&turncost.TurnCostRecord{Phases: map[turncost.Phase]float64{turncost.PhaseDecode: 0.1}})
	if !strings.Contains(zero.Prometheus(), `phase="decode"} 0.1`) {
		t.Fatal("zero-value collector did not retain its first phase")
	}
	var c *turncost.Collector
	c.Observe(&turncost.TurnCostRecord{})
	if out := c.Prometheus(); out != "" {
		t.Fatalf("nil collector rendered %q, want empty", out)
	}
	live := turncost.NewCollector()
	live.Observe(nil)
	if out := live.Prometheus(); out != "" {
		t.Fatalf("empty collector rendered %q, want empty", out)
	}
}

// TestEnabledOptIn pins the env-gate semantics shared by both
// repos: unset -> off; falsey -> off; anything else -> on.
func TestEnabledOptIn(t *testing.T) {
	t.Setenv("FAK_TURN_COST", "")
	if err := os.Unsetenv("FAK_TURN_COST"); err != nil {
		t.Fatal(err)
	}
	if turncost.Enabled() {
		t.Fatal("unset gate must remain off")
	}
	t.Setenv("FAK_TURN_COST", "")
	// Setenv with "" sets the variable to empty, which is falsey by contract.
	if turncost.Enabled() {
		t.Fatal("empty FAK_TURN_COST should disable exposure")
	}
	t.Setenv("FAK_TURN_COST", "0")
	if turncost.Enabled() {
		t.Fatal("FAK_TURN_COST=0 should disable exposure")
	}
	t.Setenv("FAK_TURN_COST", "1")
	if !turncost.Enabled() {
		t.Fatal("FAK_TURN_COST=1 should enable exposure")
	}
}

// TestPhaseVocabularyIsClosed pins that every advertised phase validates and an
// arbitrary string does not — the closed-vocabulary invariant.
func TestPhaseVocabularyIsClosed(t *testing.T) {
	if len(turncost.Phases) != 6 {
		t.Fatalf("phase vocabulary has %d entries, want 6", len(turncost.Phases))
	}
	for _, p := range turncost.Phases {
		if !p.Valid() {
			t.Fatalf("advertised phase %q does not validate", p)
		}
	}
	if turncost.Phase("bogus").Valid() {
		t.Fatal("unknown phase validated")
	}
}
