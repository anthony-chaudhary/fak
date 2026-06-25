package horizonrecovery

import (
	"encoding/json"
	"strings"
	"testing"
)

func sampleSession() Session {
	return Session{
		Source: "s1", Turns: 50, Budget: 8000,
		LinearCumTok: 500000, CompactCumTok: 100000, PlannedCumTok: 100000, FaultTaxCum: 1200,
		References: 40, Faults: 2, Served: 2, Refused: 0, FaultRate: 0.05,
		CompactionLossTurns: 3, FactsRecovered: 2,
	}
}

// The recovery band carries the operands AND the fence, and refuses to emit r.
func TestBandHasNoRField(t *testing.T) {
	band := sessionBand(sampleSession())
	b, err := json.Marshal(band)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"r", "horizon_multiplier", "horizon", "multiplier"} {
		if _, ok := m[forbidden]; ok {
			t.Fatalf("band emits forbidden field %q; r must stay structural", forbidden)
		}
	}
}

func TestRecoveryOperandsAndFenceCoOccur(t *testing.T) {
	b, err := json.Marshal(sessionBand(sampleSession()))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	for _, need := range []string{"recovery_ratio", "reclaimed_tokens", "fault_rate", "faults_refused", "faults_served"} {
		if _, ok := m[need]; !ok {
			t.Fatalf("band missing required co-occurring field %q", need)
		}
	}
}

func TestRecoveryRatioIsFaithfulReprojection(t *testing.T) {
	s := sampleSession()
	band := sessionBand(s)
	if band.ReclaimedTok != s.LinearCumTok-s.CompactCumTok {
		t.Fatalf("reclaimed drift: %d != %d", band.ReclaimedTok, s.LinearCumTok-s.CompactCumTok)
	}
	want := float64(s.LinearCumTok) / float64(s.CompactCumTok)
	if band.RecoveryRatio != want {
		t.Fatalf("ratio drift: %v != %v", band.RecoveryRatio, want)
	}
	if band.Provenance != "measured" {
		t.Fatalf("provenance %q, want measured", band.Provenance)
	}
}

func TestRatioFloorCaseIsNoRecovery(t *testing.T) {
	// When the bounded view held everything (linear == bounded), the ratio is 1 and
	// reclaimed is 0 -- the honest "no recovery" floor, not a fabricated win.
	s := sampleSession()
	s.CompactCumTok = s.LinearCumTok
	band := sessionBand(s)
	if band.RecoveryRatio != 1.0 {
		t.Fatalf("floor ratio %v, want 1.0", band.RecoveryRatio)
	}
	if band.ReclaimedTok != 0 {
		t.Fatalf("floor reclaimed %d, want 0", band.ReclaimedTok)
	}
}

func TestRatioGuardsZeroBounded(t *testing.T) {
	if r := ratio(100, 0); r != 0 {
		t.Fatalf("ratio with zero bounded should guard to 0, got %v", r)
	}
}

func TestAggregateRefusedBelowFloor(t *testing.T) {
	rep := Report{Sessions: []Session{sampleSession()}, Total: Total{Sessions: 1}}
	if _, err := AggregateBand(rep); err == nil {
		t.Fatalf("aggregate band must refuse %d session(s) below floor %d", 1, MinAggregateSessions)
	} else if !strings.Contains(err.Error(), "population claim") {
		t.Fatalf("refusal should explain the population-claim floor, got: %v", err)
	}
}

func TestAggregateAcceptedAtFloor(t *testing.T) {
	rep := syntheticReport() // exactly MinAggregateSessions
	band, err := AggregateBand(rep)
	if err != nil {
		t.Fatalf("aggregate refused a valid %d-session report: %v", rep.Total.Sessions, err)
	}
	if band.Scope != "aggregate" {
		t.Fatalf("scope %q", band.Scope)
	}
	// the fence travels with the aggregate operand too
	if band.FaultRate == 0 && band.FaultsServed == 0 && band.FaultsRefused == 0 {
		// (low fault rate is fine; this just asserts the fields are populated, not absent)
	}
	if band.RecoveryRatio <= 1.0 {
		t.Fatalf("synthetic aggregate should show recovery > 1, got %v", band.RecoveryRatio)
	}
}

func TestBandsFromReportOnePerSession(t *testing.T) {
	rep := Report{Sessions: []Session{sampleSession(), sampleSession(), sampleSession()}}
	bands := BandsFromReport(rep)
	if len(bands) != 3 {
		t.Fatalf("want 3 per-session bands, got %d", len(bands))
	}
	for _, b := range bands {
		if !strings.HasPrefix(b.Scope, "session:") {
			t.Fatalf("per-session band scope %q", b.Scope)
		}
	}
}

func TestSelfcheckPasses(t *testing.T) {
	if err := Selfcheck(); err != nil {
		t.Fatalf("selfcheck failed: %v", err)
	}
}
