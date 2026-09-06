package assumecheck

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// TestSessionBridge proves that a fixture AssumptionReport maps onto the shared
// assumecheck.Outcome vocabulary, covering all four closed outcome branches:
// holds, stale, unverifiable, and violated (#3828, epic #3818 C9).
func TestSessionBridge(t *testing.T) {
	fixtureReport := ctxplan.AssumptionReport{
		EffectSafe: false,
		Summary: ctxplan.AssumptionSummary{
			Use:     1,
			Refresh: 1,
			Query:   2,
		},
		Assessments: []ctxplan.AssumptionAssessment{
			{
				Key:        "session-holds",
				Statement:  "seat active and reachable",
				Source:     ctxplan.AssumptionWitnessed,
				Confidence: 0.95,
				Action:     ctxplan.AssumptionUse,
				Reason:     "assumption has direct source and sufficient confidence",
			},
			{
				Key:        "session-stale",
				Statement:  "cached auth token",
				Source:     ctxplan.AssumptionStale,
				Confidence: 0.20,
				Action:     ctxplan.AssumptionRefresh,
				Reason:     "stale assumption must refresh its source before effects",
			},
			{
				Key:        "session-unverifiable",
				Statement:  "target deployment region",
				Source:     ctxplan.AssumptionUnknown,
				Confidence: 0.0,
				Action:     ctxplan.AssumptionQuery,
				Reason:     "unknown assumption must be queried before effects",
			},
			{
				Key:        "session-violated",
				Statement:  "worker latency under 50ms",
				Source:     ctxplan.AssumptionInferred,
				Confidence: 0.40,
				Action:     ctxplan.AssumptionQuery,
				Reason:     "inferred assumption below confidence threshold",
			},
		},
	}

	assessed := IngestAssumptionReport(fixtureReport)
	if len(assessed) != 4 {
		t.Fatalf("IngestAssumptionReport returned %d items, want 4", len(assessed))
	}

	// Verify alias SessionBridge behaves identically.
	aliasAssessed := SessionBridge(fixtureReport)
	if len(aliasAssessed) != len(assessed) {
		t.Fatalf("SessionBridge returned %d items, want %d", len(aliasAssessed), len(assessed))
	}

	expected := map[string]struct {
		wantOutcome Outcome
		wantRefusal string
		blocks      bool
	}{
		"session-holds": {
			wantOutcome: OutcomeHolds,
			wantRefusal: "",
			blocks:      false,
		},
		"session-stale": {
			wantOutcome: OutcomeStale,
			wantRefusal: "ASSUMPTION_STALE",
			blocks:      true,
		},
		"session-unverifiable": {
			wantOutcome: OutcomeUnverifiable,
			wantRefusal: "ASSUMPTION_UNVERIFIABLE",
			blocks:      true,
		},
		"session-violated": {
			wantOutcome: OutcomeViolated,
			wantRefusal: "ASSUMPTION_VIOLATED",
			blocks:      true,
		},
	}

	for _, item := range assessed {
		exp, ok := expected[item.Assumption.ID]
		if !ok {
			t.Fatalf("unexpected assumption ID %q in results", item.Assumption.ID)
		}

		if item.Outcome != exp.wantOutcome {
			t.Errorf("assumption %q outcome = %s, want %s (reason: %s)", item.Assumption.ID, item.Outcome, exp.wantOutcome, item.Reason)
		}

		if !ValidOutcome(item.Outcome) {
			t.Errorf("assumption %q produced outcome %q outside closed vocabulary", item.Assumption.ID, item.Outcome)
		}

		if item.Assumption.Level != LevelSession {
			t.Errorf("assumption %q level = %s, want %s", item.Assumption.ID, item.Assumption.Level, LevelSession)
		}

		if !ValidLevel(item.Assumption.Level) {
			t.Errorf("assumption %q level %q outside closed Level vocabulary", item.Assumption.ID, item.Assumption.Level)
		}

		if item.Assumption.Owner != SessionAssumptionSource {
			t.Errorf("assumption %q owner = %q, want %q", item.Assumption.ID, item.Assumption.Owner, SessionAssumptionSource)
		}

		if item.Source != SessionAssumptionSource {
			t.Errorf("assessed %q source = %q, want %q", item.Assumption.ID, item.Source, SessionAssumptionSource)
		}

		if item.Assumption.WitnessKind != WitnessSessionReport {
			t.Errorf("assumption %q witness = %s, want %s", item.Assumption.ID, item.Assumption.WitnessKind, WitnessSessionReport)
		}

		if !ValidWitnessKind(item.Assumption.WitnessKind) {
			t.Errorf("assumption %q witness kind %q outside closed WitnessKind vocabulary", item.Assumption.ID, item.Assumption.WitnessKind)
		}

		if item.Assumption.WitnessStatus != WitnessWired {
			t.Errorf("assumption %q witness status = %s, want %s", item.Assumption.ID, item.Assumption.WitnessStatus, WitnessWired)
		}

		if item.Assumption.RefusalReason != exp.wantRefusal {
			t.Errorf("assumption %q refusal reason = %q, want %q", item.Assumption.ID, item.Assumption.RefusalReason, exp.wantRefusal)
		}

		if item.BlocksReliance() != exp.blocks {
			t.Errorf("assumption %q BlocksReliance() = %v, want %v", item.Assumption.ID, item.BlocksReliance(), exp.blocks)
		}

		if item.Reason == "" {
			t.Errorf("assumption %q has empty reason", item.Assumption.ID)
		}

		// Check Verdict projection.
		v := item.Verdict()
		if v.AssumptionID != item.Assumption.ID || v.Outcome != item.Outcome || v.Level != item.Assumption.Level {
			t.Errorf("Verdict projection mismatch for %q: %+v", item.Assumption.ID, v)
		}
	}
}

// TestSessionBridgeEndToEndWithAssessAssumptions proves that raw ctxplan.Assumption inputs
// evaluated through ctxplan.AssessAssumptions correctly map onto expected assumecheck outcomes.
func TestSessionBridgeEndToEndWithAssessAssumptions(t *testing.T) {
	rawAssumptions := []ctxplan.Assumption{
		{
			Key:        "a-witnessed-holds",
			Statement:  "preflight check passed",
			Source:     ctxplan.AssumptionWitnessed,
			Confidence: 0.90,
		},
		{
			Key:       "b-stale-refresh",
			Statement: "old session token",
			Source:    ctxplan.AssumptionStale,
		},
		{
			Key:       "c-unknown-query",
			Statement: "target database port",
			Source:    ctxplan.AssumptionUnknown,
		},
		{
			Key:        "d-inferred-violated",
			Statement:  "expected memory limit",
			Source:     ctxplan.AssumptionInferred,
			Confidence: 0.50, // default policy requires 0.80 for inferred
		},
	}

	assessed := IngestSessionAssumptionsDefault(rawAssumptions)
	if len(assessed) != 4 {
		t.Fatalf("IngestSessionAssumptionsDefault returned %d items, want 4", len(assessed))
	}

	byKey := make(map[string]AssessedAssumption, len(assessed))
	for _, a := range assessed {
		byKey[a.Assumption.ID] = a
	}

	tests := []struct {
		id          string
		wantOutcome Outcome
	}{
		{"a-witnessed-holds", OutcomeHolds},
		{"b-stale-refresh", OutcomeStale},
		{"c-unknown-query", OutcomeUnverifiable},
		{"d-inferred-violated", OutcomeViolated},
	}

	for _, tc := range tests {
		got, ok := byKey[tc.id]
		if !ok {
			t.Fatalf("missing assumption %q in assessed results", tc.id)
		}
		if got.Outcome != tc.wantOutcome {
			t.Errorf("assumption %q outcome = %s, want %s (reason: %s)", tc.id, got.Outcome, tc.wantOutcome, got.Reason)
		}
		if got.Assumption.Level != LevelSession {
			t.Errorf("assumption %q level = %s, want %s", tc.id, got.Assumption.Level, LevelSession)
		}
		if got.Assumption.Owner != "managed-context" {
			t.Errorf("assumption %q owner = %q, want managed-context", tc.id, got.Assumption.Owner)
		}
	}
}

// TestSessionBridgeEffectSafeVariant proves that when a report is marked EffectSafe: true,
// low-confidence known-source assumptions map to OutcomeUnverifiable rather than OutcomeViolated.
func TestSessionBridgeEffectSafeVariant(t *testing.T) {
	report := ctxplan.AssumptionReport{
		EffectSafe: true,
		Assessments: []ctxplan.AssumptionAssessment{
			{
				Key:        "low-conf-safe",
				Statement:  "inferred route hint",
				Source:     ctxplan.AssumptionInferred,
				Confidence: 0.40,
				Action:     ctxplan.AssumptionQuery,
				Reason:     "inferred assumption below confidence threshold",
			},
			{
				Key:        "direct-low-safe",
				Statement:  "weak witness",
				Source:     ctxplan.AssumptionWitnessed,
				Confidence: 0.30,
				Action:     ctxplan.AssumptionQuery,
				Reason:     "assumption below confidence threshold",
			},
		},
	}

	assessed := IngestAssumptionReport(report)
	for _, a := range assessed {
		if a.Outcome != OutcomeUnverifiable {
			t.Errorf("assumption %q with EffectSafe=true got outcome %s, want %s", a.Assumption.ID, a.Outcome, OutcomeUnverifiable)
		}
		if !strings.Contains(a.Reason, "confidence") {
			t.Errorf("assumption %q reason %q does not contain confidence", a.Assumption.ID, a.Reason)
		}
	}
}

// TestSessionBridgeHelperProjections proves AssumptionsFromReport and VerdictsFromReport.
func TestSessionBridgeHelperProjections(t *testing.T) {
	report := ctxplan.AssumptionReport{
		EffectSafe: false,
		Assessments: []ctxplan.AssumptionAssessment{
			{
				Key:        "item-1",
				Statement:  "first claim",
				Source:     ctxplan.AssumptionWitnessed,
				Confidence: 1.0,
				Action:     ctxplan.AssumptionUse,
			},
			{
				Key:       "item-2",
				Statement: "second claim",
				Source:    ctxplan.AssumptionStale,
				Action:    ctxplan.AssumptionRefresh,
			},
		},
	}

	assumptions := AssumptionsFromReport(report)
	if len(assumptions) != 2 {
		t.Fatalf("AssumptionsFromReport returned %d items, want 2", len(assumptions))
	}
	if assumptions[0].ID != "item-1" || assumptions[0].Level != LevelSession || assumptions[0].Owner != "managed-context" {
		t.Errorf("unexpected assumptions[0]: %+v", assumptions[0])
	}

	verdicts := VerdictsFromReport(report)
	if len(verdicts) != 2 {
		t.Fatalf("VerdictsFromReport returned %d items, want 2", len(verdicts))
	}
	if verdicts[0].AssumptionID != "item-1" || verdicts[0].Outcome != OutcomeHolds {
		t.Errorf("unexpected verdicts[0]: %+v", verdicts[0])
	}
	if verdicts[1].AssumptionID != "item-2" || verdicts[1].Outcome != OutcomeStale {
		t.Errorf("unexpected verdicts[1]: %+v", verdicts[1])
	}

	// Empty report returns nil.
	if got := IngestAssumptionReport(ctxplan.AssumptionReport{}); got != nil {
		t.Errorf("empty report got %+v, want nil", got)
	}
	if got := AssumptionsFromReport(ctxplan.AssumptionReport{}); got != nil {
		t.Errorf("empty report assumptions got %+v, want nil", got)
	}
	if got := VerdictsFromReport(ctxplan.AssumptionReport{}); got != nil {
		t.Errorf("empty report verdicts got %+v, want nil", got)
	}
}
