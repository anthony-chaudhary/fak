package turnavoid

import (
	"strings"
	"testing"
)

func TestFoldCreditsOnlyEquivalentRealizedWholeTurns(t *testing.T) {
	tests := []struct {
		name      string
		arm       Arm
		configure func(*Row)
		avoided   int
		withheld  int
	}{
		{
			name: "exact reuse",
			arm:  ArmExactReuse,
			configure: func(row *Row) {
				row.Lifecycle, row.Reason = LifecycleRealized, ReasonExactMatch
				row.CandidateCommittedTurns, row.CandidateGross = 0, GrossWork{}
			},
			avoided: 1,
		},
		{
			name: "fused serial round trip",
			arm:  ArmFusedBatch,
			configure: func(row *Row) {
				row.Lifecycle, row.Reason = LifecycleRealized, ReasonSerialRoundTripCollapsed
				row.CandidateCommittedTurns = 1
				row.CandidateGross = GrossWork{ModelLatencyMS: 120, ModelCostUSD: 1.2}
			},
			avoided: 1,
		},
		{
			name: "unsafe suppression",
			arm:  ArmExactReuse,
			configure: func(row *Row) {
				row.Lifecycle, row.Reason = LifecycleRealized, ReasonRequiredEffectSuppressed
				row.CandidateCommittedTurns, row.CandidateGross = 0, GrossWork{}
				row.Effects.CandidateRequired = nil
			},
			withheld: 1,
		},
		{
			name: "counterfactual only",
			arm:  ArmExactReuse,
			configure: func(row *Row) {
				row.Lifecycle, row.Reason = LifecycleCounterfactualOnly, ReasonCounterfactualOnly
				row.CandidateCommittedTurns, row.CandidateGross = 0, GrossWork{}
			},
			withheld: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			turns := 1
			if tt.arm == ArmFusedBatch {
				turns = 2
			}
			rows := fixtureGroup(tt.name, turns)
			row := fixtureArm(rows, tt.arm)
			tt.configure(row)
			report := mustFoldReport(t, rows, tt.arm)
			if report.RealizedTurnsAvoided != tt.avoided || report.WithheldTurns != tt.withheld {
				t.Fatalf("avoided/withheld = %d/%d, want %d/%d", report.RealizedTurnsAvoided, report.WithheldTurns, tt.avoided, tt.withheld)
			}
		})
	}
}

func TestFoldRetainedTurnIsCheaperButNotAvoided(t *testing.T) {
	rows := fixtureGroup("retained", 1)
	exact := fixtureArm(rows, ArmExactReuse)
	exact.Mechanism, exact.Lifecycle, exact.Reason = MechanismProviderCache, LifecycleRealized, ReasonRetainedTurnCheaper
	exact.CandidateGross = GrossWork{ModelLatencyMS: 80, ModelCostUSD: .8}
	exact.RetainedTurnReduction = &RetainedTurnReduction{Tokens: 128, LatencyMS: 20}
	report := mustFoldReport(t, rows, ArmExactReuse)
	if report.RealizedTurnsAvoided != 0 || report.RetainedTurnsMadeCheaper != 1 || report.RetainedTurnTokensReduced != 128 {
		t.Fatalf("avoided/retained/tokens = %d/%d/%d, want 0/1/128", report.RealizedTurnsAvoided, report.RetainedTurnsMadeCheaper, report.RetainedTurnTokensReduced)
	}
}

func TestFoldChargesOverheadWithoutErasingRealizedTurn(t *testing.T) {
	rows := fixtureGroup("overhead", 1)
	exact := fixtureArm(rows, ArmExactReuse)
	exact.Lifecycle, exact.Reason = LifecycleRealized, ReasonValidationOverhead
	exact.CandidateCommittedTurns, exact.CandidateGross = 0, GrossWork{}
	exact.Overhead = Overhead{ValidationLatencyMS: 250, ValidationCostUSD: 2.5}
	report := mustFoldReport(t, rows, ArmExactReuse)
	if report.RealizedTurnsAvoided != 1 || report.Accounting.GrossLatencySavedMS != 100 || report.Accounting.NetLatencySavedMS != -150 {
		t.Fatalf("avoided/gross/net = %d/%.1f/%.1f, want 1/100/-150", report.RealizedTurnsAvoided, report.Accounting.GrossLatencySavedMS, report.Accounting.NetLatencySavedMS)
	}
}

func TestFoldRejectsCrossArmDriftAndSameTurnLeakage(t *testing.T) {
	tests := []struct {
		name string
		edit func([]Row)
		want string
	}{
		{"digest drift", func(rows []Row) { rows[1].InputDigest = "sha256:" + strings.Repeat("b", 64) }, "input_digest differs"},
		{"same-turn leakage", func(rows []Row) { rows[1].DecisionBasisThroughTurn = 0 }, "same-turn result leakage"},
		{"unknown state", func(rows []Row) { rows[1].Lifecycle = "REALIZED" }, "unknown lifecycle"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rows := fixtureGroup("invalid", 1)
			tt.edit(rows)
			_, err := Fold(rows)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Fold error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestReplayRejectsUnknownJSONField(t *testing.T) {
	_, err := Replay(strings.NewReader(`{"schema_version":"fak.turnavoid.trace/v1","surprise":true}` + "\n"))
	if err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("Replay error = %v, want unknown field", err)
	}
}

func fixtureGroup(unit string, turns int) []Row {
	gross := GrossWork{ModelLatencyMS: 100 * float64(turns), ModelCostUSD: float64(turns)}
	base := Row{
		SchemaVersion:            TraceSchemaVersion,
		TraceID:                  "trace-1",
		UnitID:                   unit,
		TurnIndex:                0,
		DecisionBasisThroughTurn: -1,
		InputDigest:              "sha256:" + strings.Repeat("a", 64),
		ControlCommittedTurns:    turns,
		CandidateCommittedTurns:  turns,
		Effects: EffectObservation{
			IndependentObserver: "fixture-oracle/v1",
			ControlRequired:     []string{"answer"},
			CandidateRequired:   []string{"answer"},
		},
		ControlGross:   gross,
		CandidateGross: gross,
	}
	control, exact, fused := base, base, base
	control.Arm, control.Mechanism, control.Lifecycle, control.Reason = ArmControl, MechanismControl, LifecycleRealized, ReasonBaseline
	exact.Arm, exact.Mechanism, exact.Lifecycle, exact.Reason = ArmExactReuse, MechanismExactReuse, LifecycleOpportunity, ReasonNotApplicable
	fused.Arm, fused.Mechanism, fused.Lifecycle, fused.Reason = ArmFusedBatch, MechanismFusedBatch, LifecycleOpportunity, ReasonNotApplicable
	return []Row{control, exact, fused}
}

func fixtureArm(rows []Row, arm Arm) *Row {
	for i := range rows {
		if rows[i].Arm == arm {
			return &rows[i]
		}
	}
	panic("fixture missing arm")
}

func mustFoldReport(t *testing.T, rows []Row, arm Arm) ArmReport {
	t.Helper()
	report, err := Fold(rows)
	if err != nil {
		t.Fatal(err)
	}
	for _, candidate := range report.Arms {
		if candidate.Arm == arm {
			return candidate
		}
	}
	t.Fatalf("report missing arm %q", arm)
	return ArmReport{}
}
