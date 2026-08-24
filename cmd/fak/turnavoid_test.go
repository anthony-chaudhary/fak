package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/turnavoid"
)

func TestTurnavoidReplayFixtureProvesFiveRequiredCases(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runTurnavoid(strings.NewReader(""), &stdout, &stderr, []string{"replay", "--in", "testdata/turnavoid-replay.jsonl", "--json"})
	if code != 0 {
		t.Fatalf("turnavoid replay exit=%d, want 0\nstderr=%s", code, stderr.String())
	}
	var report turnavoid.Report
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("decode report: %v\n%s", err, stdout.String())
	}
	if report.SchemaVersion != turnavoid.ReportSchemaVersion || report.Rows != 18 || report.ImmutableInputs != 6 {
		t.Fatalf("report envelope = %q/%d/%d, want %q/18/6", report.SchemaVersion, report.Rows, report.ImmutableInputs, turnavoid.ReportSchemaVersion)
	}
	if report.HonestyVerdict != turnavoid.HonestWithheldCredit {
		t.Fatalf("honesty verdict = %q, want %q", report.HonestyVerdict, turnavoid.HonestWithheldCredit)
	}

	exact := findTurnavoidArm(t, report, turnavoid.ArmExactReuse)
	fused := findTurnavoidArm(t, report, turnavoid.ArmFusedBatch)

	// 1. The exact-match row has an equivalent independent effect witness and
	// receives one realized avoided-turn credit.
	exactMatch := findTurnavoidAttribution(t, exact, turnavoid.ReasonExactMatch)
	if exactMatch.RealizedTurnsAvoided != 1 || exactMatch.RequiredPreserved != 1 {
		t.Fatalf("exact-match attribution = avoided %d, preserved %d; want 1, 1", exactMatch.RealizedTurnsAvoided, exactMatch.RequiredPreserved)
	}

	// 2. The fused-batch arm collapses one serial model round trip.
	if fused.RealizedTurnsAvoided != 1 || fused.CandidateCommittedModelTurns != fused.ControlCommittedModelTurns-1 {
		t.Fatalf("fused report = avoided %d, committed %d/%d; want one-turn collapse", fused.RealizedTurnsAvoided, fused.CandidateCommittedModelTurns, fused.ControlCommittedModelTurns)
	}

	// 3. A cheaper retained provider-cache turn has separate token/latency
	// attribution and contributes no avoided turn.
	retained := findTurnavoidAttribution(t, exact, turnavoid.ReasonRetainedTurnCheaper)
	if retained.RealizedTurnsAvoided != 0 || exact.RetainedTurnsMadeCheaper != 1 || exact.RetainedTurnTokensReduced != 128 {
		t.Fatalf("retained report = avoided %d, turns %d, tokens %d; want 0, 1, 128", retained.RealizedTurnsAvoided, exact.RetainedTurnsMadeCheaper, exact.RetainedTurnTokensReduced)
	}

	// 4. Unsafe suppression and counterfactual-only opportunity each receive
	// zero realized credit, while their withheld opportunity remains visible.
	unsafe := findTurnavoidAttribution(t, exact, turnavoid.ReasonRequiredEffectSuppressed)
	counterfactual := findTurnavoidAttribution(t, exact, turnavoid.ReasonCounterfactualOnly)
	if unsafe.RealizedTurnsAvoided != 0 || unsafe.WithheldTurns != 1 || unsafe.RequiredSuppressed != 1 {
		t.Fatalf("unsafe attribution = avoided %d, withheld %d, suppressed %d; want 0, 1, 1", unsafe.RealizedTurnsAvoided, unsafe.WithheldTurns, unsafe.RequiredSuppressed)
	}
	if counterfactual.RealizedTurnsAvoided != 0 || counterfactual.WithheldTurns != 1 || exact.Lifecycle.CounterfactualOnly != 1 {
		t.Fatalf("counterfactual attribution = avoided %d, withheld %d, count %d; want 0, 1, 1", counterfactual.RealizedTurnsAvoided, counterfactual.WithheldTurns, exact.Lifecycle.CounterfactualOnly)
	}

	// 5. Validation overhead makes its row net-negative without erasing its
	// independently witnessed realized avoided turn.
	overhead := findTurnavoidAttribution(t, exact, turnavoid.ReasonValidationOverhead)
	if overhead.RealizedTurnsAvoided != 1 || overhead.Accounting.GrossLatencySavedMS != 100 || overhead.Accounting.NetLatencySavedMS != -900 {
		t.Fatalf("overhead attribution = avoided %d, gross/net %.1f/%.1f; want 1, 100/-900", overhead.RealizedTurnsAvoided, overhead.Accounting.GrossLatencySavedMS, overhead.Accounting.NetLatencySavedMS)
	}
	if exact.Accounting.NetLatencySavedMS >= 0 || exact.Accounting.NetCostSavedUSD >= 0 {
		t.Fatalf("exact arm net savings = %.1fms/$%.2f, want both negative", exact.Accounting.NetLatencySavedMS, exact.Accounting.NetCostSavedUSD)
	}
}

func TestTurnavoidReplayTextIsConciseAndCaptured(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runTurnavoid(strings.NewReader(""), &stdout, &stderr, []string{"replay", "--in", "testdata/turnavoid-replay.jsonl"})
	if code != 0 {
		t.Fatalf("turnavoid replay exit=%d, want 0\nstderr=%s", code, stderr.String())
	}
	text := stdout.String()
	for _, want := range []string{
		"HONEST_WITHHELD_CREDIT (6 immutable inputs, 18 rows)",
		"exact-reuse: committed=3 control=7 realized-avoided=2 withheld=2",
		"retained-cheaper=1",
		"effects preserved=6 suppressed=1",
		"latency-saved-ms gross=420.000 net=-590.000",
		"fused-batch: committed=6 control=7 realized-avoided=1",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("text report missing %q:\n%s", want, text)
		}
	}
}

func TestTurnavoidReplayRejectsMalformedInput(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runTurnavoid(strings.NewReader(`{"schema_version":"fak.turnavoid.trace/v1","unknown":true}`+"\n"), &stdout, &stderr, []string{"replay", "--in", "-", "--json"})
	if code != 2 || !strings.Contains(stderr.String(), "unknown field") {
		t.Fatalf("malformed replay exit=%d stderr=%q, want 2 + unknown field", code, stderr.String())
	}
	if stdout.Len() != 0 {
		t.Fatalf("malformed replay wrote stdout: %q", stdout.String())
	}
}

func findTurnavoidArm(t *testing.T, report turnavoid.Report, arm turnavoid.Arm) turnavoid.ArmReport {
	t.Helper()
	for _, candidate := range report.Arms {
		if candidate.Arm == arm {
			return candidate
		}
	}
	t.Fatalf("report missing arm %q", arm)
	return turnavoid.ArmReport{}
}

func findTurnavoidAttribution(t *testing.T, report turnavoid.ArmReport, reason turnavoid.Reason) turnavoid.Attribution {
	t.Helper()
	for _, attribution := range report.Attribution {
		if attribution.Reason == reason {
			return attribution
		}
	}
	t.Fatalf("arm %q missing attribution reason %q", report.Arm, reason)
	return turnavoid.Attribution{}
}
