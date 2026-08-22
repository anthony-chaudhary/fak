package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSelfcheckFindsKnownCliffAndRefusesMissingEvidence(t *testing.T) {
	manifest := testManifest(t)
	witness, err := BuildSelfcheck(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if got := witness.KnownCliff.Boundary.DeepestReliablePrefixTokens; got == nil || *got != 8192 {
		t.Fatalf("deepest reusable prefix = %v, want 8192", got)
	}
	cliff := witness.KnownCliff.Boundary.Cliff
	if cliff == nil || cliff.ReliableThroughTokens != 8192 || cliff.UnreliableAtTokens != 12288 {
		t.Fatalf("cliff = %+v, want 8192..12288", cliff)
	}
	if witness.KnownCliff.PressureRecovery.Status != "recovered" {
		t.Fatalf("pressure recovery = %+v", witness.KnownCliff.PressureRecovery)
	}
	if witness.MissingKVData.Boundary.Status != "unknown" || witness.MissingKVData.Boundary.DeepestReliablePrefixTokens != nil {
		t.Fatalf("missing evidence invented a boundary: %+v", witness.MissingKVData.Boundary)
	}
	for _, point := range witness.MissingKVData.DepthCurve {
		if point.PrefillSavedTokens != nil || point.ReuseRatio != nil || point.KVEvidenceSamples != 0 {
			t.Fatalf("missing evidence became zero-valued evidence: %+v", point)
		}
	}
}

func TestManifestAndObservedEnvelopeMeetAcceptance(t *testing.T) {
	manifest := testManifest(t)
	observations, err := SyntheticObservations(manifest, true)
	if err != nil {
		t.Fatal(err)
	}
	report, err := Analyze(manifest, observations)
	if err != nil {
		t.Fatal(err)
	}
	if len(manifest.Axes.PrefixDepthTokens) < 6 || len(manifest.Axes.SuffixPatterns) < 2 || manifest.Axes.Repetitions < 3 {
		t.Fatalf("manifest target envelope is too small: %+v", manifest.Axes)
	}
	if !report.Envelope.Complete || report.Envelope.PrefixDepths != 6 || report.Envelope.SuffixPatterns != 2 || report.Envelope.TurnCounts != 2 || report.Envelope.ConcurrencyValues != 2 || report.Envelope.PressurePhases != 3 || report.Envelope.MinimumRepetitionsPerArm != 3 || !report.Envelope.Counterbalanced {
		t.Fatalf("observed envelope = %+v", report.Envelope)
	}
	if report.Evidence.SemanticPromptEqual == report.Evidence.WarmRequests {
		t.Fatalf("suffix churn did not separate semantic equality: %+v", report.Evidence)
	}
	if report.Evidence.TokenPrefixEqual != report.Evidence.WarmRequests || report.Evidence.BackendStatusPresent == 0 || report.Evidence.ReuseMeasurementPresent == 0 {
		t.Fatalf("evidence dimensions were conflated or omitted: %+v", report.Evidence)
	}
}

func TestAnalyzeRejectsPinDrift(t *testing.T) {
	manifest := testManifest(t)
	observations, err := SyntheticObservations(manifest, true)
	if err != nil {
		t.Fatal(err)
	}
	observations[0].Pins.RuntimeRevision = "different-runtime"
	if _, err := Analyze(manifest, observations); err == nil || !strings.Contains(err.Error(), "pins differ") {
		t.Fatalf("pin drift error = %v", err)
	}
}

func TestAnalyzeRejectsOrderingThatLetsWarmupChooseTheResult(t *testing.T) {
	manifest := testManifest(t)
	observations, err := SyntheticObservations(manifest, true)
	if err != nil {
		t.Fatal(err)
	}
	for i := range observations {
		if observations[i].Repetition == 2 && observations[i].ThermalState == "cold" {
			observations[i].OrderIndex = 1
		}
	}
	if _, err := Analyze(manifest, observations); err == nil || !strings.Contains(err.Error(), "observed envelope incomplete") {
		t.Fatalf("counterbalance error = %v", err)
	}
}

func TestRequestLevelFixturesRemainReplayable(t *testing.T) {
	manifest := testManifest(t)
	for _, fixture := range []struct {
		name       string
		wantStatus string
	}{
		{name: "known-cliff.jsonl", wantStatus: "known"},
		{name: "missing-cache-evidence.jsonl", wantStatus: "unknown"},
	} {
		observations, err := readObservations(filepath.Join("testdata", fixture.name))
		if err != nil {
			t.Fatal(err)
		}
		if len(observations) == 0 || observations[0].RequestID == "" || observations[0].PromptTokens == 0 || observations[0].TTFTMillis == 0 {
			t.Fatalf("%s did not preserve request-level evidence", fixture.name)
		}
		report, err := Analyze(manifest, observations)
		if err != nil {
			t.Fatal(err)
		}
		if report.Boundary.Status != fixture.wantStatus {
			t.Fatalf("%s boundary = %+v", fixture.name, report.Boundary)
		}
	}
}

func TestCapturedWitnessMatchesAnalyzer(t *testing.T) {
	manifest := testManifest(t)
	witness, err := BuildSelfcheck(manifest)
	if err != nil {
		t.Fatal(err)
	}
	want, err := json.MarshalIndent(witness, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	want = append(want, '\n')
	got, err := os.ReadFile(filepath.Join("testdata", "known-cliff-witness.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("captured witness drifted; regenerate with -emit-fixtures after reviewing the analyzer change")
	}
}

func TestRunSelfcheckUsesCheckedInManifest(t *testing.T) {
	var stdout, stderr bytes.Buffer
	exit := run(&stdout, &stderr, []string{"-manifest", filepath.Join("testdata", "campaign.json"), "-selfcheck"})
	if exit != 0 || stderr.Len() != 0 || !strings.Contains(stdout.String(), `"known_deepest_reusable_prefix_tokens": 8192`) || !strings.Contains(stdout.String(), `"status": "unknown"`) {
		t.Fatalf("exit=%d stdout=%q stderr=%q", exit, stdout.String(), stderr.String())
	}
}

func TestRunRequiresOneAnalysisMode(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if exit := run(&stdout, &stderr, []string{"-manifest", filepath.Join("testdata", "campaign.json")}); exit != 2 || !strings.Contains(stderr.String(), "usage:") {
		t.Fatalf("exit=%d stderr=%q", exit, stderr.String())
	}
}

func testManifest(t *testing.T) Manifest {
	t.Helper()
	manifest, err := readManifest(filepath.Join("testdata", "campaign.json"))
	if err != nil {
		t.Fatal(err)
	}
	return manifest
}
