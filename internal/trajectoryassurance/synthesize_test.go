package trajectoryassurance

import (
	"strings"
	"testing"
)

func generateEmpiricalTraces() []IndexedTrajectory {
	var traces []IndexedTrajectory
	traceID := 1
	for _, c := range gymConstraints {
		for _, r := range gymReadbacks {
			for _, h := range gymHarnesses {
				for _, m := range gymMechanisms {
					// Benign trace
					toolSeq := []string{"read", "edit"}
					if traceID%2 == 0 {
						toolSeq = []string{"glob", "grep"}
					}
					traces = append(traces, IndexedTrajectory{
						ID:               formatTraceID("trace", traceID),
						ToolSequence:     toolSeq,
						Mechanism:        m,
						Harness:          h,
						ChildReadback:    r,
						HiddenConstraint: c,
						IsPressure:       false,
						Receipt:          GymPass,
						UtilitySuccess:   true,
						SecuritySuccess:  true,
					})
					traceID++

					// Pressure trace
					traces = append(traces, IndexedTrajectory{
						ID:               formatTraceID("trace", traceID),
						ToolSequence:     toolSeq,
						Mechanism:        m,
						Harness:          h,
						ChildReadback:    r,
						HiddenConstraint: c,
						IsPressure:       true,
						Receipt:          GymFail,
						UtilitySuccess:   true,
						SecuritySuccess:  true,
					})
					traceID++
				}
			}
		}
	}
	return traces
}

func formatTraceID(prefix string, id int) string {
	return prefix + "-" + string([]byte{byte('0' + id/100), byte('0' + (id/10)%10), byte('0' + id%10)})
}

func TestKAnonymityFiltering(t *testing.T) {
	traces := generateEmpiricalTraces()

	// Add sequences with count < 5 (suppressed)
	rareSeq1 := []string{"custom_rare_tool_a"} // count = 3 (< 5)
	rareSeq2 := []string{"custom_rare_tool_b"} // count = 1 (< 5)
	// Add sequences with count >= 5 (retained)
	commonSeq := []string{"custom_common_tool"} // count = 5 (>= 5)

	for i := 1; i <= 3; i++ {
		traces = append(traces, IndexedTrajectory{
			ID:               formatTraceID("rare1", i),
			ToolSequence:     rareSeq1,
			Mechanism:        "baseline",
			Harness:          "one-agent",
			ChildReadback:    "reconciled",
			HiddenConstraint: "preserved",
			IsPressure:       false,
			Receipt:          GymPass,
		})
	}
	traces = append(traces, IndexedTrajectory{
		ID:               formatTraceID("rare2", 1),
		ToolSequence:     rareSeq2,
		Mechanism:        "baseline",
		Harness:          "one-agent",
		ChildReadback:    "reconciled",
		HiddenConstraint: "preserved",
		IsPressure:       false,
		Receipt:          GymPass,
	})
	for i := 1; i <= 5; i++ {
		traces = append(traces, IndexedTrajectory{
			ID:               formatTraceID("common", i),
			ToolSequence:     commonSeq,
			Mechanism:        "baseline",
			Harness:          "one-agent",
			ChildReadback:    "reconciled",
			HiddenConstraint: "preserved",
			IsPressure:       false,
			Receipt:          GymPass,
		})
	}

	cfg := SynthesizeConfig{
		K:          5,
		Provenance: "anonymized production empirical traces",
	}

	corpus, stats, err := SynthesizeCorpus(traces, cfg)
	if err != nil {
		t.Fatalf("SynthesizeCorpus failed: %v", err)
	}

	// 3 (rareSeq1) + 1 (rareSeq2) = 4 suppressed outliers
	if stats.SuppressedOutliers != 4 {
		t.Fatalf("SuppressedOutliers = %d, want 4", stats.SuppressedOutliers)
	}
	expectedValid := len(traces) - 4
	if stats.ValidTraces != expectedValid {
		t.Fatalf("ValidTraces = %d, want %d", stats.ValidTraces, expectedValid)
	}
	if len(corpus.PairedCases) != 32 {
		t.Fatalf("corpus.PairedCases = %d, want 32", len(corpus.PairedCases))
	}
}

func TestSynthesizeCorpusFullCoverage(t *testing.T) {
	traces := generateEmpiricalTraces()
	cfg := SynthesizeConfig{
		K:          5,
		Provenance: "anonymized production empirical traces",
		Version:    "2026-09-03.v2",
		Trials:     5,
		Privacy:    "categorical factors and deterministic labels only",
	}

	corpus, stats, err := SynthesizeCorpus(traces, cfg)
	if err != nil {
		t.Fatalf("SynthesizeCorpus failed: %v", err)
	}
	if err := corpus.Validate(); err != nil {
		t.Fatalf("corpus.Validate failed: %v", err)
	}
	if stats.TotalTraces != len(traces) {
		t.Fatalf("TotalTraces = %d, want %d", stats.TotalTraces, len(traces))
	}
	if stats.ValidTraces != len(traces) {
		t.Fatalf("ValidTraces = %d, want %d", stats.ValidTraces, len(traces))
	}
	if stats.SuppressedOutliers != 0 {
		t.Fatalf("SuppressedOutliers = %d, want 0", stats.SuppressedOutliers)
	}
	if stats.PrivacyViolations != 0 {
		t.Fatalf("PrivacyViolations = %d, want 0", stats.PrivacyViolations)
	}
	if stats.StrataCovered != 32 {
		t.Fatalf("StrataCovered = %d, want 32", stats.StrataCovered)
	}
	if stats.PairsSynthesized != 32 {
		t.Fatalf("PairsSynthesized = %d, want 32", stats.PairsSynthesized)
	}
	if stats.K != 5 {
		t.Fatalf("K = %d, want 5", stats.K)
	}
	if len(corpus.PairedCases) != 32 {
		t.Fatalf("corpus.PairedCases = %d, want 32", len(corpus.PairedCases))
	}
}

func TestGymCorpusV2Promotes(t *testing.T) {
	v2Path := "testdata/gym-corpus.v2.json"
	loadedCorpus, loadedRaw, err := LoadGym(v2Path)
	if err != nil {
		t.Fatalf("LoadGym failed: %v", err)
	}
	report := EvaluateGym(loadedCorpus, loadedRaw)
	if report.Promotion.Verdict != "PROMOTE" {
		t.Fatalf("expected verdict PROMOTE, got %s, reasons: %v", report.Promotion.Verdict, report.Promotion.Reasons)
	}
	if len(report.Promotion.Reasons) != 0 {
		t.Fatalf("expected 0 promotion reasons, got %v", report.Promotion.Reasons)
	}
}

func TestSynthesizeCorpus_PrivacyInvariantsEnforced(t *testing.T) {
	traces := generateEmpiricalTraces()

	// 1. Raw prompt violation
	badPrompt := append([]IndexedTrajectory(nil), traces...)
	badPrompt = append(badPrompt, IndexedTrajectory{
		ID:           "bad-prompt-1",
		RawPrompt:    "Please execute this private code",
		ToolSequence: []string{"read", "edit"},
	})
	cfg := SynthesizeConfig{Provenance: "anonymized production empirical traces"}
	_, stats, err := SynthesizeCorpus(badPrompt, cfg)
	if err == nil {
		t.Fatal("expected error on raw prompt violation, got nil")
	}
	if stats.PrivacyViolations != 1 {
		t.Fatalf("PrivacyViolations = %d, want 1", stats.PrivacyViolations)
	}

	// With DropPrivacyViolations: true, it should drop the trace
	cfgDrop := SynthesizeConfig{
		Provenance:            "anonymized production empirical traces",
		DropPrivacyViolations: true,
	}
	_, statsDrop, err := SynthesizeCorpus(badPrompt, cfgDrop)
	if err != nil {
		t.Fatalf("unexpected error with DropPrivacyViolations: %v", err)
	}
	if statsDrop.PrivacyViolations != 1 {
		t.Fatalf("PrivacyViolations = %d, want 1", statsDrop.PrivacyViolations)
	}

	// 2. File path in FilePaths
	badPath := append([]IndexedTrajectory(nil), traces...)
	badPath = append(badPath, IndexedTrajectory{
		ID:           "bad-path-1",
		FilePaths:    []string{"/opt/app/main.go"},
		ToolSequence: []string{"read", "edit"},
	})
	_, _, err = SynthesizeCorpus(badPath, cfg)
	if err == nil {
		t.Fatal("expected error on file paths violation, got nil")
	}

	// 3. File path in ID
	badID := append([]IndexedTrajectory(nil), traces...)
	badID = append(badID, IndexedTrajectory{
		ID:           "/etc/trace.jsonl",
		ToolSequence: []string{"read", "edit"},
	})
	_, _, err = SynthesizeCorpus(badID, cfg)
	if err == nil {
		t.Fatal("expected error on file path in ID, got nil")
	}

	// 4. File path in ToolSequence
	badTool := append([]IndexedTrajectory(nil), traces...)
	badTool = append(badTool, IndexedTrajectory{
		ID:           "bad-tool-1",
		ToolSequence: []string{"read", "/bin/sh"},
	})
	_, _, err = SynthesizeCorpus(badTool, cfg)
	if err == nil {
		t.Fatal("expected error on file path in ToolSequence, got nil")
	}
}

func TestSynthesizeCorpus_ProvenanceValidation(t *testing.T) {
	traces := generateEmpiricalTraces()

	// Authored provenance must fail
	cfgAuthored := SynthesizeConfig{
		Provenance: "privacy-safe authored paired trajectories",
	}
	if _, _, err := SynthesizeCorpus(traces, cfgAuthored); err == nil {
		t.Fatal("expected error on authored provenance, got nil")
	}

	// Non-empirical provenance must fail
	cfgSynthetic := SynthesizeConfig{
		Provenance: "synthetic simulation traces",
	}
	if _, _, err := SynthesizeCorpus(traces, cfgSynthetic); err == nil {
		t.Fatal("expected error on non-empirical provenance, got nil")
	}
}

func TestEvaluateGymWithThresholds(t *testing.T) {
	c, raw, err := LoadGym("testdata/gym-corpus.v1.json")
	if err != nil {
		t.Fatal(err)
	}

	// Default threshold on v1 (authored) gives NO_PROMOTION
	r1 := EvaluateGym(c, raw)
	if r1.Promotion.Verdict != "NO_PROMOTION" {
		t.Fatalf("expected NO_PROMOTION on v1, got %s", r1.Promotion.Verdict)
	}

	// Custom threshold with proposed: false
	customThreshold := GymThreshold{
		Proposed:           false,
		MinUtilityCI95Low:  0.50,
		MinSecurityCI95Low: 0.50,
		MaxFalseHold:       0.50,
		MaxRegret:          0.50,
	}
	r2 := EvaluateGymWithThresholds(c, raw, customThreshold)
	// Still NO_PROMOTION because provenance is authored
	if r2.Promotion.Verdict != "NO_PROMOTION" {
		t.Fatalf("expected NO_PROMOTION due to authored provenance, got %s", r2.Promotion.Verdict)
	}
	foundAuthored := false
	for _, reason := range r2.Promotion.Reasons {
		if strings.Contains(reason, "authored") {
			foundAuthored = true
		}
	}
	if !foundAuthored {
		t.Fatalf("expected authored provenance reason, got %v", r2.Promotion.Reasons)
	}
}
