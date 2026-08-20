package modelreg

import (
	"strings"
	"testing"
	"time"
)

var defaultEvidenceNow = time.Date(2026, 8, 20, 12, 0, 0, 0, time.UTC)

func TestEvaluateDefaultEvidencePromotesCompleteFreshInputs(t *testing.T) {
	got := EvaluateDefaultEvidence(passingDefaultEvidence(), defaultEvidenceNow)
	if got.Verdict != VerdictPromote || len(got.Reasons) != 0 || len(got.EvidenceRefs) != 5 {
		t.Fatalf("evaluation = %+v", got)
	}
}

func TestEvaluateDefaultEvidenceHoldsEveryRequiredMissingFamily(t *testing.T) {
	tests := map[string]func(*DefaultEvidenceInput){
		"macbook":    func(in *DefaultEvidenceInput) { in.MacBook = HardwareInput{State: StateMissing} },
		"nvidia":     func(in *DefaultEvidenceInput) { in.NVIDIA = HardwareInput{State: StateMissing} },
		"cache":      func(in *DefaultEvidenceInput) { in.Cache = ReuseInput{State: StateMissing} },
		"comparison": func(in *DefaultEvidenceInput) { in.Comparison = FrontierInput{State: StateMissing} },
		"support":    func(in *DefaultEvidenceInput) { in.Support = FreshnessInput{State: StateMissing} },
	}
	for family, mutate := range tests {
		t.Run(family, func(t *testing.T) {
			in := passingDefaultEvidence()
			mutate(&in)
			got := EvaluateDefaultEvidence(in, defaultEvidenceNow)
			if got.Verdict != VerdictHold || !hasEvaluationReason(got, family, "MISSING_EVIDENCE") {
				t.Fatalf("evaluation = %+v", got)
			}
		})
	}
}

func TestEvaluateDefaultEvidenceRollsBackActiveDefaultAtNamedThreshold(t *testing.T) {
	in := passingDefaultEvidence()
	in.CurrentlyDefault = true
	in.PreviousDefault = "qwen35:27b"
	in.Cache.NetSavingsPercent = 0
	got := EvaluateDefaultEvidence(in, defaultEvidenceNow)
	if got.Verdict != VerdictRollback || got.PreviousDefault != "qwen35:27b" || !hasEvaluationReason(got, "cache", "NET_REUSE_VALUE_FAILED") {
		t.Fatalf("evaluation = %+v", got)
	}
}

func TestEvaluateDefaultEvidenceHoldsExpiredInput(t *testing.T) {
	in := passingDefaultEvidence()
	in.Support.Artifact.StaleAfter = defaultEvidenceNow.Format(time.RFC3339)
	got := EvaluateDefaultEvidence(in, defaultEvidenceNow)
	if got.Verdict != VerdictHold || !hasEvaluationReason(got, "support", "STALE_EVIDENCE") {
		t.Fatalf("evaluation = %+v", got)
	}
}

func TestEvaluateDefaultEvidenceFailsClosedOnContradictoryOrSubstitutedInputs(t *testing.T) {
	tests := map[string]func(*DefaultEvidenceInput){
		"contradictory": func(in *DefaultEvidenceInput) { in.Comparison.Decision = VerdictHold },
		"substituted":   func(in *DefaultEvidenceInput) { in.NVIDIA.Artifact.Identity.Revision = "other" },
		"wrong-source":  func(in *DefaultEvidenceInput) { in.Cache.Artifact.SourceIssue = 8061 },
		"unproven":      func(in *DefaultEvidenceInput) { in.Cache.Artifact.SHA256 = "not-a-hash" },
		"na":            func(in *DefaultEvidenceInput) { in.MacBook.State = StateNA },
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			in := passingDefaultEvidence()
			mutate(&in)
			if got := EvaluateDefaultEvidence(in, defaultEvidenceNow).Verdict; got == VerdictPromote {
				t.Fatalf("contradictory input promoted")
			}
		})
	}
}

func passingDefaultEvidence() DefaultEvidenceInput {
	identity := DefaultCheckpoint{Alias: DefaultAlias, Ref: DefaultRef(), Revision: "rev-immutable", CheckpointSHA256: strings.Repeat("1", 64), TokenizerSHA256: strings.Repeat("2", 64), ChatTemplateSHA256: strings.Repeat("3", 64), Quant: qwen38DefaultQuant}
	ref := func(issue int, backend string, digit string) ArtifactRef {
		return ArtifactRef{SourceIssue: issue, URI: "docs/_witnesses/qwen38/" + digit + ".json", SHA256: strings.Repeat(digit, 64), ObservedAt: defaultEvidenceNow.Add(-time.Hour).Format(time.RFC3339), StaleAfter: defaultEvidenceNow.Add(time.Hour).Format(time.RFC3339), Backend: backend, Identity: identity}
	}
	return DefaultEvidenceInput{
		Schema:     DefaultEvidenceSchema,
		Candidate:  identity,
		MacBook:    HardwareInput{State: StatePass, Artifact: ref(8061, "metal", "4"), TextOK: true, JSONOK: true, ToolOK: true, NoFallback: true},
		NVIDIA:     HardwareInput{State: StatePass, Artifact: ref(8061, "cuda", "5"), TextOK: true, JSONOK: true, ToolOK: true, NoFallback: true},
		Cache:      ReuseInput{State: StatePass, Artifact: ref(8127, "metal", "6"), SemanticallyEquivalent: true, NetSavingsPercent: 1},
		Comparison: FrontierInput{State: StatePass, Artifact: ref(8128, "mixed", "7"), Alternatives: 10, QualityPassed: true, Decision: VerdictPromote},
		Support:    FreshnessInput{State: StatePass, Artifact: ref(8129, "mixed", "8"), Fresh: true},
	}
}

func hasEvaluationReason(got DefaultEvaluation, family, code string) bool {
	for _, reason := range got.Reasons {
		if reason.Family == family && reason.Code == code {
			return true
		}
	}
	return false
}
