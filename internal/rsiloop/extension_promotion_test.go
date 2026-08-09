package rsiloop

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func extensionFixture() ExtensionProposal {
	sum := sha256.Sum256([]byte("candidate-v1"))
	return ExtensionProposal{Kind: ExtensionSkill, Provenance: "agent/run-7", ClaimedMetric: "task-success", Scope: []string{"skills/demo"}, ArtifactDigest: hex.EncodeToString(sum[:]), RollbackRecipe: "remove skills/demo", CandidatePaths: []string{"skills/demo"}}
}

func passingEvidence() PromotionEvidence {
	return PromotionEvidence{IsolationRef: "worktree/7", WitnessRef: "ci/42", TestsPassed: true, TruthPassed: true, MetricMeasured: true, TunedBaseline: 0.7, CandidateMetric: 0.8}
}

func TestExtensionPromotionEndToEndFixtures(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*ExtensionProposal, *PromotionEvidence)
		want   PromotionVerdict
		reason string
	}{
		{"keep", nil, PromotionKeep, "strict net-true"},
		{"no gain", func(_ *ExtensionProposal, e *PromotionEvidence) { e.CandidateMetric = e.TunedBaseline }, PromotionRevert, "no strict net-true"},
		{"regression", func(_ *ExtensionProposal, e *PromotionEvidence) { e.CandidateMetric = 0.2 }, PromotionRevert, "no strict net-true"},
		{"missing witness", func(_ *ExtensionProposal, e *PromotionEvidence) { e.WitnessRef = "" }, PromotionRevert, "missing independent witness"},
		{"judge tampering", func(p *ExtensionProposal, _ *PromotionEvidence) {
			p.CandidatePaths = []string{"internal/rsiloop/judge"}
		}, PromotionRefuse, "overlaps witness-controlled"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, e := extensionFixture(), passingEvidence()
			if tt.mutate != nil {
				tt.mutate(&p, &e)
			}
			got, reason := EvaluateExtensionProposal(p, e, []string{"internal/rsiloop/judge", "internal/rsiloop/verifier", "internal/rsiloop/metric", "internal/rsiloop/policy"})
			if got != tt.want || !strings.Contains(reason, tt.reason) {
				t.Fatalf("got (%s,%q), want %s containing %q", got, reason, tt.want, tt.reason)
			}
		})
	}
}

func TestRunExtensionPromotionPersistsFinalReceipt(t *testing.T) {
	p := extensionFixture()
	calls := 0
	got, err := RunExtensionPromotion(t.TempDir(), p, []string{"internal/rsiloop/judge"}, func(ExtensionProposal) PromotionEvidence { calls++; return passingEvidence() })
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 || got.Verdict != PromotionKeep || got.State != "FINAL" {
		t.Fatalf("unexpected run: calls=%d receipt=%+v", calls, got)
	}
}

func TestRunExtensionPromotionMissingWitnessFinalizesRevert(t *testing.T) {
	p := extensionFixture()
	got, err := RunExtensionPromotion(t.TempDir(), p, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got.State != "FINAL" || got.Verdict != PromotionRevert || !strings.Contains(got.Reason, "missing independent witness") {
		t.Fatalf("nil witness did not fail closed: %+v", got)
	}
}

func TestExtensionPromotionAlwaysProtectsJudgePaths(t *testing.T) {
	p := extensionFixture()
	p.CandidatePaths = []string{"internal/rsiloop/metric/override.go"}
	got, reason := EvaluateExtensionProposal(p, passingEvidence(), nil)
	if got != PromotionRefuse || !strings.Contains(reason, "overlaps witness-controlled") {
		t.Fatalf("built-in judge protection bypassed: (%s, %q)", got, reason)
	}
}

func TestRunExtensionPromotionCrashLeavesPreparedNotKept(t *testing.T) {
	p := extensionFixture()
	dir := t.TempDir()
	panicValue := ""
	func() {
		defer func() {
			if v := recover(); v != nil {
				panicValue = v.(string)
			}
		}()
		_, _ = RunExtensionPromotion(dir, p, nil, func(ExtensionProposal) PromotionEvidence { panic("crash") })
	}()
	if panicValue != "crash" {
		t.Fatalf("witness panic not observed: %q", panicValue)
	}
	b, err := os.ReadFile(filepath.Join(dir, p.ArtifactDigest+".json"))
	if err != nil {
		t.Fatal(err)
	}
	var r PromotionReceipt
	if err = json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	if r.State != "PREPARED" || r.Verdict == PromotionKeep {
		t.Fatalf("crash receipt could be mistaken for keep: %+v", r)
	}
}

func TestExtensionProposalEnvelopeRejectsIncompleteAndBadDigest(t *testing.T) {
	p := extensionFixture()
	p.RollbackRecipe = ""
	if got, _ := EvaluateExtensionProposal(p, passingEvidence(), nil); got != PromotionRefuse {
		t.Fatalf("incomplete proposal got %s", got)
	}
	p = extensionFixture()
	p.ArtifactDigest = "author-says-valid"
	if got, _ := EvaluateExtensionProposal(p, passingEvidence(), nil); got != PromotionRefuse {
		t.Fatalf("bad digest got %s", got)
	}
}
