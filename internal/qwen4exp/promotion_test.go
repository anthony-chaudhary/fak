package qwen4exp

import (
	"testing"
	"time"
)

func goodEvidence(now time.Time, b string) EnvelopeEvidence {
	return EnvelopeEvidence{Backend: b, Artifact: "sha256:exact", Engine: "fak-native", Fallback: "none", Quality: true, Text: true, StructuredJSON: true, CorrelatedTools: true, Continues: true, CapturedAt: now.Add(-time.Hour), ExpiresAt: now.Add(time.Hour)}
}
func TestPromotionRequiresCUDAAndMetal(t *testing.T) {
	now := time.Now()
	v := EvaluatePromotion(PromotionInput{ExpectedArtifact: "sha256:exact", Now: now, Envelopes: []EnvelopeEvidence{goodEvidence(now, "cuda")}})
	if v.State != PromotionHold {
		t.Fatal(v)
	}
	v = EvaluatePromotion(PromotionInput{ExpectedArtifact: "sha256:exact", Now: now, Envelopes: []EnvelopeEvidence{goodEvidence(now, "cuda"), goodEvidence(now, "metal")}})
	if v.State != PromotionReady {
		t.Fatal(v)
	}
}
func TestPromotionRollsBackStaleOrWrongIdentity(t *testing.T) {
	now := time.Now()
	bad := goodEvidence(now, "cuda")
	bad.Engine = "mlx"
	v := EvaluatePromotion(PromotionInput{ExpectedArtifact: "sha256:exact", Now: now, Envelopes: []EnvelopeEvidence{bad, goodEvidence(now, "metal")}})
	if v.State != PromotionRollback {
		t.Fatal(v)
	}
}
func TestDenseQwenAliasIsNeverUsed(t *testing.T) {
	v := EvaluatePromotion(PromotionInput{})
	for _, a := range v.Aliases {
		if a == "Qwen3.8-27B" || a == "qwen3.8" {
			t.Fatal("dense alias conflated")
		}
	}
}
