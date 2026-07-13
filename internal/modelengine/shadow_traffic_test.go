package modelengine

import (
	"strings"
	"testing"
)

func TestShadowEvaluationFaithfulAndNoServingInfluence(t *testing.T) {
	live := shadowFixtureRequest()
	ref := shadowReferenceEngine()
	cand := shadowFaithfulCandidate()
	served := shadowServeLive(live, ref)
	got := shadowEvaluate(live, ref, cand)
	if !got.Verdict.Pass {
		t.Fatalf("faithful shadow must pass: %+v", got)
	}
	if strings.Join(got.Live, "|") != strings.Join(served, "|") {
		t.Fatalf("candidate influenced serving response: served=%v result=%v", served, got.Live)
	}
	if got.Verdict.Artifact != nil {
		t.Fatalf("passing run unexpectedly emitted failure artifact: %+v", got.Verdict.Artifact)
	}
}

func TestShadowEvaluationPlantedDefectsLocalizeAndNeverPass(t *testing.T) {
	live := shadowFixtureRequest()
	ref := shadowReferenceEngine()
	for _, cand := range shadowDefectiveCandidates() {
		t.Run(cand.name, func(t *testing.T) {
			got := shadowEvaluate(live, ref, cand)
			if got.Verdict.Pass {
				t.Fatalf("defect passed: %+v", got)
			}
			a := got.Verdict.Artifact
			if a == nil || a.Divergence == nil || a.FailPath == "" || !a.Provenance.complete() {
				t.Fatalf("defect not localized/provenanced: %+v", got)
			}
			for _, secret := range shadowSecretsOf(live) {
				if secret != "" && strings.Contains(a.String(), secret) {
					t.Fatalf("replay leaked secret %q: %s", secret, a.String())
				}
			}
			if strings.Join(got.Live, "|") != strings.Join(shadowServeLive(live, ref), "|") {
				t.Fatalf("shadow changed served response")
			}
		})
	}
}

func TestShadowJudgeMissingEvidenceIsInconclusive(t *testing.T) {
	v := shadowJudge(nil, nil, shadowProvenance{})
	if v.Pass {
		t.Fatalf("missing evidence passed: %+v", v)
	}
}
