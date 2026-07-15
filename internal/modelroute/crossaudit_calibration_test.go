package modelroute

import (
	"strings"
	"testing"
)

func calibrationFixture(t *testing.T) ([]CalibrationTruth, []CalibrationObservation) {
	t.Helper()
	fixtures := AccidentalCorpus()[:4]
	truth := make([]CalibrationTruth, 0, len(fixtures))
	obs := make([]CalibrationObservation, 0, len(fixtures)*2)
	for i, f := range fixtures {
		b, err := f.Bundle()
		if err != nil {
			t.Fatal(err)
		}
		truth = append(truth, CalibrationTruth{ID: f.ID, Class: f.Class, Corrupt: f.Corrupt, BundleDigest: b.BundleDigest})
		verdict := CrossAuditPass
		if f.Corrupt {
			verdict = CrossAuditRefute
		}
		for _, aud := range []AuditIdentity{{Provider: "p1", Family: "f1", Model: "m1", WeightsRevision: "r1", ReasoningPosture: "xhigh", Driver: "codex"}, {Provider: "local", Family: "f2", Model: "m2", WeightsRevision: "sha256:x", ReasoningPosture: "provider-default", Driver: "http"}} {
			v := verdict
			if aud.Model == "m2" && i == 0 {
				v = CrossAuditPass
			}
			obs = append(obs, CalibrationObservation{ID: f.ID, Auditor: aud, Verdict: v, BundleDigest: b.BundleDigest, PolicyDigest: "policy-digest", PromptVersion: CrossAuditPromptVersion, PromptDigest: IssueAuditContentDigest(CrossAuditSystemPrompt), DurationNanos: int64(i+1) * 100, Usage: AuditTokenCost{InputTokens: 10, OutputTokens: 2, CostMicrosUSD: 3}, EvidenceTruncated: i == 3})
		}
	}
	return truth, obs
}

func TestCrossAuditCalibrationReproducesMetricsAndDisagreement(t *testing.T) {
	truth, obs := calibrationFixture(t)
	r, err := BuildCrossAuditCalibrationReport(truth, obs)
	if err != nil {
		t.Fatal(err)
	}
	if r.Schema != CrossAuditCalibrationSchema || len(r.Arms) != 2 || len(r.Disagreements) != 1 {
		t.Fatalf("report shape %+v", r)
	}
	var perfect, miss CalibrationArm
	for _, a := range r.Arms {
		if a.Auditor.Model == "m1" {
			perfect = a
		} else {
			miss = a
		}
	}
	if perfect.Metrics.Precision != 1 || perfect.Metrics.Recall != 1 || perfect.Metrics.FalsePositiveRate != 0 {
		t.Fatalf("perfect metrics %+v", perfect.Metrics)
	}
	if miss.Metrics.Confusion.FalseNegative != 1 || miss.Metrics.Recall >= 1 {
		t.Fatalf("miss metrics %+v", miss.Metrics)
	}
	if r.Disagreements[0].Different != 1 || r.Disagreements[0].Compared != 4 {
		t.Fatalf("disagreement %+v", r.Disagreements[0])
	}
	if perfect.Metrics.InputTokens != 40 || perfect.Metrics.CostMicrosUSD != 12 || perfect.Metrics.LatencyP95Nanos != 400 {
		t.Fatalf("cost/latency %+v", perfect.Metrics)
	}
}

func TestCrossAuditCalibrationRejectsMismatchedProvenance(t *testing.T) {
	truth, obs := calibrationFixture(t)
	obs[0].BundleDigest = "tampered"
	_, err := BuildCrossAuditCalibrationReport(truth, obs)
	if err == nil || !strings.Contains(err.Error(), "provenance mismatch") {
		t.Fatalf("mismatch error=%v", err)
	}
}

func TestCrossAuditCalibrationRejectsMixedPromptVersions(t *testing.T) {
	truth, obs := calibrationFixture(t)
	obs[1].PromptVersion = "other"
	_, err := BuildCrossAuditCalibrationReport(truth, obs)
	if err == nil || !strings.Contains(err.Error(), "prompt provenance") {
		t.Fatalf("mixed provenance error=%v", err)
	}
}
