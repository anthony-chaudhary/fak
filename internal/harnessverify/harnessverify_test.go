package harnessverify

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
)

func TestVerifyCleanRuntime(t *testing.T) {
	asset := harnesscompose.EffectiveAsset{Kind: "tool", ID: "search", Source: "repo", Value: "kb"}
	lock := harnessresolve.Lock{ID: "sha256:lock", Assets: []harnesscompose.EffectiveAsset{asset}}
	report, err := Verify(lock, Observation{Schema: ObservationSchema, LockID: lock.ID, RunID: "run", Capabilities: []Capability{{Capability: "tool:search", Source: "repo", Value: "kb"}}})
	if err != nil {
		t.Fatal(err)
	}
	if report.Verdict != "verified" || report.Matched != 1 || report.Added+report.Changed+report.Omitted != 0 {
		t.Fatalf("report=%+v", report)
	}
}

func TestVerifyRejectsAmbiguousObservation(t *testing.T) {
	lock := harnessresolve.Lock{ID: "sha256:lock"}
	cases := []Observation{
		{Schema: "wrong", LockID: lock.ID, RunID: "run"},
		{Schema: ObservationSchema, LockID: lock.ID},
		{Schema: ObservationSchema, LockID: lock.ID, RunID: "run", Capabilities: []Capability{{Capability: "skill"}}},
		{Schema: ObservationSchema, LockID: lock.ID, RunID: "run", Capabilities: []Capability{{Capability: "tool:x"}, {Capability: "tool:x"}}},
	}
	for _, observation := range cases {
		if _, err := Verify(lock, observation); err == nil || strings.TrimSpace(err.Error()) == "" {
			t.Fatalf("observation=%+v err=%v", observation, err)
		}
	}
}
