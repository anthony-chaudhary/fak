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

func TestVerifyDeviationsAndRender(t *testing.T) {
	lock := harnessresolve.Lock{
		ID: "sha256:lock-dev",
		Assets: []harnesscompose.EffectiveAsset{
			{Kind: "tool", ID: "search", Source: "repo", Value: "kb"},
			{Kind: "tool", ID: "bash", Source: "repo", Value: "safe", Boundary: "sandbox", Grants: []string{"read"}, Denies: []string{"net"}},
			{Kind: "workflow", ID: "lint", Source: "repo"},
		},
	}
	obs := Observation{
		Schema: ObservationSchema,
		LockID: lock.ID,
		RunID:  "run-dev",
		Capabilities: []Capability{
			{Capability: "tool:search", Source: "repo", Value: "kb"},
			{Capability: "tool:bash", Source: "override", Value: "unsafe", Boundary: "host", Grants: []string{"write"}, Denies: []string{"none"}},
			{Capability: "tool:extra", Source: "agent"},
		},
		Events: []Event{
			{Kind: "gate", Capability: "tool:extra", Source: "agent", Outcome: "deny"},
		},
	}

	report, err := Verify(lock, obs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if report.Verdict != "deviation" {
		t.Fatalf("expected deviation verdict, got %s", report.Verdict)
	}
	if report.Matched != 1 || report.Changed != 1 || report.Added != 1 || report.Omitted != 1 {
		t.Fatalf("unexpected counts: %+v", report)
	}

	rendered := Render(report)
	if !strings.Contains(rendered, "HARNESS VERIFY RUN | DEVIATION") {
		t.Fatalf("expected header in rendered output:\n%s", rendered)
	}
	if !strings.Contains(rendered, "changed source,value,boundary,grants,denies") {
		t.Fatalf("expected diff string in rendered output:\n%s", rendered)
	}
	if !strings.Contains(rendered, "runtime decisions:") {
		t.Fatalf("expected runtime decisions in rendered output:\n%s", rendered)
	}
}
