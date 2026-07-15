package executionroute

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

func TestRouteChoosesHarnessModelAndSessionTogether(t *testing.T) {
	decision, err := Route(Request{
		HarnessCandidates: []string{"openai-generic", "codex"},
		Harness:           HarnessRequirements{Rotatable: true},
		Model:             modelroute.Subject{Aspect: modelroute.AspectToolCall, Tool: "write_repository"},
		Session:           SessionSubject{ID: "session-7", PreserveContinuity: true, ContextUtilization: .91},
	}, harnessprofile.Builtins(), modelroute.DefaultManifest())
	if err != nil {
		t.Fatal(err)
	}
	if got := decision.Harness.Profile.Name; got != "codex" {
		t.Fatalf("harness=%q want codex", got)
	}
	if got := decision.Model.Plan.Primary(); got == "" {
		t.Fatal("model plan has no primary model")
	}
	if got := decision.Session.Action; got != SessionCompactResume {
		t.Fatalf("session=%q want %q", got, SessionCompactResume)
	}
}

func TestRouteHarnessRequirementsCanSelectWire(t *testing.T) {
	decision, err := Route(Request{
		Harness: HarnessRequirements{Wire: harnessprofile.WireAnthropic, Repoint: harnessprofile.RepointSettingsFile},
		Model:   modelroute.Subject{Aspect: modelroute.AspectRequest},
	}, harnessprofile.Builtins(), modelroute.DefaultManifest())
	if err != nil {
		t.Fatal(err)
	}
	if got := decision.Harness.Profile.Name; got != "claude" {
		t.Fatalf("harness=%q want claude", got)
	}
	if got := decision.Session.Action; got != SessionStart {
		t.Fatalf("session=%q want start", got)
	}
}

func TestRouteRejectsUnsatisfiedHarnessRequirements(t *testing.T) {
	_, err := Route(Request{
		HarnessCandidates: []string{"openai-generic"},
		Harness:           HarnessRequirements{Rotatable: true},
		Model:             modelroute.Subject{Aspect: modelroute.AspectRequest},
	}, harnessprofile.Builtins(), modelroute.DefaultManifest())
	if err == nil {
		t.Fatal("expected unsatisfied harness route to fail")
	}
}

func TestRouteForksPortableSession(t *testing.T) {
	decision, err := Route(Request{
		Model:   modelroute.Subject{Aspect: modelroute.AspectRequest},
		Session: SessionSubject{ID: "old", Portable: true},
	}, harnessprofile.Builtins(), modelroute.DefaultManifest())
	if err != nil {
		t.Fatal(err)
	}
	if got := decision.Session.Action; got != SessionFork {
		t.Fatalf("session=%q want fork", got)
	}
}
