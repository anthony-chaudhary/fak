package main

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
)

// TestDefaultLaunchSpawnBrokerHeldByContainment is the LIVE-path witness that
// the breaker bites: a capability-valid spawn is DENIED before the capability
// broker runs when the containment gate returns a refusal, and the deny reason
// carries the closed verdict so the hold is auditable. This is the wiring that
// turns DecideContainment from a reportable API into an enforced admission gate.
func TestDefaultLaunchSpawnBrokerHeldByContainment(t *testing.T) {
	attempt := newLaunchBrokerAttempt("dispatch-surface", "claude",
		[]string{"claude", "--dangerously-skip-permissions"}, map[string]string{}, t.TempDir())

	orig := launchContainmentGate
	t.Cleanup(func() { launchContainmentGate = orig })

	// Gate returns a storm refusal -> the launch is held, capability never consulted.
	launchContainmentGate = func(surface string, live int) toolprocgate.ContainmentDecision {
		return toolprocgate.ContainmentDecision{Verdict: toolprocgate.ContainBreakerOpen, Admit: false}
	}
	held := defaultLaunchSpawnBroker(attempt)
	if held.Allow {
		t.Fatalf("expected containment to HOLD the spawn, got Allow=true")
	}
	if held.Reason != "CONTAINMENT_BREAKER_OPEN" {
		t.Fatalf("deny reason = %q, want CONTAINMENT_BREAKER_OPEN", held.Reason)
	}
	if held.SpawnGrant.GrantID != "" {
		t.Fatalf("held spawn must not carry a capability grant, got %q", held.SpawnGrant.GrantID)
	}

	// Gate admits -> the capability broker runs and grants as before.
	launchContainmentGate = func(surface string, live int) toolprocgate.ContainmentDecision {
		return toolprocgate.ContainmentDecision{Verdict: toolprocgate.ContainAdmit, Admit: true}
	}
	ok := defaultLaunchSpawnBroker(attempt)
	if !ok.Allow {
		t.Fatalf("expected admit to fall through to a capability grant, got deny %q", ok.Reason)
	}
	if ok.SpawnGrant.GrantID == "" {
		t.Fatalf("admitted spawn should carry a capability grant id")
	}
}

// TestDefaultLaunchContainmentGateSurfacesVerdict checks the wired default gate
// end to end: passing the attempt's surface through DefaultContainmentPolicy
// against a real (empty) history admits, so the breaker only ever ADDS a
// refusal off recorded faults — it never wedges a clean launch.
func TestDefaultLaunchContainmentGateAdmitsCleanHistory(t *testing.T) {
	dec := defaultLaunchContainmentGate("dispatch-surface", 0)
	if !dec.Admit {
		t.Fatalf("clean history should admit, got %s", dec.Verdict)
	}
}

// TestReadConsoleFaultJournalForContainmentFailOpen proves the live reader
// folds every read problem to an empty history (admit), never a hard error:
// a missing file and a drifted row both yield nil, so a broken journal can
// only fail-open, never wedge dispatch.
func TestReadConsoleFaultJournalForContainmentFailOpen(t *testing.T) {
	if got := readConsoleFaultJournalForContainment(filepath.Join(t.TempDir(), "absent.jsonl")); got != nil {
		t.Fatalf("missing journal should read as nil, got %d events", len(got))
	}
	if got := parseConsoleFaultJournalLenient(strings.NewReader(`{"class":"NOPE","at_unix_ms":1}` + "\n")); got != nil {
		t.Fatalf("drifted journal should read as nil, got %d events", len(got))
	}
}
