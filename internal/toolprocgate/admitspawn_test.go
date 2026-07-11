package toolprocgate

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

// TestAdmitSpawnLiveContainment proves the closed loop: a console fault recorded
// through ExitConsoleFault is remembered, and a subsequent AdmitSpawn onto the
// same crashing surface is refused (QUARANTINE_SURFACE) rather than admitted —
// the protection the crash-observability alone did not provide.
func TestAdmitSpawnLiveContainment(t *testing.T) {
	s := NewSupervisor(toolproc.Config{})
	pol := DefaultContainmentPolicy()
	now := int64(1_000_000)

	// A clean surface admits.
	if dec := s.AdmitSpawn(ContainmentRequest{Surface: "conpty-A", NowMS: now}, pol); !dec.Admit {
		t.Fatalf("clean surface should admit, got %q", dec.Verdict)
	}

	// Two child calls fault on conpty-A (a re-crash loop).
	for i, id := range []string{"call-1", "call-2"} {
		if err := s.Spawn(id, "Bash", "sess-1", 0, 0, now-int64(2000-i*100), func() {}); err != nil {
			t.Fatalf("spawn %s: %v", id, err)
		}
		if _, err := s.ExitConsoleFault(id, now-int64(1500-i*100), ConsoleHostFailFast, ConsoleSurfacePTY, "0xE9 FailFast"); err != nil {
			t.Fatalf("ExitConsoleFault %s: %v", id, err)
		}
	}

	// The next spawn onto conpty-A must now be contained, not admitted.
	dec := s.AdmitSpawn(ContainmentRequest{Surface: string(ConsoleSurfacePTY), NowMS: now}, pol)
	if dec.Admit {
		t.Fatalf("spawn onto a re-crashing surface should be refused, got ADMIT")
	}
	if dec.Verdict != ContainQuarantineSurface {
		t.Fatalf("verdict = %q, want QUARANTINE_SURFACE", dec.Verdict)
	}

	// A DIFFERENT, clean surface still admits — containment is scoped, not a
	// fleet-wide freeze (only 1 session faulted, so the breaker stays closed).
	if dec := s.AdmitSpawn(ContainmentRequest{Surface: "conpty-Z", NowMS: now}, pol); !dec.Admit {
		t.Fatalf("a clean sibling surface should still admit, got %q", dec.Verdict)
	}
}
